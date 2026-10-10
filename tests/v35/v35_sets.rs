//! Image sets (`{f(n) : n ∈ ℤ}`, SymPy's `ImageSet(Lambda(n, f),
//! Integers)`) and the solution sets built from them.  Before, `solve_as_set`
//! of a periodic equation returned its principal solutions as a finite set
//! — `sin(x).solve_as_set(x)` was `{0, π}`, a claim that there are no other
//! solutions — and `reduce_inequalities` of an unbounded periodic equation a
//! `ConditionSet`.  Each test says what was wrong before and cites its
//! oracle: SymPy 1.14 (`solveset`, `ImageSet` operations), and members
//! substituted back.

use symplex::prelude::*;

fn parse(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

fn set_of(ctx: &Context, x: &Ex, s: &str) -> SetEx {
    parse(ctx, s).solve_as_set(x)
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

/// The image sets of `set` (a union, or one image set), as `(var, body)`.
fn families(ctx: &Context, set: &SetEx) -> Vec<(Ex, Ex)> {
    use symplex::tree::ExprTree;
    let mut out = Vec::new();
    let mut stack = vec![set.as_ex().to_tree()];
    while let Some(t) = stack.pop() {
        match t {
            ExprTree::SetUnion { sets } => stack.extend(sets),
            ExprTree::SetComplement { set, .. } => stack.push(*set),
            ExprTree::ImageSet { var, body } => {
                out.push((ctx.from_tree(&var), ctx.from_tree(&body)))
            }
            _ => {}
        }
    }
    out
}

/// Every member `n = −3..3` of every family of `set` solves `e = 0`.
fn assert_members_are_roots(ctx: &Context, e: &Ex, x: &Ex, set: &SetEx) {
    let fams = families(ctx, set);
    assert!(!fams.is_empty(), "{set}");
    for (n, body) in &fams {
        for k in -3..=3 {
            let m = body.subs_i64(n, k).eval();
            assert_root(e, x, &m);
            assert_eq!(set.contains(&m), Some(true), "{m} ∈ {set}");
        }
    }
}

/// `sin(x).solve_as_set(x)` was `{0, π}` (the principal solutions, also for
/// `sin x = 1/2`: `{π/6, 5π/6}`), a false claim about the solution set.  SymPy
/// 1.14: `solveset(sin(x), x)` → `Union(ImageSet(Lambda(_n, 2*_n*pi),
/// Integers), ImageSet(Lambda(_n, 2*_n*pi + pi), Integers))`, which symplex
/// merges to `{n·π}`; `solveset(sin(x) - 1/2, x)` → `2nπ + π/6 ∪ 2nπ +
/// 5π/6`; `solveset(cos(x), x)` → `2nπ + π/2 ∪ 2nπ + 3π/2` (= `nπ + π/2`);
/// `solveset(tan(x), x)` → `ImageSet(Lambda(_n, _n*pi), Integers)`;
/// `solveset(sin(3*x)/sin(x), x)` → `2nπ + π/3, 2π/3, 4π/3, 5π/3` (= `nπ ±
/// π/3`); `solveset(sin(x)*(x-1), x)` → `Union({1}, 2nπ, 2nπ + π)`.
#[test]
fn periodic_equations_have_image_sets_as_solution_sets() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cases: &[(&str, &str)] = &[
        ("sin(x)", "ImageSet(Lambda(_n, _n*pi), Integers)"),
        (
            "sin(x) - 1/2",
            "ImageSet(Lambda(_n, 2*_n*pi + 1/6*pi), Integers) ∪ ImageSet(Lambda(_n, 2*_n*pi + 5/6*pi), Integers)",
        ),
        ("cos(x)", "ImageSet(Lambda(_n, _n*pi + 1/2*pi), Integers)"),
        ("tan(x)", "ImageSet(Lambda(_n, _n*pi), Integers)"),
        (
            "sin(3*x)/sin(x)",
            "ImageSet(Lambda(_n, _n*pi + 1/3*pi), Integers) ∪ ImageSet(Lambda(_n, _n*pi + 2/3*pi), Integers)",
        ),
        (
            "sin(x)*(x - 1)",
            "{1} ∪ ImageSet(Lambda(_n, _n*pi), Integers)",
        ),
        ("x*sin(x)", "ImageSet(Lambda(_n, _n*pi), Integers)"),
        (
            "sin(x)^2 - sin(x)",
            "ImageSet(Lambda(_n, _n*pi), Integers) ∪ ImageSet(Lambda(_n, 2*_n*pi + 1/2*pi), Integers)",
        ),
        (
            "cos(2*x) - cos(x)",
            "ImageSet(Lambda(_n, 2/3*_n*pi), Integers)",
        ),
    ];
    for (src, want) in cases {
        let e = parse(&ctx, src);
        let set = e.solve_as_set(&x);
        assert_eq!(set.to_string(), *want, "{src}");
        assert_members_are_roots(&ctx, &e, &x, &set);
    }
    let sin = set_of(&ctx, &x, "sin(x)");
    assert_eq!(sin.contains(&parse(&ctx, "5*pi")), Some(true));
    assert_eq!(sin.contains(&parse(&ctx, "-7*pi")), Some(true));
    assert_eq!(sin.contains(&ctx.int(1)), Some(false));
    // Non-periodic equations keep their finite sets.
    assert_eq!(set_of(&ctx, &x, "x^2 - 1").to_string(), "{-1, 1}");
    assert_eq!(set_of(&ctx, &x, "ln(x) - 1").to_string(), "{E}");
}

/// The members of a family that are poles are left out.  Before:
/// `sin(x)/x` gave `{π}` (and `solve_general` refused: the family `2nπ`
/// holds the pole `0`), `sin(x)/(x² − π²)` `{0}`, `sin(x)/(x − 10π)` `{0,
/// π}`, `tan(x)/x` a `ConditionSet`.  SymPy 1.14: `solveset(sin(x)/x, x)` →
/// `Union(Complement(ImageSet(Lambda(_n, 2*_n*pi), Integers), {0}),
/// Complement(ImageSet(Lambda(_n, 2*_n*pi + pi), Integers), {0}))`,
/// `solveset(tan(x)/x, x)` → `Complement(ImageSet(Lambda(_n, _n*pi),
/// Integers), {0})`, `solveset(sin(x)/(x**2 - pi**2), x)` → the families
/// minus `{-pi, pi}`, `solveset(sin(x)/(x - 10*pi), x)` → minus `{10*pi}`.
#[test]
fn poles_on_a_family_are_excluded() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cases: &[(&str, &str)] = &[
        ("sin(x)/x", "ImageSet(Lambda(_n, _n*pi), Integers) \\ {0}"),
        ("tan(x)/x", "ImageSet(Lambda(_n, _n*pi), Integers) \\ {0}"),
        (
            "sin(x)/(x^2 - pi^2)",
            "ImageSet(Lambda(_n, _n*pi), Integers) \\ {-pi, pi}",
        ),
        (
            "sin(x)/(x - 10*pi)",
            "ImageSet(Lambda(_n, _n*pi), Integers) \\ {10*pi}",
        ),
        (
            "(1 - cos(x))/x",
            "ImageSet(Lambda(_n, 2*_n*pi), Integers) \\ {0}",
        ),
    ];
    for (src, want) in cases {
        let e = parse(&ctx, src);
        let set = e.solve_as_set(&x);
        assert_eq!(set.to_string(), *want, "{src}");
    }
    let s = set_of(&ctx, &x, "sin(x)/x");
    assert_eq!(s.contains(&ctx.int(0)), Some(false));
    assert_eq!(s.contains(&ctx.pi()), Some(true));
    assert_eq!(s.contains(&parse(&ctx, "-3*pi")), Some(true));
    // x/sin(x) = 0 has no solution: its root 0 is a pole.
    assert_eq!(set_of(&ctx, &x, "x/sin(x)").to_string(), "EmptySet");
}

/// The set is over the declared domain of the variable, ℂ for a plain
/// symbol.  Before, `solve`'s real-variable conventions made it miss the
/// non-real solutions: `exp(x) − 1` gave `{0}`, `exp(2x) − 3eˣ + 2` `{0,
/// ln 2}`, `sin(x) − 2` and `cos(x) + 2` `EmptySet`, `|x| − 1` `{−1, 1}`
/// (the unit circle solves it), `x·eˣ − 1` `{W(1)}` (every branch `W_k(1)`
/// does).  SymPy 1.14: `solveset(exp(x) - 1, x)` → `ImageSet(Lambda(_n,
/// 2*_n*I*pi), Integers)`; `solveset(exp(2*x) - 3*exp(x) + 2, x)` →
/// `2nπi ∪ 2nπi + log(2)`; `solveset(sin(x) - 2, x)` → the families
/// `2nπ + asin(2)` and `2nπ + π − asin(2)`; `solveset(cos(x) + 2, x)` →
/// `2nπ ± acos(−2)`;
/// `solveset(Abs(x) - 1, x)` raises "Absolute values cannot be inverted in
/// the complex domain"; `solveset(x*exp(x) - 1, x)` →
/// `ConditionSet(x, Eq(x*exp(x) - 1, 0), Complexes)`.  For a variable
/// declared real the sets are the real ones.
#[test]
fn the_set_is_over_the_complex_numbers_for_a_plain_symbol() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cases: &[(&str, &str)] = &[
        ("exp(x) - 1", "ImageSet(Lambda(_n, 2*_n*pi*I), Integers)"),
        (
            "exp(2*x) - 3*exp(x) + 2",
            "ImageSet(Lambda(_n, 2*_n*pi*I), Integers) ∪ ImageSet(Lambda(_n, 2*_n*pi*I + ln(2)), Integers)",
        ),
        (
            "sin(x) - 2",
            "ImageSet(Lambda(_n, 2*_n*pi - asin(2) + pi), Integers) ∪ ImageSet(Lambda(_n, 2*_n*pi + asin(2)), Integers)",
        ),
        (
            "cos(x) + 2",
            "ImageSet(Lambda(_n, 2*_n*pi - acos(-2)), Integers) ∪ ImageSet(Lambda(_n, 2*_n*pi + acos(-2)), Integers)",
        ),
    ];
    for (src, want) in cases {
        let e = parse(&ctx, src);
        let set = e.solve_as_set(&x);
        assert_eq!(set.to_string(), *want, "{src}");
        assert_members_are_roots(&ctx, &e, &x, &set);
    }
    let i = ctx.i_unit();
    let exp1 = set_of(&ctx, &x, "exp(x) - 1");
    assert_eq!(exp1.contains(&(4 * ctx.pi() * &i)), Some(true));
    assert_eq!(exp1.contains(&(ctx.pi() * &i)), Some(false));
    for src in ["abs(x) - 1", "x*exp(x) - 1"] {
        let s = set_of(&ctx, &x, src).to_string();
        assert!(s.starts_with("ConditionSet"), "{src}: {s}");
    }
    // A real variable: the real solution sets.
    let r = ctx.symbol_with("r", &[Assumption::Real]).unwrap();
    let real_cases: &[(&str, &str)] = &[
        ("exp(r) - 1", "{0}"),
        ("sin(r) - 2", "EmptySet"),
        ("abs(r) - 1", "{-1, 1}"),
        ("sin(r)", "ImageSet(Lambda(_n, _n*pi), Integers)"),
    ];
    for (src, want) in real_cases {
        assert_eq!(
            parse(&ctx, src).solve_as_set(&r).to_string(),
            *want,
            "{src}"
        );
    }
}

/// The node itself: canonical form, printing, parsing, trees,
/// substitution.  SymPy 1.14: `imageset(Lambda(n, -2*n*pi + 5*pi/2),
/// S.Integers)` → `ImageSet(Lambda(n, 2*pi*n + pi/2), Integers)` (the
/// offset modulo the step); `latex(ImageSet(Lambda(n, 2*n*pi), S.Integers))`
/// → `\left\{2 \pi n\; \middle|\; n \in \mathbb{Z}\right\}`;
/// `srepr` → `ImageSet(Lambda(…), Integers)`.
#[test]
fn image_set_node_prints_parses_and_substitutes() {
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let pi = ctx.pi();
    let evens = (2 * &n * &pi).image_set(&n).unwrap();
    let shown = evens.to_string();
    assert_eq!(shown, "ImageSet(Lambda(_n, 2*_n*pi), Integers)");
    // display → parse is the identity
    assert_eq!(ctx.parse(&shown).unwrap(), evens.as_ex());
    let fresh = Context::new();
    assert_eq!(fresh.parse(&shown).unwrap().to_string(), shown);
    // tree / JSON round trip
    assert_eq!(ctx.from_tree(&evens.as_ex().to_tree()), evens.as_ex());
    assert_eq!(
        ctx.from_json(&evens.as_ex().to_json().unwrap()).unwrap(),
        evens.as_ex()
    );
    assert_eq!(
        evens.as_ex().to_srepr(),
        "ImageSet(Lambda(Symbol('_n'), Mul(Integer(2), Symbol('_n'), pi)), Integers)"
    );
    assert_eq!(
        evens.as_ex().to_latex(),
        r"\left\{2n \pi\; \middle|\; n \in \mathbb{Z}\right\}"
    );
    // canonical: positive step, offset reduced modulo the step
    let shifted = (-2 * &n * &pi + 5 * &pi / 2).image_set(&n).unwrap();
    assert_eq!(
        shifted.to_string(),
        "ImageSet(Lambda(_n, 2*_n*pi + 1/2*pi), Integers)"
    );
    assert_eq!(
        parse(&ctx, "ImageSet(Lambda(k, 3 - k), Integers)").to_string(),
        "ImageSet(Lambda(_n, _n), Integers)"
    );
    // free of the parameter: a finite set
    assert_eq!(ctx.int(7).image_set(&n).unwrap().to_string(), "{7}");
    assert!((&n + 1).image_set(&ctx.int(2)).is_err());
    assert!(ctx.parse("ImageSet(Lambda(n, n), Reals)").is_err());
    // substitution of a parameter goes back to the canonical form
    let a = ctx.symbol("a");
    let fam = (2 * &n * &pi + &a).image_set(&n).unwrap();
    assert_eq!(
        fam.as_ex().subs(&a, &(3 * &pi)).to_string(),
        "ImageSet(Lambda(_n, 2*_n*pi + pi), Integers)"
    );
    // the bound variable is not free, and substituting it does nothing
    assert!(
        !fam.as_ex()
            .free_symbols()
            .iter()
            .any(|s| s.to_string() == "_n")
    );
}

/// Membership solves `f(n) = x` for an integer `n`.  SymPy 1.14:
/// `ImageSet(Lambda(n, 2*n*pi), S.Integers).contains(2*pi)` → `True`,
/// `.contains(1)` → `False`, `.contains(2*k*pi)` with `k` integer →
/// `True`; `ImageSet(Lambda(n, 2*n*I*pi), S.Integers).contains(4*I*pi)` →
/// `True`.
#[test]
fn membership_in_an_image_set() {
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let pi = ctx.pi();
    let evens = (2 * &n * &pi).image_set(&n).unwrap();
    assert_eq!(evens.contains(&(2 * &pi)), Some(true));
    assert_eq!(evens.contains(&(-6 * &pi)), Some(true));
    assert_eq!(evens.contains(&ctx.int(0)), Some(true));
    assert_eq!(evens.contains(&ctx.int(1)), Some(false));
    assert_eq!(evens.contains(&pi), Some(false));
    assert_eq!(evens.contains(&ctx.int(2).sqrt()), Some(false));
    assert_eq!(evens.contains(&ctx.i_unit()), Some(false));
    // √(4π²) is 2π
    assert_eq!(evens.contains(&parse(&ctx, "sqrt(4*pi^2)")), Some(true));
    // symbolic elements: by the assumptions
    let k = ctx.symbol_with("k", &[Assumption::Integer]).unwrap();
    assert_eq!(evens.contains(&(2 * &k * &pi)), Some(true));
    assert_eq!(evens.contains(&ctx.symbol("x")), None);
    // a non-linear family: n² + 1
    let sq = (&n * &n + 1).image_set(&n).unwrap();
    assert_eq!(sq.contains(&ctx.int(10)), Some(true));
    assert_eq!(sq.contains(&ctx.int(3)), Some(false));
    assert_eq!(sq.is_empty(), Some(false));
}

/// Set operations with image sets.  SymPy 1.14:
/// `ImageSet(Lambda(n, 2*n*pi), S.Integers).intersect(Interval(-7, 7))` →
/// `{0, -2*pi, 2*pi}`; `ImageSet(Lambda(n, 2*n*I*pi),
/// S.Integers).intersect(S.Reals)` → `{0}`; `ImageSet(Lambda(n, n*pi),
/// S.Integers).intersect(FiniteSet(0, 1, pi))` → `{0, pi}`;
/// `ImageSet(Lambda(n, 2*n*pi), S.Integers).is_disjoint(Interval(1, 6))` →
/// `True`.  (SymPy keeps `Union` of `2nπ` and `2nπ + π` and the point `1` in
/// `Complement(ImageSet(n*pi), {1})`; symplex merges and drops it.)
#[test]
fn intersections_unions_and_complements_with_image_sets() {
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let pi = ctx.pi();
    let evens = (2 * &n * &pi).image_set(&n).unwrap();
    let odds = (2 * &n * &pi + &pi).image_set(&n).unwrap();
    let window = ctx.interval(&ctx.int(-7), &ctx.int(7), IntervalKind::Closed);
    assert_eq!(
        evens.intersection(&window).simplify().to_string(),
        "{0, -2*pi, 2*pi}"
    );
    let open = ctx.interval(&ctx.int(0), &(2 * &pi), IntervalKind::Open);
    assert_eq!(evens.intersection(&open).simplify().to_string(), "EmptySet");
    let half = ctx.interval(&ctx.int(0), &(2 * &pi), IntervalKind::RightOpen);
    assert_eq!(evens.intersection(&half).simplify().to_string(), "{0}");
    let imag = (2 * &n * &pi * ctx.i_unit()).image_set(&n).unwrap();
    assert_eq!(
        imag.intersection(&ctx.reals()).simplify().to_string(),
        "{0}"
    );
    assert_eq!(
        evens.intersection(&ctx.reals()).simplify().to_string(),
        evens.to_string()
    );
    let multiples = (&n * &pi).image_set(&n).unwrap();
    let pts = ctx.finite_set(&[ctx.int(0), ctx.int(1), pi.clone()]);
    assert_eq!(
        multiples.intersection(&pts).simplify().to_string(),
        "{0, pi}"
    );
    let six = ctx.interval(&ctx.int(1), &ctx.int(6), IntervalKind::Closed);
    assert_eq!(evens.is_disjoint(&six), Some(true));
    assert_eq!(evens.is_disjoint(&window), Some(false));
    assert_eq!(evens.intersection(&six).is_empty(), Some(true));
    // unions merge commensurable families, absorb member points
    assert_eq!(
        evens.union(&odds).simplify().to_string(),
        multiples.to_string()
    );
    let zero = ctx.finite_set(&[ctx.int(0), ctx.int(1)]);
    assert_eq!(
        multiples.union(&zero).simplify().to_string(),
        "{1} ∪ ImageSet(Lambda(_n, _n*pi), Integers)"
    );
    let quarter = (2 * &n * &pi + &pi / 2).image_set(&n).unwrap();
    let three_quarter = (2 * &n * &pi + 3 * &pi / 2).image_set(&n).unwrap();
    assert_eq!(
        evens
            .union(&odds)
            .union(&quarter)
            .union(&three_quarter)
            .simplify()
            .to_string(),
        "ImageSet(Lambda(_n, 1/2*_n*pi), Integers)"
    );
    // complements forget the points that are not members
    let one = ctx.finite_set(&[ctx.int(1)]);
    assert_eq!(
        multiples.difference(&one).to_string(),
        multiples.to_string()
    );
    let both = ctx.finite_set(&[ctx.int(1), pi.clone()]);
    assert_eq!(
        multiples.difference(&both).to_string(),
        "ImageSet(Lambda(_n, _n*pi), Integers) \\ {pi}"
    );
    assert_eq!(multiples.difference(&both).contains(&pi), Some(false));
    // as a condition: x = n·π for an integer n  ⇔  sin(π·x/π) = 0
    let x = ctx.symbol("x");
    assert_eq!(
        multiples.to_condition(&x).unwrap().to_string(),
        "sin(x) == 0"
    );
}

/// `reduce_inequalities` of a periodic equation with no bounds was the
/// `ConditionSet(x, sin(x) == 0)`; it is the real image set now, and with
/// bounds the members listed agree with the earlier listing.  Provably
/// unsatisfiable equations were errors ("could not solve the equation"):
/// `cos(x) = −2`, and `x/sin(x) = 0` (its root `0` is a pole).  SymPy 1.14:
/// `solveset(sin(x), x, S.Reals)` → `Union(ImageSet(Lambda(_n, 2*_n*pi),
/// Integers), ImageSet(Lambda(_n, 2*_n*pi + pi), Integers))`,
/// `solveset(sin(x), x, Interval(-4, 4))` → `{0, -pi, pi}`,
/// `reduce_inequalities(Eq(cos(x), -2))` → `False` (SymPy raises
/// `NotImplementedError` for `Eq(x/sin(x), 0)`).
#[test]
fn reduce_inequalities_with_image_sets() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let solve = |s: &str| ctx.parse_bool(s).unwrap().solve_for(&x).unwrap();
    assert_eq!(
        solve("sin(x) == 0").to_string(),
        "ImageSet(Lambda(_n, _n*pi), Integers)"
    );
    // over the reals: the real members of complex families
    assert_eq!(solve("exp(x) == 1").to_string(), "{0}");
    assert_eq!(
        solve("sin(x) == 0 & x > -4 & x < 4").to_string(),
        "{0, -pi, pi}"
    );
    assert_eq!(
        solve("sin(x) == 0 & x > 0").to_string(),
        "(0, oo) ∩ ImageSet(Lambda(_n, _n*pi), Integers)"
    );
    assert_eq!(
        solve("sin(x)/x == 0 & x > -7 & x < 7").to_string(),
        "{-pi, -2*pi, 2*pi, pi}"
    );
    assert_eq!(solve("cos(x) == -2").to_string(), "EmptySet");
    assert_eq!(solve("x/sin(x) == 0").to_string(), "EmptySet");
}

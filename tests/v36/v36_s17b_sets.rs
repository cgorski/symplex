//! Tests of session 17, wave 18 (s17b_sets): set differences, unions and inequalities with non-real members.
//! Every test says what was wrong before and cites its oracle.
//!
//! One root: an interval, the real line `ℝ = (−∞, ∞)` and every set in
//! normal form hold real numbers only, so containing, absorbing or removing
//! an element needs the element proved real.  Up to 0.59 the set evaluator
//! gave `UniversalSet` and `ℝ` one value and took every operand next to
//! "the whole line" for a set of real numbers, and the `|f| rel k` shortcut
//! of the inequality solver took `k` for a real bound.  The oracle of every
//! test is the definition (`i ∉ ℝ`; an order relation holds only between
//! real numbers); SymPy 1.14 is quoted as a second opinion.

#![allow(unused_imports)]
use symplex::prelude::*;

fn show(s: &SetEx) -> String {
    format!("{s}")
}

fn imaginary_unit(ctx: &Context) -> Ex {
    ctx.int(-1).sqrt()
}

/// `{i} \ ℝ` is `{i}`: removing the real numbers removes no member that is
/// not real.
///
/// Before (0.59.0): `{I}.difference(reals)` was `EmptySet`, and so were
/// `{1, I} \ ℝ`, `{x} \ ℝ` for a symbol without assumptions, `{2kπi} \ ℝ`,
/// `{zoo} \ ℝ` and `UniversalSet \ ℝ` — `A \ B` was `∅` whenever `B`
/// evaluated to the whole line.
///
/// Now a member is removed only when it is proved real; one that is
/// certainly not real stays, and an undecided one stays under the
/// complement (the expression is not simplified).
///
/// Oracle: the definition, `A \ B = {z ∈ A : z ∉ B}` with `i ∉ ℝ`.  SymPy
/// 1.14: `FiniteSet(I) - S.Reals` is `{I}`, `FiniteSet(1, I) - S.Reals` is
/// `{I}`, `FiniteSet(x) - S.Reals` is `Complement({x}, Reals)`,
/// `FiniteSet(r) - S.Reals` is `EmptySet` for a real `r`, `FiniteSet(1, I,
/// x) - S.Reals` is `Union(Complement({x}, Reals), {I})`, `FiniteSet(zoo) -
/// S.Reals` is `{zoo}`, `FiniteSet(oo) - S.Reals` is `{oo}`.
#[test]
fn a_difference_with_the_reals_keeps_the_members_that_are_not_real() {
    let ctx = Context::new();
    let i = imaginary_unit(&ctx);
    let x = ctx.symbol("x");
    let r = ctx.symbol_with("r", &[Assumption::Real]).unwrap();
    let p = ctx.symbol_with("p", &[Assumption::Positive]).unwrap();
    let k = ctx.symbol("k");
    let reals = ctx.reals();
    let one = ctx.int(1);

    // (1) of the task: before `EmptySet`.
    let d = ctx.finite_set(std::slice::from_ref(&i)).difference(&reals);
    assert_eq!(show(&d), "{I}");
    assert_eq!(d.contains(&i), Some(true));

    // certainly not real: kept (before `EmptySet` each)
    for (set, expected) in [
        (ctx.finite_set(&[one.clone(), i.clone()]), "{I}"),
        (
            ctx.finite_set(&[ctx.int(-2).sqrt(), ctx.int(3)]),
            "{sqrt(2)*I}",
        ),
        (ctx.finite_set(&[ctx.int(2).acos()]), "{acos(2)}"),
        (ctx.finite_set(&[&p * &i]), "{p*I}"),
        (ctx.finite_set(&[ctx.complex_infinity()]), "{zoo}"),
        (ctx.finite_set(&[ctx.infinity()]), "{oo}"),
        (ctx.finite_set(&[ctx.neg_infinity(), one.clone()]), "{-oo}"),
    ] {
        assert_eq!(show(&set.difference(&reals)), expected, "{set} \\ ℝ");
    }
    // not known to be real: the difference stays (before `EmptySet` each)
    for (set, expected) in [
        (ctx.finite_set(std::slice::from_ref(&x)), "{x} \\ (-oo, oo)"),
        (
            ctx.finite_set(&[x.clone(), i.clone(), one.clone()]),
            "{I} ∪ ({x} \\ (-oo, oo))",
        ),
        (
            ctx.finite_set(&[ctx.symbol("a").sqrt()]),
            "{sqrt(a)} \\ (-oo, oo)",
        ),
        (
            (&(&ctx.int(2) * &k) * &(&ctx.pi() * &i))
                .image_set(&k)
                .unwrap(),
            "ImageSet(Lambda(_n, 2*_n*pi*I), Integers) \\ (-oo, oo)",
        ),
        (ctx.universal_set(), "UniversalSet \\ (-oo, oo)"),
    ] {
        assert_eq!(show(&set.difference(&reals)), expected, "{set} \\ ℝ");
    }
    // proved real: removed, as before
    for set in [
        ctx.finite_set(std::slice::from_ref(&r)),
        ctx.finite_set(&[one.clone(), ctx.pi()]),
        ctx.finite_set(&[(&i * &ctx.pi()).exp()]),
        (&(&ctx.int(2) * &k) * &ctx.pi()).image_set(&k).unwrap(),
        ctx.interval(&x, &(&x + &one), IntervalKind::Closed),
        (&r - &r.cos()).solve_as_set(&r),
    ] {
        assert_eq!(show(&set.difference(&reals)), "EmptySet", "{set} \\ ℝ");
    }
    // The same against a part of the line: `{i} \ [0, 1]` is `{i}` (before:
    // left as `{I} \ [0, 1]`), and a real set loses nothing to `{i}`.
    let unit = ctx.interval(&ctx.int(0), &one, IntervalKind::Closed);
    assert_eq!(
        show(&ctx.finite_set(std::slice::from_ref(&i)).difference(&unit)),
        "{I}"
    );
    assert_eq!(
        show(&unit.difference(&ctx.finite_set(std::slice::from_ref(&i)))),
        "[0, 1]"
    );
    assert_eq!(
        show(
            &ctx.finite_set(&[x.clone(), i.clone(), one.clone()])
                .difference(&unit)
        ),
        "{I} ∪ ({x} \\ [0, 1])"
    );
}

/// `{i} ∪ ℝ` keeps `i`: an interval of real numbers does not absorb a point
/// it does not contain.
///
/// Before (0.59.0): `({I} ∪ reals).simplify()` was `(-oo, oo)`, and so were
/// `{x} ∪ ℝ` for a symbol without assumptions and `{2kπi} ∪ ℝ`; the lost
/// member was then "not known" to the result (`contains(I)` was `None`) and
/// `inf`/`sup`/`measure` of the union were `-oo`/`oo`/`oo`.
///
/// Now the line absorbs only what is proved real.
///
/// Oracle: the definition (`i ∈ {i} ∪ ℝ`, and `i ∉ ℝ`).  SymPy 1.14:
/// `Union(FiniteSet(I), S.Reals)` is `Union({I}, Reals)`, `Union(FiniteSet(x),
/// S.Reals)` is `Union({x}, Reals)`, `Union(FiniteSet(r), S.Reals)` is
/// `Reals` for a real `r`, `Union(FiniteSet(1, I), S.Reals)` is `Union({I},
/// Reals)`; `.sup` of such a union raises "The argument 'I' is not
/// comparable".
#[test]
fn a_union_with_the_reals_keeps_the_members_that_are_not_real() {
    let ctx = Context::new();
    let i = imaginary_unit(&ctx);
    let x = ctx.symbol("x");
    let r = ctx.symbol_with("r", &[Assumption::Real]).unwrap();
    let k = ctx.symbol("k");
    let reals = ctx.reals();
    let one = ctx.int(1);

    // (2) of the task: before `(-oo, oo)`.
    let u = ctx
        .finite_set(std::slice::from_ref(&i))
        .union(&reals)
        .simplify();
    assert_eq!(show(&u), "(-oo, oo) ∪ {I}");
    assert_eq!(u.contains(&i), Some(true), "before: None");
    assert_eq!(u.contains(&one), Some(true));
    assert_eq!(u.is_subset(&reals), Some(false));
    // no bounds for a set with a member off the line (before: -oo, oo, oo)
    assert!(u.inf().is_none() && u.sup().is_none() && u.measure().is_none());

    for (set, expected) in [
        (ctx.finite_set(&[one.clone(), i.clone()]), "(-oo, oo) ∪ {I}"),
        (ctx.finite_set(std::slice::from_ref(&x)), "(-oo, oo) ∪ {x}"),
        (
            ctx.finite_set(&[x.clone(), i.clone(), one.clone()]),
            "(-oo, oo) ∪ {x, I}",
        ),
        (
            ctx.finite_set(&[ctx.complex_infinity()]),
            "(-oo, oo) ∪ {zoo}",
        ),
        (ctx.finite_set(&[ctx.infinity()]), "(-oo, oo) ∪ {oo}"),
        (
            (&(&ctx.int(2) * &k) * &(&ctx.pi() * &i))
                .image_set(&k)
                .unwrap(),
            "(-oo, oo) ∪ ImageSet(Lambda(_n, 2*_n*pi*I), Integers)",
        ),
    ] {
        assert_eq!(show(&set.union(&reals).simplify()), expected, "{set} ∪ ℝ");
    }
    // proved real: absorbed, as before
    for set in [
        ctx.finite_set(std::slice::from_ref(&r)),
        ctx.finite_set(&[ctx.int(2).sqrt(), ctx.pi()]),
        (&(&ctx.int(2) * &k) * &ctx.pi()).image_set(&k).unwrap(),
        ctx.interval(&x, &(&x + &one), IntervalKind::Closed),
        (&r - &r.cos()).solve_as_set(&r),
    ] {
        assert_eq!(
            show(&set.union(&reals).simplify()),
            "(-oo, oo)",
            "{set} ∪ ℝ"
        );
    }
    // What the union holds on the line is still exact: `({i} ∪ ℝ) ∩ [0, 1]`.
    let unit = ctx.interval(&ctx.int(0), &one, IntervalKind::Closed);
    assert_eq!(show(&u.intersection(&unit).simplify()), "[0, 1]");
    assert_eq!(u.is_disjoint(&unit), Some(false));
    // The line written in two parts absorbs no more than the line.
    let left = ctx.interval(&ctx.neg_infinity(), &ctx.int(0), IntervalKind::LeftOpen);
    let right = ctx.interval(&ctx.int(0), &ctx.infinity(), IntervalKind::Open);
    let two = left.union(&right);
    let with_i = ctx
        .finite_set(std::slice::from_ref(&i))
        .union(&two)
        .simplify();
    assert_eq!(show(&with_i), "(-oo, oo) ∪ {I}");
    // The symmetric difference `{i} △ ℝ` is `ℝ ∪ {i}` (before: `ℝ \ {i}`,
    // without `i`).
    let sym = ctx
        .finite_set(std::slice::from_ref(&i))
        .symmetric_difference(&reals);
    assert_eq!(show(&sym), "(-oo, oo) ∪ {I}");
    assert_eq!(sym.contains(&i), Some(true));
}

/// `|t| + i > 0` holds for no real `t`: `|t| + i` is not real, and an order
/// relation holds only between real numbers.
///
/// Before (0.59.0): `(abs(t) + I).try_solve_gt(t)` was `(-I, oo) ∪ (-oo,
/// I)` — the shortcut `c·|a·t + b| + d rel 0 ⇔ |a·t + b| rel k` with `k =
/// −d/c` put `±k` on the real line without asking whether `k` is real
/// (`a` and `b` were asked).  `<` gave `(I, -I)`, and `|t| + acos(2) > 0`
/// gave `(-acos(2), oo) ∪ (-oo, acos(2))` (`acos(2) ≈ 1.317·i`).
///
/// Now `EmptySet`, by the convention of the inequality solvers (the module
/// documentation of `transforms/inequalities.rs`: "a point where `f` is
/// undefined or not real is never part of the solution"; `x + i > 0` was
/// already `EmptySet`).  A bound that is not known to be real (`√a` for a
/// plain `a`) is no longer taken either: the solve is refused, as `t − √a >
/// 0` always was.
///
/// Oracle: the definition — for every real `t`, `im(|t| + i) = 1 ≠ 0`.
/// SymPy 1.14: `solveset(Abs(t) + I > 0, t, S.Reals)` raises `TypeError:
/// Invalid comparison of non-real Abs(t) + I` (a refusal; never a set).
#[test]
fn an_absolute_value_is_not_compared_with_a_bound_that_is_not_real() {
    let ctx = Context::new();
    let i = imaginary_unit(&ctx);
    let t = ctx.symbol_with("t", &[Assumption::Real]).unwrap();
    let x = ctx.symbol("x");
    let a = ctx.symbol("a");

    // (3) of the task: before `(-I, oo) ∪ (-oo, I)`.
    let e = &t.abs() + &i;
    assert_eq!(show(&e.try_solve_gt(&t).unwrap()), "EmptySet");
    assert_eq!(
        show(&e.try_solve_lt(&t).unwrap()),
        "EmptySet",
        "before: (I, -I)"
    );
    assert_eq!(show(&e.try_solve_ge(&t).unwrap()), "EmptySet");
    assert_eq!(show(&e.try_solve_le(&t).unwrap()), "EmptySet");
    // without a literal `i`
    for e in [
        &t.abs() + &ctx.int(2).acos(),
        &t.abs() + &ctx.int(-2).sqrt(),
        &(&(&ctx.int(2) * &t.abs()) + &ctx.int(1)) + &i,
    ] {
        assert_eq!(show(&e.try_solve_gt(&t).unwrap()), "EmptySet", "{e} > 0");
        assert_eq!(show(&e.try_solve_le(&t).unwrap()), "EmptySet", "{e} <= 0");
    }
    // between two sides: `|t| > i` (before `(I, oo) ∪ (-oo, -I)`), and a
    // side that is not real whatever the difference of the sides is: `t + i
    // > i` (before `(0, oo)`), `|t| + i > i` (before `(-oo, 0) ∪ (0, oo)`).
    for c in [t.abs().gt(&i), (&t + &i).gt(&i), (&t.abs() + &i).gt(&i)] {
        assert_eq!(show(&c.solve_for(&t).unwrap()), "EmptySet", "{c}");
        assert_eq!(c.satisfiable(), Some(false), "{c}");
    }
    // A side with the variable that is real nowhere, or at one point: `t + i
    // > 2t + i` (before `(-oo, 0)`), and `i·t > i·t − 1`, which holds at `t =
    // 0` alone (before `UniversalSet`).
    let c = (&t + &i).gt(&(&(&ctx.int(2) * &t) + &i));
    assert_eq!(show(&c.solve_for(&t).unwrap()), "EmptySet");
    assert_eq!(c.satisfiable(), Some(false));
    let c = (&t * &i).gt(&(&(&t * &i) - &ctx.int(1)));
    assert_eq!(show(&c.solve_for(&t).unwrap()), "{0}");
    // An equation is about the difference alone: `t + i = i` is `{0}`.
    assert_eq!(show(&(&t + &i).eq_expr(&i).solve_for(&t).unwrap()), "{0}");

    // Real bounds as before; a plain parameter is read as real in an
    // inequality (`SetEx::reduce_inequalities`).
    let e = &(&(&ctx.int(2) * &t) + &ctx.int(1)).abs() - &ctx.int(3);
    assert_eq!(show(&e.try_solve_lt(&t).unwrap()), "(-2, 1)");
    assert_eq!(show(&e.try_solve_ge(&t).unwrap()), "(-oo, -2] ∪ [1, oo)");
    assert_eq!(show(&(&t.abs() - &x).try_solve_lt(&t).unwrap()), "(-x, x)");
    assert_eq!(
        show(&(&t.abs() + &ctx.int(1)).try_solve_gt(&t).unwrap()),
        "(-oo, oo)"
    );
    // A bound not known to be real is refused (before `(-sqrt(a),
    // sqrt(a))`, an interval between two imaginary numbers for `a < 0`).
    assert!((&t.abs() - &a.sqrt()).try_solve_lt(&t).is_err());
}

/// The neighbours with the same root in intersections and complements.
///
/// Before (0.59.0):
/// * `{i} ∩ ((−∞, 0] ∪ (0, ∞))` was `{I}` (the cut by the line was made only
///   next to an operand written as one interval; `{i} ∩ ℝ` was `∅`);
/// * `ℝ ∩ {x | x = cos x}` lost its `ℝ` for a symbol `x` without
///   assumptions, whose condition set holds the non-real solutions too;
/// * `{i}.absolute_complement().absolute_complement()` was `{I}` (it is `ℝ ∩
///   {i} = ∅`), and `{i}.absolute_complement()` stayed `(-oo, oo) \ {I}`.
///
/// Oracle: the definitions.  SymPy 1.14: `Intersection(FiniteSet(I),
/// S.Reals)` is `EmptySet`, `Intersection(FiniteSet(x), S.Reals)` stays,
/// `Intersection(ConditionSet(x, Eq(x - cos(x), 0), S.Complexes), S.Reals)`
/// is `ConditionSet(x, Eq(x - cos(x), 0), Reals)`, `S.Reals - FiniteSet(I)`
/// is `Reals`.
#[test]
fn the_real_line_cuts_and_removes_only_what_is_proved_real() {
    let ctx = Context::new();
    let i = imaginary_unit(&ctx);
    let x = ctx.symbol("x");
    let r = ctx.symbol_with("r", &[Assumption::Real]).unwrap();
    let reals = ctx.reals();
    let fi = ctx.finite_set(std::slice::from_ref(&i));
    let fx = ctx.finite_set(std::slice::from_ref(&x));
    let left = ctx.interval(&ctx.neg_infinity(), &ctx.int(0), IntervalKind::LeftOpen);
    let right = ctx.interval(&ctx.int(0), &ctx.infinity(), IntervalKind::Open);
    let two = left.union(&right);

    assert_eq!(
        show(&fi.intersection(&two).simplify()),
        "EmptySet",
        "before: {{I}}"
    );
    assert_eq!(show(&fi.intersection(&reals).simplify()), "EmptySet");
    assert_eq!(
        show(&fx.intersection(&two).simplify()),
        "(-oo, oo) ∩ {x}",
        "before: {{x}}"
    );
    assert_eq!(show(&fx.intersection(&reals).simplify()), "(-oo, oo) ∩ {x}");
    // a part of the line as well (before: left as `[0, 1] ∩ {I}`)
    let closed_unit = ctx.interval(&ctx.int(0), &ctx.int(1), IntervalKind::Closed);
    assert_eq!(show(&fi.intersection(&closed_unit).simplify()), "EmptySet");
    assert_eq!(
        show(
            &ctx.finite_set(&[x.clone(), i.clone()])
                .intersection(&closed_unit)
                .simplify()
        ),
        "[0, 1] ∩ {x}"
    );

    let roots_x = (&x - &x.cos()).solve_as_set(&x);
    let roots_r = (&r - &r.cos()).solve_as_set(&r);
    assert_eq!(
        show(&roots_x.intersection(&reals).simplify()),
        "ConditionSet(x, x - cos(x) == 0) ∩ (-oo, oo)",
        "before: without the line"
    );
    assert_eq!(
        show(&roots_r.intersection(&reals).simplify()),
        "ConditionSet(r, r - cos(r) == 0)"
    );

    // complements in ℝ
    assert_eq!(show(&fi.absolute_complement()), "(-oo, oo)");
    assert_eq!(
        show(&fi.absolute_complement().absolute_complement()),
        "EmptySet"
    );
    assert_eq!(show(&fx.absolute_complement()), "(-oo, oo) \\ {x}");
    assert_eq!(
        show(&fx.absolute_complement().absolute_complement()),
        "(-oo, oo) ∩ {x}",
        "before: {{x}}"
    );
    // `(Aᶜ)ᶜ = A` for a set of real numbers, as before
    let iv = ctx.interval(&x, &(&x + &ctx.int(1)), IntervalKind::Closed);
    assert_eq!(iv.absolute_complement().absolute_complement(), iv);
    let unit = ctx.interval(&ctx.int(0), &ctx.int(1), IntervalKind::RightOpen);
    assert_eq!(show(&unit.absolute_complement()), "(-oo, 0) ∪ [1, oo)");
    assert_eq!(unit.absolute_complement().absolute_complement(), unit);
}

/// The universal set is not the real line.
///
/// Before (0.59.0) the evaluator gave both one value: `UniversalSet \ {0}`
/// was `(-oo, 0) ∪ (0, oo)`, `UniversalSet \ [0, 1]` was `(-oo, 0) ∪ (1,
/// oo)` (both drop `i`, a member of the left-hand sides), and
/// `UniversalSet.inf()`/`sup()`/`closure()` were `-oo`/`oo`/`(-oo, oo)`.
/// The declared domain of a nonzero symbol that is not declared real went
/// through that difference: `(z² + a).solve_as_set(z)` for a nonzero `z` was
/// `{√(−a), −√(−a)} ∩ ((−∞, 0) ∪ (0, ∞))`, real roots only.
///
/// Now `U \ A` stays (`U \ U = ∅`, `U \ ∅ = U` and `U \ (U \ A) = A` are
/// exact), the bounds of `U` are not known, and the domain of such a `z` is
/// `UniversalSet \ {0}`.
///
/// Oracle: the definition (`i ∈ U \ {0}`, `i ∉ ℝ \ {0}`).  SymPy 1.14:
/// `S.UniversalSet - Interval(0, 1)` is `Complement(UniversalSet,
/// Interval(0, 1))`, `S.UniversalSet - S.Reals` is
/// `Complement(UniversalSet, Reals)`.
#[test]
fn the_universal_set_is_not_the_real_line() {
    let ctx = Context::new();
    let i = imaginary_unit(&ctx);
    let u = ctx.universal_set();
    let reals = ctx.reals();
    let zero = ctx.finite_set(&[ctx.int(0)]);
    let unit = ctx.interval(&ctx.int(0), &ctx.int(1), IntervalKind::Closed);

    let d = u.difference(&zero);
    assert_eq!(
        show(&d),
        "UniversalSet \\ {0}",
        "before: (-oo, 0) ∪ (0, oo)"
    );
    assert_eq!(d.contains(&i), Some(true));
    assert_eq!(d.contains(&ctx.int(0)), Some(false));
    assert_eq!(show(&u.difference(&unit)), "UniversalSet \\ [0, 1]");
    assert_eq!(
        show(&u.difference(&reals)),
        "UniversalSet \\ (-oo, oo)",
        "before: EmptySet"
    );
    // exact identities of the universal set
    assert_eq!(show(&u.difference(&u)), "EmptySet");
    assert_eq!(show(&reals.difference(&u)), "EmptySet");
    assert_eq!(u.difference(&ctx.empty_set()), u);
    assert_eq!(
        u.difference(&u.complement(&unit)),
        unit,
        "U \\ (U \\ A) = A"
    );
    assert_eq!(show(&u.intersection(&reals).simplify()), "(-oo, oo)");
    assert_eq!(show(&u.union(&reals).simplify()), "UniversalSet");
    assert_eq!(u.is_subset(&reals), Some(false));
    assert_eq!(reals.is_subset(&u), Some(true));
    assert_eq!(
        u.is_disjoint(&ctx.finite_set(std::slice::from_ref(&i))),
        Some(false)
    );
    // the absolute complement is taken in ℝ
    assert_eq!(show(&u.absolute_complement()), "EmptySet");
    assert_eq!(show(&ctx.empty_set().absolute_complement()), "(-oo, oo)");
    // no bounds, no interval form (before: -oo, oo, oo, `(-oo, oo)`)
    assert!(u.inf().is_none() && u.sup().is_none() && u.measure().is_none());
    assert!(u.closure().is_none() && u.as_intervals().is_none());
    assert_eq!(u.is_empty(), Some(false));
    // a formula true for every value still decides
    let x = ctx.symbol("x");
    let always = ctx.int(1).gt(&ctx.int(0)).or(&x.gt(&ctx.int(0)));
    assert_eq!(show(&always.solve_for(&x).unwrap()), "UniversalSet");
    assert_eq!(always.is_tautology(), Some(true));

    // the domain of a nonzero symbol not declared real
    let z = ctx.symbol_with("z", &[Assumption::NonZero]).unwrap();
    let a = ctx.symbol("a");
    assert_eq!(
        show(&(&z.powi(2) + &a).solve_as_set(&z)),
        "{sqrt(-a), -sqrt(-a)} \\ {0}",
        "before: ∩ ((-oo, 0) ∪ (0, oo))"
    );
    // `A ∩ (U \ X) = A \ X` and `A \ (U \ X) = A ∩ X` are exact
    let away = u.complement(&zero);
    let seg = ctx.interval(&ctx.int(-1), &ctx.int(3), IntervalKind::Closed);
    assert_eq!(
        show(&seg.intersection(&away).simplify()),
        "[-1, 0) ∪ (0, 3]"
    );
    assert_eq!(show(&seg.difference(&away)), "{0}");
    let fi = ctx.finite_set(std::slice::from_ref(&i));
    assert_eq!(show(&fi.intersection(&away).simplify()), "{I}");
    assert_eq!(show(&fi.difference(&away)), "EmptySet");
    assert_eq!(
        show(&(&z.powi(2) + &ctx.int(1)).solve_as_set(&z)),
        "{-I, I}"
    );
    // of a real nonzero one, as before
    let w = ctx
        .symbol_with("w", &[Assumption::Real, Assumption::NonZero])
        .unwrap();
    let real_roots = show(&(&w.powi(2) + &a).solve_as_set(&w));
    assert!(
        real_roots.contains("(-oo, 0) ∪ (0, oo)") && !real_roots.contains("UniversalSet"),
        "{real_roots}"
    );
}

/// Membership in a set of real numbers needs a real element.
///
/// Before (0.59.0): `[i, i + 1].contains(i + 1/2)` was `Some(true)` (the
/// order of the differences `1/2`, `1/2`, taken as if the three were real);
/// `{1, 2}.contains(i)`, `[x, x + 1].contains(i)` and `{i} ⊆ [0, 1]` were
/// `None`.
///
/// Now an element that is certainly not real (`i`, `±∞`, `zoo`) is not a
/// member of an interval or of a set in normal form; an element proved real
/// is compared; for one not known to be real the answer is `None`.
///
/// Oracle: the definition (an interval is a set of real numbers).  SymPy
/// 1.14: `Interval(0, 1).contains(I)` is `False`, `S.Reals.contains(zoo)`,
/// `S.Reals.contains(oo)` and `FiniteSet(1, 2).contains(I)` are `False`,
/// `S.Reals.contains(x)` stays a condition; `Interval(I, 2*I)` raises
/// "Non-real intervals are not supported".
#[test]
fn a_member_of_a_real_set_is_proved_real() {
    let ctx = Context::new();
    let i = imaginary_unit(&ctx);
    let half = ctx.rational(1, 2);
    let x = ctx.symbol("x");
    let r = ctx.symbol_with("r", &[Assumption::Real]).unwrap();
    let p = ctx.symbol_with("p", &[Assumption::Positive]).unwrap();
    let reals = ctx.reals();
    let unit = ctx.interval(&ctx.int(0), &ctx.int(1), IntervalKind::Closed);

    let off = ctx.interval(&i, &(&i + &ctx.int(1)), IntervalKind::Closed);
    assert_eq!(
        off.contains(&(&i + &half)),
        Some(false),
        "before: Some(true)"
    );
    assert_eq!(off.contains(&i), Some(false), "before: Some(true)");

    for set in [
        reals.clone(),
        unit.clone(),
        ctx.finite_set(&[ctx.int(1), ctx.int(2)]),
        ctx.interval(&x, &(&x + &ctx.int(1)), IntervalKind::Closed),
        ctx.interval(&x, &ctx.infinity(), IntervalKind::RightOpen),
    ] {
        for e in [
            i.clone(),
            &ctx.int(1) + &i,
            ctx.int(2).acos(),
            &p * &i,
            ctx.infinity(),
            ctx.neg_infinity(),
            ctx.complex_infinity(),
        ] {
            assert_eq!(set.contains(&e), Some(false), "{e} ∈ {set}");
        }
    }
    // proved real: in the line; undecided against a part of it
    for e in [r.clone(), &r + &ctx.pi(), x.abs(), r.sin()] {
        assert_eq!(reals.contains(&e), Some(true), "{e} ∈ ℝ");
        assert_eq!(unit.contains(&e), None, "{e} ∈ [0, 1]");
    }
    // not known to be real: undecided
    for e in [x.clone(), &x + &half, ctx.symbol("a").sqrt(), ctx.nan()] {
        assert_eq!(reals.contains(&e), None, "{e} ∈ ℝ");
    }
    // where the interval is defined, `x` is real: as before
    let iv = ctx.interval(&x, &(&x + &ctx.int(1)), IntervalKind::Closed);
    assert_eq!(iv.contains(&(&x + &half)), Some(true));
    assert_eq!(iv.contains(&(&x + &ctx.int(2))), Some(false));

    // A constant is a point of the line only when it is proved real: its
    // 16-digit value is no proof.  `e = 1 + i·(sin²1 + cos²1 − 1 + e⁻⁵⁰⁰⁰)`
    // has the imaginary part `e⁻⁵⁰⁰⁰ ≠ 0`, which reads as 0 to every
    // precision the evaluation reaches.  Before: `e` was the point `1`
    // (`[0, 2].contains(e)` was `Some(true)`, `{e} ∪ [0, 2]` was `[0, 2]`,
    // `{e} \ ℝ` was `EmptySet`).  Now undecided, and nothing is absorbed.
    // Oracle: `im e = e⁻⁵⁰⁰⁰ > 0` exactly (`sin² + cos² = 1`).
    let hidden = &(&(&ctx.int(1).cos().powi(2) + &ctx.int(1).sin().powi(2)) - &ctx.int(1))
        + &ctx.int(-5000).exp();
    let e = &ctx.int(1) + &(&i * &hidden);
    let two = ctx.interval(&ctx.int(0), &ctx.int(2), IntervalKind::Closed);
    let fe = ctx.finite_set(std::slice::from_ref(&e));
    assert_eq!(two.contains(&e), None, "before: Some(true)");
    assert_eq!(reals.contains(&e), None, "before: Some(true)");
    assert_ne!(show(&fe.union(&two).simplify()), "[0, 2]");
    assert_eq!(fe.union(&two).simplify().contains(&e), Some(true));
    assert_ne!(show(&fe.difference(&reals)), "EmptySet");
    // with an imaginary part that evaluates (`e⁻⁵⁰⁰⁰` alone): decided
    let tiny = &ctx.int(1) + &(&i * &ctx.int(-5000).exp());
    assert_eq!(two.contains(&tiny), Some(false));
    assert_eq!(
        show(
            &ctx.finite_set(std::slice::from_ref(&tiny))
                .difference(&reals)
        ),
        "{exp(-5000)*I + 1}"
    );
    // real constants keep their places on the line (`π − 355/113 < 0`)
    let wide = ctx.interval(&ctx.int(-1), &ctx.int(2), IntervalKind::Closed);
    for c in [
        ctx.int(2).sqrt(),
        ctx.rational(1, 2).asin(),
        ctx.int(1).sin(),
        &ctx.int(1) + &ctx.int(-70).exp(),
        &ctx.pi() - &ctx.rational(355, 113),
    ] {
        assert_eq!(wide.contains(&c), Some(true), "{c} ∈ [-1, 2]");
    }
    assert_eq!(
        two.contains(&(&ctx.pi() - &ctx.rational(355, 113))),
        Some(false)
    );

    // subsets: a member that is certainly not real is in no set of reals
    let fi = ctx.finite_set(std::slice::from_ref(&i));
    assert_eq!(fi.is_subset(&unit), Some(false), "before: None");
    assert_eq!(fi.is_subset(&reals), Some(false));
    assert_eq!(unit.is_subset(&reals.union(&fi)), Some(true));
    assert_eq!(
        ctx.finite_set(std::slice::from_ref(&x)).is_subset(&reals),
        None
    );
    assert_eq!(
        ctx.finite_set(std::slice::from_ref(&r)).is_subset(&reals),
        Some(true)
    );
}

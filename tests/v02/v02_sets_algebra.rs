//! symplex 0.2 — set algebra on `SetEx`: normal form, eager operations,
//! three-valued queries, bounds/measure, topology, accessors.

use symplex::prelude::*;

/// `[a, b]` with the given ends excluded (`lo_open`, `hi_open`).
fn iv(ctx: &Context, a: i64, b: i64, lo_open: bool, hi_open: bool) -> SetEx {
    ctx.interval(
        &ctx.int(a),
        &ctx.int(b),
        IntervalKind::from_open_ends(lo_open, hi_open),
    )
}

fn s(e: &SetEx) -> String {
    format!("{e}")
}

// ═══════════════════════════════════════════════════════════════════════════
// Normal form
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn intersection_of_overlapping_intervals() {
    let ctx = Context::new();
    let a = iv(&ctx, 0, 2, false, false);
    let b = iv(&ctx, 1, 3, false, false);
    assert_eq!(s(&a.intersection(&b).simplify()), "[1, 2]");
    // construction stays structural
    assert!(s(&a.intersection(&b)).contains('∩'));
}

#[test]
fn union_merges_and_sorts() {
    let ctx = Context::new();
    let a = iv(&ctx, 3, 4, false, false);
    let b = iv(&ctx, 0, 1, false, true);
    let c = iv(&ctx, 1, 2, false, false);
    let u = a.union(&b).union(&c).simplify();
    assert_eq!(s(&u), "[0, 2] ∪ [3, 4]");
}

#[test]
fn open_endpoints_do_not_merge() {
    let ctx = Context::new();
    let a = iv(&ctx, 0, 1, true, true);
    let b = iv(&ctx, 1, 2, true, true);
    assert_eq!(s(&a.union(&b).simplify()), "(0, 1) ∪ (1, 2)");
    // but adding the point bridges them
    let p = ctx.finite_set(&[ctx.int(1)]);
    assert_eq!(s(&a.union(&b).union(&p).simplify()), "(0, 2)");
}

#[test]
fn solver_output_intersected_with_window() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sol = (&x.powi(2) - 4).solve_gt(&x);
    let window = iv(&ctx, 0, 5, false, false);
    assert_eq!(s(&sol.intersection(&window).simplify()), "(2, 5]");
    let sol_le = (&x.powi(2) - 4).solve_le(&x);
    assert_eq!(s(&sol_le.simplify()), "[-2, 2]");
}

#[test]
fn points_absorbed_into_intervals() {
    let ctx = Context::new();
    let a = iv(&ctx, 0, 2, true, true);
    let fs = ctx.finite_set(&[ctx.int(1), ctx.int(2), ctx.int(7)]);
    assert_eq!(s(&a.union(&fs).simplify()), "(0, 2] ∪ {7}");
}

#[test]
fn simplify_is_idempotent_and_order_independent() {
    let ctx = Context::new();
    let a = iv(&ctx, 0, 2, false, true);
    let b = iv(&ctx, 1, 3, true, false);
    let c = iv(&ctx, 5, 6, false, false);
    let u1 = a.union(&b).union(&c).simplify();
    let u2 = c.union(&b).union(&a).simplify();
    assert_eq!(u1, u2);
    assert_eq!(u1.simplify(), u1);
}

#[test]
fn irrational_endpoints_are_ordered_numerically() {
    let ctx = Context::new();
    let pi = ctx.pi();
    let e = ctx.e();
    let a = ctx.interval(&ctx.int(0), &pi, IntervalKind::Closed);
    let b = ctx.interval(&e, &ctx.int(4), IntervalKind::Closed);
    assert_eq!(s(&a.intersection(&b).simplify()), "[E, pi]");
    let sqrt2 = ctx.int(2).sqrt();
    let c = ctx.interval(&ctx.int(1), &sqrt2, IntervalKind::Closed);
    let d = ctx.interval(&sqrt2, &ctx.int(2), IntervalKind::LeftOpen);
    assert_eq!(s(&c.union(&d).simplify()), "[1, 2]");
}

#[test]
fn evaluable_endpoints_are_evaluated() {
    let ctx = Context::new();
    let four = ctx.int(4);
    let a = ctx.interval(&four.sqrt(), &ctx.int(5), IntervalKind::Closed);
    assert_eq!(s(&a.simplify()), "[2, 5]");
}

#[test]
fn symbolic_sets_keep_safe_identities() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sym = ctx.interval(&x, &ctx.int(1), IntervalKind::Closed);
    assert_eq!(sym.union(&ctx.empty_set()).simplify(), sym);
    assert_eq!(sym.intersection(&ctx.universal_set()).simplify(), sym);
    assert_eq!(sym.union(&sym).simplify(), sym);
    assert_eq!(sym.intersection(&sym).simplify(), sym);
    assert_eq!(
        sym.intersection(&ctx.empty_set()).simplify(),
        ctx.empty_set()
    );
    assert_eq!(
        sym.union(&ctx.universal_set()).simplify(),
        ctx.universal_set()
    );
    assert_eq!(sym.absolute_complement().absolute_complement(), sym);
    assert_eq!(sym.difference(&sym), ctx.empty_set());
    // structural pieces survive next to evaluated ones
    let num = iv(&ctx, 0, 1, false, false).union(&iv(&ctx, 1, 2, false, false));
    let mixed = num.union(&sym).simplify();
    let d = s(&mixed);
    assert!(d.contains("[0, 2]") && d.contains("[x, 1]"), "{d}");
}

#[test]
fn nested_structure_is_flattened() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let a = ctx.interval(&x, &ctx.int(1), IntervalKind::Closed);
    let b = ctx.interval(&y, &ctx.int(1), IntervalKind::Closed);
    let c = ctx.interval(&x, &ctx.int(2), IntervalKind::Closed);
    let u = a.union(&b.union(&c)).simplify();
    assert_eq!(u.as_ex().args().len(), 3, "{u}");
    assert_eq!(u.as_ex().expr_type(), ExprType::Set);
}

// ═══════════════════════════════════════════════════════════════════════════
// Eager operations
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn difference_and_symmetric_difference() {
    let ctx = Context::new();
    let a = iv(&ctx, 0, 3, false, false);
    let b = iv(&ctx, 1, 2, true, true);
    assert_eq!(s(&a.difference(&b)), "[0, 1] ∪ [2, 3]");
    assert_eq!(s(&b.difference(&a)), "EmptySet");
    assert_eq!(s(&a.symmetric_difference(&b)), "[0, 1] ∪ [2, 3]");
    let c = iv(&ctx, 2, 5, false, false);
    assert_eq!(s(&a.symmetric_difference(&c)), "[0, 2) ∪ (3, 5]");
    assert_eq!(s(&a.difference(&ctx.reals())), "EmptySet");
    assert_eq!(a.difference(&ctx.empty_set()), a);
}

#[test]
fn absolute_complement_examples() {
    let ctx = Context::new();
    let a = iv(&ctx, 0, 1, false, false);
    assert_eq!(s(&a.absolute_complement()), "(-oo, 0) ∪ (1, oo)");
    assert_eq!(s(&ctx.reals().absolute_complement()), "EmptySet");
    assert_eq!(s(&ctx.universal_set().absolute_complement()), "EmptySet");
    assert_eq!(s(&ctx.empty_set().absolute_complement()), "(-oo, oo)");
    let p = ctx.finite_set(&[ctx.int(0)]);
    assert_eq!(s(&p.absolute_complement()), "(-oo, 0) ∪ (0, oo)");
    assert_eq!(s(&p.absolute_complement().absolute_complement()), "{0}");
}

#[test]
fn lazy_complement_simplifies() {
    let ctx = Context::new();
    let a = iv(&ctx, 0, 3, false, false);
    let b = iv(&ctx, 1, 2, false, false);
    let lazy = a.complement(&b);
    assert!(s(&lazy).contains('\\'));
    assert_eq!(s(&lazy.simplify()), "[0, 1) ∪ (2, 3]");
}

// ═══════════════════════════════════════════════════════════════════════════
// Queries
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn contains_and_is_in() {
    let ctx = Context::new();
    let a = iv(&ctx, 0, 1, true, false);
    assert_eq!(a.contains(&ctx.int(0)), Some(false));
    assert_eq!(a.contains(&ctx.int(1)), Some(true));
    assert_eq!(a.contains(&ctx.rational(1, 2)), Some(true));
    assert_eq!(ctx.rational(1, 2).is_in(&a), Some(true));
    assert_eq!(a.contains(&ctx.int(2)), Some(false));
    assert_eq!(a.contains(&ctx.symbol("x")), None);
    assert_eq!(ctx.empty_set().contains(&ctx.symbol("x")), Some(false));
    assert_eq!(ctx.universal_set().contains(&ctx.symbol("x")), Some(true));
    // real symbol is in ℝ; unconstrained symbol is unknown
    let r = ctx.symbol_with("r", &[Assumption::Real]);
    assert_eq!(ctx.reals().contains(&r), Some(true));
    assert_eq!(ctx.reals().contains(&ctx.symbol("z")), None);
    // infinities are never members of real sets
    assert_eq!(ctx.reals().contains(&ctx.infinity()), Some(false));
    // pi in (3, 4)
    assert_eq!(iv(&ctx, 3, 4, true, true).contains(&ctx.pi()), Some(true));
}

#[test]
fn contains_on_symbolic_sets() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sym = ctx.interval(&x, &(&x + 1), IntervalKind::RightOpen);
    assert_eq!(sym.contains(&x), Some(true));
    assert_eq!(sym.contains(&(&x + 1)), Some(false));
    assert_eq!(sym.contains(&(&x + 2)), Some(false));
    assert_eq!(sym.contains(&(&x + &ctx.rational(1, 2))), Some(true));
    assert_eq!(sym.contains(&ctx.int(0)), None);
    let fs = ctx.finite_set(&[x.clone(), ctx.int(1)]);
    assert_eq!(fs.contains(&x), Some(true));
    assert_eq!(fs.contains(&ctx.int(1)), Some(true));
    assert_eq!(fs.contains(&ctx.int(2)), None, "2 might equal x");
    // sin(x) - 2 > 0 never holds: whether the solver returns ∅ or a
    // ConditionSet, membership of 0 must be decided as false.
    let cs = (&x.sin() - 2).solve_gt(&x);
    assert_eq!(cs.contains(&ctx.int(0)), Some(false), "{cs}");
}

#[test]
fn subset_superset_disjoint_empty() {
    let ctx = Context::new();
    let a = iv(&ctx, 0, 1, false, false);
    let b = iv(&ctx, -1, 2, true, true);
    let c = iv(&ctx, 1, 3, true, false);
    assert_eq!(a.is_subset(&b), Some(true));
    assert_eq!(b.is_superset(&a), Some(true));
    assert_eq!(b.is_subset(&a), Some(false));
    assert_eq!(a.is_disjoint(&c), Some(true));
    assert_eq!(a.is_disjoint(&b), Some(false));
    assert_eq!(a.is_empty(), Some(false));
    assert_eq!(a.intersection(&c).is_empty(), Some(true));
    assert_eq!(ctx.empty_set().is_subset(&a), Some(true));
    assert_eq!(a.is_subset(&ctx.universal_set()), Some(true));
    let x = ctx.symbol("x");
    let sym = ctx.interval(&x, &ctx.int(1), IntervalKind::Closed);
    assert_eq!(sym.is_empty(), None);
    assert_eq!(sym.is_subset(&sym), Some(true));
    assert_eq!(sym.is_subset(&a), None);
    assert_eq!(a.is_subset(&sym), None);
    assert_eq!(
        ctx.finite_set(std::slice::from_ref(&x)).is_empty(),
        Some(false)
    );
    let fx = ctx.finite_set(std::slice::from_ref(&x));
    let fxy = ctx.finite_set(&[x.clone(), ctx.symbol("y")]);
    assert_eq!(fx.is_subset(&fxy), Some(true));
}

#[test]
fn inf_sup_measure() {
    let ctx = Context::new();
    let a = iv(&ctx, 0, 1, true, true);
    let b = iv(&ctx, 2, 5, false, false);
    let u = a.union(&b);
    assert_eq!(format!("{}", u.inf().unwrap()), "0");
    assert_eq!(format!("{}", u.sup().unwrap()), "5");
    assert_eq!(format!("{}", u.measure().unwrap()), "4");
    assert_eq!(format!("{}", ctx.reals().inf().unwrap()), "-oo");
    assert_eq!(format!("{}", ctx.reals().sup().unwrap()), "oo");
    assert_eq!(format!("{}", ctx.reals().measure().unwrap()), "oo");
    let ray = ctx.interval(&ctx.int(0), &ctx.infinity(), IntervalKind::RightOpen);
    assert_eq!(format!("{}", ray.measure().unwrap()), "oo");
    assert!(ctx.empty_set().inf().is_none());
    assert_eq!(format!("{}", ctx.empty_set().measure().unwrap()), "0");
    let fs = ctx.finite_set(&[ctx.int(3), ctx.int(-1)]);
    assert_eq!(format!("{}", fs.inf().unwrap()), "-1");
    assert_eq!(format!("{}", fs.sup().unwrap()), "3");
    assert_eq!(format!("{}", fs.measure().unwrap()), "0");
    let pi_iv = ctx.interval(&ctx.int(0), &ctx.pi(), IntervalKind::Closed);
    assert_eq!(format!("{}", pi_iv.measure().unwrap()), "pi");
    let x = ctx.symbol("x");
    assert!(
        ctx.interval(&x, &ctx.int(1), IntervalKind::Closed)
            .inf()
            .is_none()
    );
}

#[test]
fn topology() {
    let ctx = Context::new();
    let a = iv(&ctx, 0, 1, true, false);
    let p = ctx.finite_set(&[ctx.int(2)]);
    let u = a.union(&p);
    assert_eq!(s(&u.closure().unwrap()), "[0, 1] ∪ {2}");
    assert_eq!(s(&u.interior().unwrap()), "(0, 1)");
    assert_eq!(s(&u.boundary().unwrap()), "{0, 1, 2}");
    assert_eq!(u.is_open(), Some(false));
    assert_eq!(u.is_closed(), Some(false));
    assert_eq!(u.closure().unwrap().is_closed(), Some(true));
    assert_eq!(u.interior().unwrap().is_open(), Some(true));
    assert_eq!(ctx.reals().is_open(), Some(true));
    assert_eq!(ctx.reals().is_closed(), Some(true));
    assert_eq!(s(&ctx.reals().boundary().unwrap()), "EmptySet");
    assert_eq!(ctx.empty_set().is_open(), Some(true));
    assert_eq!(ctx.empty_set().is_closed(), Some(true));
    let half = ctx.interval(&ctx.neg_infinity(), &ctx.int(0), IntervalKind::LeftOpen);
    assert_eq!(half.is_closed(), Some(true));
    assert_eq!(s(&half.boundary().unwrap()), "{0}");
    let x = ctx.symbol("x");
    let fx = ctx.finite_set(&[x]);
    assert_eq!(fx.is_closed(), Some(true));
    assert_eq!(fx.is_open(), Some(false));
    assert!(
        ctx.interval(&ctx.symbol("y"), &ctx.int(1), IntervalKind::Open)
            .is_open()
            .is_none()
    );
}

#[test]
fn accessors() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sol = (&x.powi(2) - 4).solve_ge(&x);
    let parts = sol.as_intervals().unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(format!("{}", parts[0].lower), "-oo");
    assert_eq!(format!("{}", parts[0].upper), "-2");
    assert_eq!(parts[0].kind, IntervalKind::LeftOpen);
    assert_eq!(format!("{}", parts[1].lower), "2");
    assert_eq!(format!("{}", parts[1].upper), "oo");
    assert_eq!(parts[1].kind, IntervalKind::RightOpen);
    // Round trip through `Interval<Ex>::to_set`.
    assert_eq!(s(&parts[0].to_set()), "(-oo, -2]");
    assert_eq!(s(&parts[1].to_set()), "[2, oo)");

    let mixed = iv(&ctx, 0, 1, false, false).union(&ctx.finite_set(&[ctx.int(5)]));
    let parts = mixed.as_intervals().unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[1], Interval::point(ctx.int(5)));
    assert!(mixed.as_finite_set().is_none());

    let fs = ctx.finite_set(&[ctx.int(3), ctx.int(1)]);
    let elems = fs.as_finite_set().unwrap();
    assert_eq!(elems.len(), 2);
    assert_eq!(format!("{}", elems[0]), "1");
    assert_eq!(ctx.empty_set().as_finite_set(), Some(vec![]));
    assert!(
        ctx.interval(&x, &ctx.int(1), IntervalKind::Closed)
            .as_intervals()
            .is_none()
    );
    let sym_fs = ctx.finite_set(&[x.clone(), ctx.int(2)]);
    assert_eq!(sym_fs.as_finite_set().unwrap().len(), 2);
}

#[test]
fn eval_only_evaluates_endpoints() {
    let ctx = Context::new();
    let a = ctx.interval(&ctx.int(4).sqrt(), &ctx.int(5), IntervalKind::Closed);
    let b = iv(&ctx, 1, 3, false, false);
    let e = a.intersection(&b).eval();
    let d = s(&e);
    assert!(d.contains("[2, 5]") && d.contains('∩'), "{d}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Never-wrong property: membership of the normal form agrees with the
// structural membership condition at many sample points.
// ═══════════════════════════════════════════════════════════════════════════

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn range(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn random_set(ctx: &Context, rng: &mut Lcg, depth: u32) -> SetEx {
    if depth == 0 || rng.range(3) == 0 {
        return match rng.range(4) {
            0 => {
                let a = rng.range(9) as i64 - 4;
                let b = a + rng.range(4) as i64;
                iv(ctx, a, b, rng.range(2) == 0, rng.range(2) == 0)
            }
            1 => {
                let n = rng.range(3) + 1;
                let elems: Vec<Ex> = (0..n).map(|_| ctx.int(rng.range(9) as i64 - 4)).collect();
                ctx.finite_set(&elems)
            }
            2 => {
                let a = rng.range(9) as i64 - 4;
                if rng.range(2) == 0 {
                    let kind = IntervalKind::from_open_ends(true, rng.range(2) == 0);
                    ctx.interval(&ctx.neg_infinity(), &ctx.int(a), kind)
                } else {
                    let kind = IntervalKind::from_open_ends(rng.range(2) == 0, true);
                    ctx.interval(&ctx.int(a), &ctx.infinity(), kind)
                }
            }
            _ => ctx.empty_set(),
        };
    }
    let l = random_set(ctx, rng, depth - 1);
    let r = random_set(ctx, rng, depth - 1);
    match rng.range(4) {
        0 => l.union(&r),
        1 => l.intersection(&r),
        2 => l.complement(&r),
        _ => l.absolute_complement(),
    }
}

#[test]
fn normal_form_membership_matches_structural_condition() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let mut rng = Lcg(0x5EED);
    let points: Vec<Ex> = (-10..=10).map(|k| ctx.rational(k, 2)).collect();
    for case in 0..200 {
        let set = random_set(&ctx, &mut rng, 3);
        let nf = set.simplify();
        assert_eq!(nf.simplify(), nf, "idempotent: {set}");
        let cond = set.to_condition(&x).unwrap();
        let nf_cond = nf.to_condition(&x).unwrap();
        for p in &points {
            let truth = format!("{}", cond.subs(&x, p).eval());
            let expected = match truth.as_str() {
                "True" => true,
                "False" => false,
                other => panic!("condition did not evaluate: {other} (case {case}: {set})"),
            };
            assert_eq!(
                nf.contains(p),
                Some(expected),
                "case {case}: {set} → {nf}; point {p}"
            );
            assert_eq!(
                set.contains(p),
                Some(expected),
                "case {case} (structural): {set}; point {p}"
            );
            assert_eq!(
                format!("{}", nf_cond.subs(&x, p).eval()),
                truth,
                "case {case} (nf condition): {set} → {nf}; point {p}"
            );
        }
        // complement law and difference law on the normal form
        let comp = nf.absolute_complement();
        assert_eq!(nf.intersection(&comp).simplify(), ctx.empty_set(), "{nf}");
        assert_eq!(
            format!("{}", nf.union(&comp).simplify()),
            "(-oo, oo)",
            "{nf}"
        );
        assert_eq!(nf.is_disjoint(&comp), Some(true));
        assert_eq!(nf.is_subset(&nf), Some(true));
    }
}

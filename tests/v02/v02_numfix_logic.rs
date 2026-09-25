//! Regression tests for `src/transforms/logic.rs` (0.2 numfix):
//!
//! * `piecewise_simplify` did not prune a branch whose condition follows
//!   from the assumption system (`pos > 0` for a positive symbol);
//! * `BoolEx::simplify` missed consensus / resolution
//!   (`(p∧q) ∨ (p∧¬q) → p` and its dual).

use symplex::prelude::*;

fn s<T: std::fmt::Display>(e: &T) -> String {
    format!("{e}")
}

/// A condition that folds to `True` (there is no public `True` constructor).
fn always(ctx: &Context) -> BoolEx {
    ctx.int(2).gt(&ctx.int(1))
}

// ── piecewise_simplify with assumptions ───────────────────────────────────

#[test]
fn piecewise_prunes_branch_true_under_positive_assumption() {
    let ctx = Context::new();
    let pos = ctx.symbol_with("pos", &[Assumption::Positive]).unwrap();
    let zero = ctx.int(0);
    let always = always(&ctx);
    let neg = -&pos;
    // |pos| written as a piecewise: the first branch is always taken.
    let pw = Ex::piecewise(&[(&pos, &pos.gt(&zero)), (&neg, &always)]);
    assert_eq!(pw.piecewise_simplify(), pos);
}

#[test]
fn piecewise_drops_branch_false_under_assumption() {
    let ctx = Context::new();
    let pos = ctx.symbol_with("pos", &[Assumption::Positive]).unwrap();
    let zero = ctx.int(0);
    let pw = Ex::piecewise(&[(&ctx.int(1), &pos.lt(&zero)), (&ctx.int(2), &pos.ge(&zero))]);
    assert_eq!(pw.piecewise_simplify(), ctx.int(2));
}

#[test]
fn piecewise_with_negative_and_nonzero_assumptions() {
    let ctx = Context::new();
    let neg = ctx.symbol_with("n", &[Assumption::Negative]).unwrap();
    let zero = ctx.int(0);
    let minus = -&neg;
    let pw = Ex::piecewise(&[(&neg, &neg.gt(&zero)), (&minus, &neg.le(&zero))]);
    assert_eq!(pw.piecewise_simplify(), minus);

    let nz = ctx.symbol_with("k", &[Assumption::NonZero]).unwrap();
    let pw = Ex::piecewise(&[
        (&ctx.int(0), &nz.eq_expr(&zero)),
        (&(ctx.int(1) / &nz), &nz.ne_expr(&zero)),
    ]);
    assert_eq!(pw.piecewise_simplify(), ctx.int(1) / &nz);
}

#[test]
fn piecewise_without_assumptions_is_unchanged() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let neg_x = -&x;
    let abs = Ex::piecewise(&[(&x, &x.gt(&zero)), (&neg_x, &x.le(&zero))]);
    assert_eq!(abs.piecewise_simplify(), abs);
}

#[test]
fn piecewise_assumption_folding_reaches_nested_piecewise() {
    let ctx = Context::new();
    let pos = ctx.symbol_with("pos", &[Assumption::Positive]).unwrap();
    let zero = ctx.int(0);
    let inner = Ex::piecewise(&[(&ctx.int(1), &pos.gt(&zero)), (&ctx.int(0), &always(&ctx))]);
    let e = &inner * 3 + &pos;
    assert_eq!(e.piecewise_simplify(), &pos + 3);
}

// ── consensus / resolution in BoolEx::simplify ────────────────────────────

#[test]
fn consensus_or_of_ands_collapses_to_common_part() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let p = x.gt(&ctx.int(0));
    let q = x.lt(&ctx.int(5));
    // (p ∧ q) ∨ (p ∧ ¬q) → p
    let e = p.and(&q).or(&p.and(&q.not()));
    assert_eq!(e.simplify(), p);
}

#[test]
fn consensus_and_of_ors_collapses_to_common_part() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let p = x.gt(&ctx.int(0));
    let q = x.lt(&ctx.int(5));
    // (p ∨ q) ∧ (p ∨ ¬q) → p
    let e = p.or(&q).and(&p.or(&q.not()));
    assert_eq!(e.simplify(), p);
}

#[test]
fn consensus_with_opaque_atoms() {
    let ctx = Context::new();
    let y = ctx.symbol("y");
    let z = ctx.symbol("z");
    // Relationals in different symbols act as independent atoms.
    let p = y.gt(&ctx.int(1));
    let q = z.gt(&ctx.int(2));
    let r = z.lt(&ctx.int(-3));
    // (p∧q∧r) ∨ (p∧q∧¬r) → p∧q
    let e = p.and(&q).and(&r).or(&p.and(&q).and(&r.not()));
    assert_eq!(e.simplify(), p.and(&q).simplify());
}

#[test]
fn consensus_theorem_drops_redundant_term() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let p = x.gt(&ctx.int(0));
    let q = x.lt(&ctx.int(5));
    let r = y.gt(&ctx.int(0));
    // (p∧q) ∨ (¬p∧r) ∨ (q∧r) → (p∧q) ∨ (¬p∧r)
    let e = p.and(&q).or(&p.not().and(&r)).or(&q.and(&r));
    let want = p.and(&q).or(&p.not().and(&r)).simplify();
    assert_eq!(e.simplify(), want, "{}", s(&e.simplify()));
    assert_eq!(s(&e.simplify()), "5 > x & x > 0 | y > 0 & 0 >= x");
}

#[test]
fn consensus_merges_relationals_on_same_pair() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let c = y.gt(&ctx.int(0));
    // (c ∧ x > 0) ∨ (c ∧ x = 0) → c ∧ x ≥ 0
    let e = c
        .and(&x.gt(&ctx.int(0)))
        .or(&c.and(&x.eq_expr(&ctx.int(0))));
    assert_eq!(e.simplify(), c.and(&x.ge(&ctx.int(0))).simplify());
}

#[test]
fn consensus_all_cases_covered_is_true() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let p = x.gt(&ctx.int(0));
    let q = y.gt(&ctx.int(0));
    // (p∧q) ∨ (p∧¬q) ∨ (¬p∧q) ∨ (¬p∧¬q) → true
    let e = p
        .and(&q)
        .or(&p.and(&q.not()))
        .or(&p.not().and(&q))
        .or(&p.not().and(&q.not()));
    let simplified = e.simplify();
    assert_eq!(s(&simplified), "True");
    assert_eq!(simplified.is_tautology(), Some(true));
}

#[test]
fn consensus_is_sound_on_truth_table() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let z = ctx.symbol("z");
    let p = x.gt(&ctx.int(0));
    let q = y.gt(&ctx.int(0));
    let r = z.gt(&ctx.int(0));
    let cases = [
        p.and(&q).or(&p.and(&q.not())),
        p.or(&q).and(&p.or(&q.not())),
        p.and(&q).or(&p.not().and(&r)).or(&q.and(&r)),
        p.or(&q).and(&p.not().or(&r)).and(&q.or(&r)),
        p.and(&q).and(&r).or(&p.and(&q.not()).and(&r)),
    ];
    let vars = [p.clone(), q.clone(), r.clone()];
    for e in &cases {
        let simplified = e.simplify();
        assert_eq!(
            e.truth_table(&vars).unwrap(),
            simplified.truth_table(&vars).unwrap(),
            "{e} vs {simplified}"
        );
    }
}

#[test]
fn consensus_leaves_non_resolvable_terms_alone() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let z = ctx.symbol("z");
    let p = x.gt(&ctx.int(0));
    let q = y.gt(&ctx.int(0));
    let r = z.gt(&ctx.int(0));
    // (p∧q) ∨ (¬p∧¬q) — two complementary positions: irreducible.
    let e = p.and(&q).or(&p.not().and(&q.not()));
    assert_eq!(s(&e.simplify()), "x > 0 & y > 0 | 0 >= x & 0 >= y");
    // (p∧q) ∨ (p∧r) — no complementary literal: irreducible.
    let f = p.and(&q).or(&p.and(&r));
    assert_eq!(f.simplify(), f);
}

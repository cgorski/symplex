//! symplex 0.2 — boolean logic on `BoolEx` (simplify, eval, normal forms,
//! satisfiability, truth tables), `Ex::piecewise_simplify`, and the 0.2
//! assumption-system additions.

use symplex::prelude::*;

fn s<T: std::fmt::Display>(e: &T) -> String {
    format!("{e}")
}

// ═══════════════════════════════════════════════════════════════════════════
// simplify
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn simplify_flattens_and_dedupes() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let p = x.gt(&ctx.int(0));
    let q = y.gt(&ctx.int(0));
    let nested = p.and(&q).and(&p.and(&q)).and(&p);
    let simp = nested.simplify();
    assert_eq!(simp.args().len(), 2, "{simp}");
    assert_eq!(simp, q.and(&p).simplify(), "order independent");
    assert_eq!(simp.simplify(), simp, "idempotent");
}

#[test]
fn simplify_constant_folding() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let p = x.gt(&ctx.int(0));
    let t = ctx.int(2).gt(&ctx.int(1));
    let f = ctx.int(1).gt(&ctx.int(2));
    assert_eq!(p.and(&t).simplify(), p);
    assert_eq!(s(&p.and(&f).simplify()), "False");
    assert_eq!(s(&p.or(&t).simplify()), "True");
    assert_eq!(p.or(&f).simplify(), p);
    assert_eq!(s(&ctx.pi().gt(&ctx.int(3)).simplify()), "True");
    assert_eq!(s(&ctx.e().gt(&ctx.int(3)).simplify()), "False");
    assert_eq!(s(&(&x + 1).gt(&x).simplify()), "True");
    assert_eq!(s(&x.ge(&x).simplify()), "True");
    assert_eq!(s(&x.gt(&x).simplify()), "False");
    assert_eq!(s(&x.eq_expr(&x).simplify()), "True");
}

#[test]
fn simplify_relational_negation_and_merging() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    assert_eq!(s(&x.gt(&zero).not().simplify()), "0 >= x");
    assert_eq!(s(&x.ge(&zero).not().simplify()), "0 > x");
    assert_eq!(s(&x.lt(&zero).not().simplify()), "x >= 0");
    assert_eq!(s(&x.eq_expr(&zero).not().simplify()), "x != 0");
    assert_eq!(s(&x.ne_expr(&zero).not().simplify()), "x == 0");
    assert_eq!(x.gt(&zero).and(&x.ge(&zero)).simplify(), x.gt(&zero));
    assert_eq!(x.gt(&zero).or(&x.eq_expr(&zero)).simplify(), x.ge(&zero));
    assert_eq!(x.ge(&zero).and(&x.le(&zero)).simplify(), x.eq_expr(&zero));
    assert_eq!(s(&x.gt(&zero).and(&x.lt(&zero)).simplify()), "False");
    assert_eq!(s(&x.gt(&zero).or(&x.le(&zero)).simplify()), "True");
    assert_eq!(s(&x.gt(&zero).and(&x.ne_expr(&zero)).simplify()), "x > 0");
}

#[test]
fn simplify_absorption_and_complement() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let p = x.gt(&ctx.int(0));
    let q = y.gt(&ctx.int(0));
    assert_eq!(p.and(&p.or(&q)).simplify(), p);
    assert_eq!(p.or(&p.and(&q)).simplify(), p);
    assert_eq!(s(&p.and(&p.not()).simplify()), "False");
    assert_eq!(s(&p.or(&p.not()).simplify()), "True");
    assert_eq!(p.and(&p.not().or(&q)).simplify(), p.and(&q).simplify());
    assert_eq!(s(&p.implies(&p).simplify()), "True");
    assert_eq!(s(&p.equivalent(&p).simplify()), "True");
    assert_eq!(s(&p.xor(&p).simplify()), "False");
}

#[test]
fn simplify_de_morgan_and_nnf() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let p = x.gt(&ctx.int(0));
    let q = y.gt(&ctx.int(0));
    let e = p.and(&q).not();
    assert_eq!(s(&e.to_nnf()), "0 >= x | 0 >= y");
    assert_eq!(s(&e.simplify()), "0 >= x | 0 >= y");
    assert_eq!(s(&p.or(&q).not().to_nnf()), "0 >= x & 0 >= y");
    assert_eq!(p.not().not().simplify(), p);
    // opaque atoms
    let a = ctx.symbol("a").gt(&ctx.int(0));
    let b = ctx.symbol("b").gt(&ctx.int(0));
    let inner = a.and(&b.not());
    assert_eq!(s(&inner.not().to_nnf()), "0 >= a | b > 0");
}

#[test]
fn simplify_numeric_subexpressions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = (&x.sin().powi(2) + &x.cos().powi(2)).gt(&ctx.int(0));
    assert_eq!(s(&e.simplify()), "True");
}

// ═══════════════════════════════════════════════════════════════════════════
// eval with assumptions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eval_uses_assumptions() {
    let ctx = Context::new();
    let t = ctx.symbol_with("t", &[Assumption::Positive]);
    let n = ctx.symbol_with("n", &[Assumption::Negative]);
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    assert_eq!(s(&t.gt(&zero).eval()), "True");
    assert_eq!(s(&t.ge(&zero).eval()), "True");
    assert_eq!(s(&t.lt(&zero).eval()), "False");
    assert_eq!(s(&t.le(&zero).eval()), "False");
    assert_eq!(s(&t.eq_expr(&zero).eval()), "False");
    assert_eq!(s(&t.ne_expr(&zero).eval()), "True");
    assert_eq!(s(&n.lt(&zero).eval()), "True");
    assert_eq!(s(&(&t + &t.powi(2)).gt(&zero).eval()), "True");
    assert_eq!(s(&(&t + 1).gt(&ctx.int(1)).eval()), "True");
    // unknown stays symbolic; connectives fold around it
    assert_eq!(s(&x.gt(&zero).eval()), "x > 0");
    assert_eq!(x.gt(&zero).and(&t.gt(&zero)).eval(), x.gt(&zero));
    assert_eq!(s(&x.gt(&zero).or(&t.gt(&zero)).eval()), "True");
    assert_eq!(s(&x.gt(&zero).and(&t.lt(&zero)).eval()), "False");
    // numeric folding still works
    assert_eq!(s(&ctx.int(5).gt(&ctx.int(3)).eval()), "True");
    assert_eq!(s(&ctx.int(5).gt(&ctx.int(3)).not().eval()), "False");
}

// ═══════════════════════════════════════════════════════════════════════════
// CNF / DNF
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cnf_dnf() {
    let ctx = Context::new();
    let p = ctx.symbol("p").gt(&ctx.int(0));
    let q = ctx.symbol("q").gt(&ctx.int(0));
    let r = ctx.symbol("r").gt(&ctx.int(0));
    let e = p.or(&q.and(&r));
    assert_eq!(s(&e.to_cnf()), "(p > 0 | q > 0) & (p > 0 | r > 0)");
    assert_eq!(e.to_dnf(), e.simplify());
    let e2 = p.and(&q.or(&r));
    assert_eq!(s(&e2.to_dnf()), "p > 0 & q > 0 | p > 0 & r > 0");
    assert_eq!(e2.to_cnf(), e2.simplify());
    // xor: (p ∧ ¬q) ∨ (¬p ∧ q)
    let x = p.xor(&q);
    let cnf = x.to_cnf();
    assert_eq!(s(&cnf), "(p > 0 | q > 0) & (0 >= p | 0 >= q)");
    let dnf = x.to_dnf();
    assert_eq!(s(&dnf), "p > 0 & 0 >= q | q > 0 & 0 >= p");
    // equivalence of the forms
    assert_eq!(cnf.equivalent(&dnf).is_tautology(), Some(true));
    assert_eq!(x.equivalent(&cnf).is_tautology(), Some(true));
    // tautological clause disappears, contradiction stays false
    assert_eq!(s(&p.or(&p.not()).to_cnf()), "True");
    assert_eq!(s(&p.and(&p.not()).to_dnf()), "False");
}

// ═══════════════════════════════════════════════════════════════════════════
// Tautology / satisfiability / atoms / truth tables
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn propositional_decisions() {
    let ctx = Context::new();
    let p = ctx.symbol("p").gt(&ctx.int(0));
    let q = ctx.symbol("q").gt(&ctx.int(0));
    let r = ctx.symbol("r").gt(&ctx.int(0));
    // modus ponens, hypothetical syllogism, Peirce's law
    assert_eq!(p.implies(&q).and(&p).implies(&q).is_tautology(), Some(true));
    assert_eq!(
        p.implies(&q)
            .and(&q.implies(&r))
            .implies(&p.implies(&r))
            .is_tautology(),
        Some(true)
    );
    assert_eq!(
        p.implies(&q).implies(&p).implies(&p).is_tautology(),
        Some(true)
    );
    // contradiction and satisfiable
    assert_eq!(p.and(&p.not()).is_contradiction(), Some(true));
    assert_eq!(p.xor(&p).is_contradiction(), Some(true));
    // p, q are independent linear atoms in distinct symbols: exact answers
    assert_eq!(p.equivalent(&q).is_contradiction(), Some(false));
    assert_eq!(p.equivalent(&q).is_tautology(), Some(false));
    assert_eq!(p.is_tautology(), Some(false));
    assert_eq!(p.satisfiable(), Some(true));
    // atoms() returns the relational itself
    let a = p.atoms();
    assert_eq!(a, vec![p.clone()]);
}

#[test]
fn relational_decisions_are_exact_over_the_reals() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let gt1 = x.gt(&ctx.int(1));
    let gt0 = x.gt(&ctx.int(0));
    assert_eq!(gt1.implies(&gt0).is_tautology(), Some(true));
    assert_eq!(gt0.implies(&gt1).is_tautology(), Some(false));
    assert_eq!(gt1.and(&x.lt(&ctx.int(0))).is_contradiction(), Some(true));
    assert_eq!(gt1.and(&x.lt(&ctx.int(0))).satisfiable(), Some(false));
    assert_eq!(x.powi(2).ge(&ctx.int(0)).is_tautology(), Some(true));
    assert_eq!(x.powi(2).lt(&ctx.int(0)).is_contradiction(), Some(true));
    assert_eq!(
        x.powi(2)
            .gt(&ctx.int(4))
            .and(&x.lt(&ctx.int(5)))
            .satisfiable(),
        Some(true)
    );
    // x² > 4 ∧ |x| < 2 is a contradiction
    assert_eq!(
        x.powi(2)
            .gt(&ctx.int(4))
            .and(&x.abs().lt(&ctx.int(2)))
            .is_contradiction(),
        Some(true)
    );
    // independent linear atoms in distinct symbols: exact
    let y = ctx.symbol("y");
    assert_eq!(
        x.gt(&ctx.int(0)).and(&y.gt(&ctx.int(0))).satisfiable(),
        Some(true)
    );
    // dependent multivariate relationals: honest None
    assert_eq!(x.gt(&y).is_tautology(), None);
    assert_eq!(x.gt(&y).and(&x.gt(&ctx.int(0))).satisfiable(), None);
    assert_eq!(x.powi(2).gt(&y).satisfiable(), None);
    // non-linear single-symbol pairs are not treated as free
    assert_eq!(
        x.powi(2).gt(&ctx.int(0)).is_tautology(),
        Some(false),
        "fails at x = 0"
    );
    assert_eq!(
        x.powi(2)
            .gt(&ctx.int(0))
            .or(&x.eq_expr(&ctx.int(0)))
            .is_tautology(),
        Some(true)
    );
}

#[test]
fn decisions_respect_assumptions() {
    let ctx = Context::new();
    let t = ctx.symbol_with("t", &[Assumption::Positive]);
    let n = ctx.symbol_with("n", &[Assumption::Integer]);
    assert_eq!(t.gt(&ctx.int(0)).is_tautology(), Some(true));
    assert_eq!(t.gt(&ctx.int(-1)).is_tautology(), Some(true));
    assert_eq!(t.lt(&ctx.int(0)).satisfiable(), Some(false));
    assert_eq!(t.le(&ctx.int(0)).is_contradiction(), Some(true));
    // undecided rather than wrong: n² ≥ n holds for integers but not reals
    let claim = n.powi(2).ge(&n);
    assert_ne!(claim.is_tautology(), Some(false));
    // t < 5 is neither a tautology nor a contradiction for positive t;
    // the solver route is disabled for constrained symbols, so no guess
    assert_ne!(t.lt(&ctx.int(5)).is_tautology(), Some(true));
    assert_ne!(t.lt(&ctx.int(5)).is_contradiction(), Some(true));
    // an unconstrained symbol still gets the exact answer
    let x = ctx.symbol("x");
    assert_eq!(x.lt(&ctx.int(5)).is_tautology(), Some(false));
}

#[test]
fn atoms_and_truth_table() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let p = x.gt(&ctx.int(0));
    let q = x.lt(&ctx.int(5));
    let e = p.and(&q).or(&p.not());
    let atoms = e.atoms();
    assert_eq!(atoms.len(), 2);
    assert!(atoms.contains(&p) && atoms.contains(&q));
    let rows = e.truth_table(&[p.clone(), q.clone()]).unwrap();
    assert_eq!(
        rows,
        vec![
            (vec![false, false], true),
            (vec![false, true], true),
            (vec![true, false], false),
            (vec![true, true], true),
        ]
    );
    // relational variable and its negation are the same variable
    let rows = p
        .or(&p.not())
        .truth_table(std::slice::from_ref(&p))
        .unwrap();
    assert_eq!(rows, vec![(vec![false], true), (vec![true], true)]);
    // inconsistent rows are omitted: x > 0 and x < 0 both true is impossible
    let lt0 = x.lt(&ctx.int(0));
    let rows = p.or(&lt0).truth_table(&[p.clone(), lt0.clone()]).unwrap();
    assert_eq!(rows.len(), 3);
    // errors
    assert!(p.and(&q).truth_table(std::slice::from_ref(&p)).is_err());
    let many: Vec<BoolEx> = (0..9)
        .map(|i| ctx.symbol(&format!("v{i}")).gt(&ctx.int(0)))
        .collect();
    assert!(p.truth_table(&many).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// Piecewise
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn piecewise_simplify_cases() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pos = x.gt(&ctx.int(0));
    let nonpos = x.le(&ctx.int(0));
    let never = ctx.int(1).gt(&ctx.int(2));
    let always = ctx.int(2).gt(&ctx.int(1));
    let neg_x = -&x;

    // false branch dropped, identical adjacent values merged to `true`
    let pw = Ex::piecewise(&[(&x, &never), (&x, &pos), (&x, &nonpos)]);
    assert_eq!(pw.piecewise_simplify(), x);

    // stop at first true branch
    let pw = Ex::piecewise(&[(&x, &pos), (&neg_x, &always), (&ctx.int(7), &pos)]);
    let simp = pw.piecewise_simplify();
    let d = s(&simp);
    assert!(d.starts_with("Piecewise(") && !d.contains('7'), "{d}");

    // |x| as a piecewise stays a piecewise (two distinct values)
    let abs = Ex::piecewise(&[(&x, &pos), (&neg_x, &nonpos)]);
    assert_eq!(abs.piecewise_simplify(), abs);

    // repeated condition is unreachable
    let pw = Ex::piecewise(&[(&x, &pos), (&ctx.int(1), &pos), (&neg_x, &nonpos)]);
    assert_eq!(pw.piecewise_simplify(), abs);

    // nested inside arithmetic
    let inner = Ex::piecewise(&[(&ctx.int(1), &always)]);
    let e = &inner * &x + 2;
    assert_eq!(s(&e.piecewise_simplify()), "x + 2");

    // conditions are simplified
    let pw = Ex::piecewise(&[(&x, &pos.and(&pos.or(&nonpos))), (&neg_x, &always)]);
    let simp = pw.piecewise_simplify();
    assert_eq!(s(&simp), "Piecewise(x if x > 0, -x if True)");
}

// ═══════════════════════════════════════════════════════════════════════════
// Assumptions (0.2 additions)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn extended_real_assumption() {
    let ctx = Context::new();
    let r = ctx.symbol_with("r", &[Assumption::Real]);
    assert_eq!(r.query(Props::EXTENDED_REAL), Some(true));
    let e = ctx.symbol_with("e", &[Assumption::ExtendedReal]);
    assert_eq!(e.query(Props::REAL), None);
    let ef = ctx.symbol_with("f", &[Assumption::ExtendedReal, Assumption::Finite]);
    assert_eq!(ef.query(Props::REAL), Some(true));
    assert_eq!(ctx.infinity().query(Props::EXTENDED_REAL), Some(true));
    assert_eq!(ctx.infinity().query(Props::REAL), None);
    let ne = ctx.symbol_with("g", &[Assumption::NotExtendedReal]);
    assert_eq!(ne.query(Props::REAL), Some(false));
    assert_eq!(ne.query(Props::POSITIVE), Some(false));
}

#[test]
fn forward_chain_rules_via_symbols() {
    let ctx = Context::new();
    let p = ctx.symbol_with("p", &[Assumption::Prime]);
    assert_eq!(p.query(Props::INTEGER), Some(true));
    assert_eq!(p.query(Props::POSITIVE), Some(true));
    let e = ctx.symbol_with("e", &[Assumption::Even]);
    assert_eq!(e.query(Props::INTEGER), Some(true));
    let i = ctx.symbol_with("i", &[Assumption::Irrational]);
    assert_eq!(i.query(Props::REAL), Some(true));
    assert_eq!(i.query(Props::NONZERO), Some(true));
    assert_eq!(i.query(Props::INTEGER), Some(false));
    let t = ctx.symbol_with("t", &[Assumption::Transcendental]);
    assert_eq!(
        t.query(Props::IRRATIONAL),
        None,
        "complex transcendental numbers exist"
    );
    let tr = ctx.symbol_with("u", &[Assumption::Transcendental, Assumption::Real]);
    assert_eq!(tr.query(Props::IRRATIONAL), Some(true));
    let im = ctx.symbol_with("w", &[Assumption::Imaginary]);
    assert_eq!(im.query(Props::NONZERO), Some(true));
    assert_eq!(im.query(Props::REAL), Some(false));
    assert_eq!(im.query(Props::COMPLEX), Some(true));
    // contradiction is detectable on the assumption set
    let mut bad = Assumptions::default();
    bad.assert_true(Props::IRRATIONAL);
    bad.assert_true(Props::ZERO);
    assert!(bad.is_contradictory());
}

#[test]
fn assumptions_helpers() {
    let mut pos = Assumptions::default();
    pos.assert_true(Props::POSITIVE);
    let mut nn = Assumptions::default();
    nn.assert_true(Props::NONNEGATIVE);
    assert!(pos.implies(&nn));
    assert!(!nn.implies(&pos));
    assert_eq!(Assumption::Positive.negate(), Assumption::NotPositive);
    assert_eq!(Assumption::NotPositive.negate(), Assumption::Positive);
    assert_eq!(
        Assumption::ExtendedReal.negate(),
        Assumption::NotExtendedReal
    );
    let shown = s(&pos);
    assert!(
        shown.contains("positive") && shown.contains("!negative"),
        "{shown}"
    );
    assert_eq!(s(&Assumptions::default()), "unknown");
}

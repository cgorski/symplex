//! Set algebra, boolean logic and inequality reduction (symplex 0.2).
//!
//! Demonstrates:
//! - `SetEx` construction (`interval`, `finite_set`, `reals`) and algebra
//!   (`union`, `intersection`, `difference`, `symmetric_difference`,
//!   `absolute_complement`, `simplify`);
//! - three-valued set queries: `contains`, `is_subset`, `is_disjoint`,
//!   `is_empty`, `is_open`/`is_closed`;
//! - `inf`/`sup`/`measure`, `boundary`/`closure`/`interior`,
//!   `as_intervals`/`as_finite_set`, and `to_condition` (set → `BoolEx`);
//! - `reduce_inequalities` and `BoolEx::solve_for` (`BoolEx` → `SetEx`);
//! - `BoolEx` normal forms (`to_nnf/cnf/dnf`), `is_tautology`,
//!   `is_contradiction`, `satisfiable`, `atoms`, `truth_table`;
//! - `BoolEx::eval` folding relations through the assumption system;
//! - `Ex::piecewise_simplify`; `Assumptions::implies`, `Assumption::negate`.
//!
//! Run with: `cargo run --example sets_and_logic`

use symplex::prelude::*;

fn main() {
    println!("=== Sets and Logic ===\n");

    let ctx = Context::new();
    symplex::syms!(ctx; x, p, q, r);
    let int = |n: i64| ctx.int(n);

    // ── 1. Intervals and set algebra ────────────────────────────────────
    println!("--- Set algebra ---");
    // interval(lo, hi, left_open, right_open)
    let a = ctx.interval(&int(0), &int(5), false, false); // [0, 5]
    let b = ctx.interval(&int(3), &int(10), true, false); // (3, 10]
    println!("A = {a}    B = {b}");
    // Construction is cheap and lazy; `simplify()` computes the normal form.
    println!("A ∪ B = {}  →  {}", a.union(&b), a.union(&b).simplify());
    println!(
        "A ∩ B = {}  →  {}",
        a.intersection(&b),
        a.intersection(&b).simplify()
    );
    println!("A \\ B = {}", a.difference(&b));
    println!("A Δ B = {}", a.symmetric_difference(&b));
    println!("ℝ \\ A = {}", a.absolute_complement());
    let s = ctx.finite_set(&[int(1), int(2), int(3), int(7)]);
    println!("S = {s}");
    println!("S ∩ A = {}", s.intersection(&a).simplify());
    println!("S ∪ A = {}", s.union(&a).simplify());
    println!("S \\ {{2}} = {}", s.difference(&ctx.finite_set(&[int(2)])));
    println!("ℝ = {}", ctx.reals());

    // ── 2. Queries (three-valued) ───────────────────────────────────────
    println!("\n--- Queries ---");
    println!(
        "3 ∈ A: {:?}    7 ∈ A: {:?}    x ∈ A: {:?}",
        a.contains(&int(3)),
        a.contains(&int(7)),
        a.contains(&x)
    );
    println!(
        "A ⊆ [0, 10]: {:?}    A ∩ (5, 6) = ∅: {:?}",
        a.is_subset(&ctx.interval(&int(0), &int(10), false, false)),
        a.is_disjoint(&ctx.interval(&int(5), &int(6), true, true))
    );
    let empty = a.intersection(&ctx.interval(&int(6), &int(7), false, false));
    println!("{empty} is empty: {:?}", empty.is_empty());
    println!(
        "B is open: {:?}    A is closed: {:?}",
        b.is_open(),
        a.is_closed()
    );
    let ab = a.union(&b).simplify();
    println!(
        "inf/sup/measure of {ab}: {}, {}, {}",
        ab.inf().unwrap(),
        ab.sup().unwrap(),
        ab.measure().unwrap()
    );
    println!(
        "∂A = {}    closure(B) = {}    interior(A) = {}",
        a.boundary().unwrap(),
        b.closure().unwrap(),
        a.interior().unwrap()
    );

    // ── 3. Between sets and conditions ──────────────────────────────────
    println!("\n--- Sets ↔ conditions ---");
    for (lo, hi, lopen, ropen) in ab.as_intervals().unwrap() {
        let l = if lopen { "(" } else { "[" };
        let r = if ropen { ")" } else { "]" };
        println!("as_intervals: {l}{lo}, {hi}{r}");
    }
    println!(
        "as_finite_set(S) = {:?}",
        s.as_finite_set()
            .unwrap()
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
    );
    println!("to_condition(A, x) = {}", a.to_condition(&x).unwrap());
    println!(
        "to_condition(A ∪ B, x) = {}",
        a.union(&b).to_condition(&x).unwrap()
    );

    // ── 4. Inequality reduction ─────────────────────────────────────────
    println!("\n--- reduce_inequalities / solve_for ---");
    let conds = [x.gt(&int(0)), x.le(&int(5)), (&x.powi(2) - 4).gt(&int(0))];
    println!(
        "x > 0 ∧ x ≤ 5 ∧ x² − 4 > 0   →  {}",
        reduce_inequalities(&conds, &x).unwrap()
    );
    let cond = x.gt(&int(0)).and(&x.lt(&int(3)));
    println!("({cond}).solve_for(x)  →  {}", cond.solve_for(&x).unwrap());
    let cond = (&x.powi(2) - 1).ge(&int(0));
    println!("({cond}).solve_for(x)  →  {}", cond.solve_for(&x).unwrap());
    let sol = (&x.powi(2) - 4).solve_gt(&x);
    println!(
        "x² − 4 > 0: {sol};  3 ∈ sol: {:?},  0 ∈ sol: {:?}",
        int(3).is_in(&sol),
        int(0).is_in(&sol)
    );

    // ── 5. Boolean logic ────────────────────────────────────────────────
    println!("\n--- Boolean logic ---");
    // Atoms are relations; here p, q, r stand in via `p > 0` etc.
    let (pp, qq, rr) = (p.gt(&int(0)), q.gt(&int(0)), r.gt(&int(0)));
    println!("¬(p ∧ q) in NNF      = {}", pp.and(&qq).not().to_nnf());
    println!("(p ∧ q) ∨ r in CNF   = {}", pp.and(&qq).or(&rr).to_cnf());
    println!("(p ∨ q) ∧ r in DNF   = {}", pp.or(&qq).and(&rr).to_dnf());
    println!(
        "(p ∧ q) ∨ p simplify = {}   (absorption)",
        pp.and(&qq).or(&pp).simplify()
    );
    println!(
        "p ∨ (¬p ∧ q) simplify = {}",
        pp.or(&pp.not().and(&qq)).simplify()
    );
    println!(
        "p ∨ ¬p tautology:     {:?}",
        pp.or(&pp.not()).is_tautology()
    );
    println!(
        "p ∧ ¬p contradiction: {:?}",
        pp.and(&pp.not()).is_contradiction()
    );
    println!("p ∧ q satisfiable:    {:?}", pp.and(&qq).satisfiable());
    println!(
        "(p ∧ q) → p tautology: {:?}",
        pp.and(&qq).implies(&pp).is_tautology()
    );
    let f = pp.and(&qq).or(&pp.and(&qq.not()));
    println!(
        "atoms of {f}: {:?}",
        f.atoms().iter().map(|a| a.to_string()).collect::<Vec<_>>()
    );
    println!("truth table of p ∧ q:");
    for (inputs, out) in pp.and(&qq).truth_table(&[pp.clone(), qq.clone()]).unwrap() {
        println!("   p={:<5} q={:<5} → {out}", inputs[0], inputs[1]);
    }

    // ── 6. Relations fold through assumptions ───────────────────────────
    println!("\n--- BoolEx::eval with assumptions ---");
    let pos = ctx.symbol_with("pos", &[Assumption::Positive]);
    let t = ctx.symbol_with("t", &[Assumption::Real]);
    println!("pos > 0   →  {}", pos.gt(&int(0)).eval());
    println!("pos < 0   →  {}", pos.lt(&int(0)).eval());
    println!("t² ≥ 0    →  {}   (t real)", t.powi(2).ge(&int(0)).eval());
    println!(
        "x² + 1 > 0 →  {}   (x unassumed: could be complex)",
        (&x.powi(2) + 1).gt(&int(0)).eval()
    );

    // ── 7. Piecewise simplification ─────────────────────────────────────
    println!("\n--- piecewise_simplify ---");
    let never = int(1).gt(&int(2));
    let pw = Ex::piecewise(&[
        (&x, &never),
        (&x.powi(2), &x.gt(&int(0))),
        (&x.powi(2), &x.le(&int(0))),
    ]);
    println!("{pw}\n  →  {}", pw.piecewise_simplify());

    // ── 8. Assumptions as values ────────────────────────────────────────
    println!("\n--- Assumptions ---");
    let mut positive = Assumptions::default();
    positive.assert_true(Props::POSITIVE);
    println!("positive forward-chains to: {positive}");
    let nonneg = Assumptions {
        known_true: Props::NONNEGATIVE,
        ..Assumptions::default()
    };
    println!("positive ⇒ nonnegative: {}", positive.implies(&nonneg));
    println!("negate(Positive) = {:?}", Assumption::Positive.negate());
    let e = ctx.symbol_with("e", &[Assumption::ExtendedReal]);
    println!(
        "ExtendedReal symbol e: is_real {:?}, is_finite {:?}  (may be ±∞)",
        e.is_real(),
        e.is_finite()
    );

    println!("\n✓ Done!");
}

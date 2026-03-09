//! Tests for extended logic connectives: xor, implies, equivalent, nand, nor, ite.
//!
//! These operations are composed from And/Or/Not — no new ExprNode variants needed.

use symplex::prelude::*;

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Shared context for all logic-ext helpers so `bool_true()` and
/// `bool_false()` produce expressions in the same context.
fn logic_ctx() -> Context {
    static CTX: std::sync::OnceLock<Context> = std::sync::OnceLock::new();
    CTX.get_or_init(Context::new).clone()
}

/// A boolean expression that evaluates to True.
fn bool_true() -> BoolEx {
    let __ctx = logic_ctx();
    __ctx.int(1).gt(&__ctx.int(0))
}

/// A boolean expression that evaluates to False.
fn bool_false() -> BoolEx {
    let __ctx = logic_ctx();
    __ctx.int(0).gt(&__ctx.int(1))
}

fn eval_str(b: &BoolEx) -> String {
    format!("{}", b.eval())
}

// ═══════════════════════════════════════════════════════════════════════════════
// XOR
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn xor_truth_table() {
    let t = bool_true();
    let f = bool_false();

    // T xor T = F
    assert_eq!(eval_str(&t.xor(&t)), "False", "xor(T, T)");
    // T xor F = T
    assert_eq!(eval_str(&t.xor(&f)), "True", "xor(T, F)");
    // F xor T = T
    assert_eq!(eval_str(&f.xor(&t)), "True", "xor(F, T)");
    // F xor F = F
    assert_eq!(eval_str(&f.xor(&f)), "False", "xor(F, F)");
}

#[test]
fn xor_is_commutative() {
    let __ctx = Context::new();
    let a = __ctx.int(5).gt(&__ctx.int(0)); // true
    let b = __ctx.int(5).lt(&__ctx.int(0)); // false

    assert_eq!(eval_str(&a.xor(&b)), eval_str(&b.xor(&a)));
}

// ═══════════════════════════════════════════════════════════════════════════════
// IMPLIES
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn implies_truth_table() {
    let t = bool_true();
    let f = bool_false();

    // T → T = T
    assert_eq!(eval_str(&t.implies(&t)), "True", "T → T");
    // T → F = F
    assert_eq!(eval_str(&t.implies(&f)), "False", "T → F");
    // F → T = T
    assert_eq!(eval_str(&f.implies(&t)), "True", "F → T");
    // F → F = T
    assert_eq!(eval_str(&f.implies(&f)), "True", "F → F");
}

#[test]
fn implies_false_antecedent_always_true() {
    // "ex falso quodlibet" — false implies anything is true
    let f = bool_false();
    let t = bool_true();

    assert_eq!(eval_str(&f.implies(&t)), "True");
    assert_eq!(eval_str(&f.implies(&f)), "True");
}

// ═══════════════════════════════════════════════════════════════════════════════
// EQUIVALENT (biconditional)
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn equivalent_truth_table() {
    let t = bool_true();
    let f = bool_false();

    // T ↔ T = T
    assert_eq!(eval_str(&t.equivalent(&t)), "True", "T ↔ T");
    // T ↔ F = F
    assert_eq!(eval_str(&t.equivalent(&f)), "False", "T ↔ F");
    // F ↔ T = F
    assert_eq!(eval_str(&f.equivalent(&t)), "False", "F ↔ T");
    // F ↔ F = T
    assert_eq!(eval_str(&f.equivalent(&f)), "True", "F ↔ F");
}

#[test]
fn equivalent_is_commutative() {
    let t = bool_true();
    let f = bool_false();

    assert_eq!(eval_str(&t.equivalent(&f)), eval_str(&f.equivalent(&t)),);
}

// ═══════════════════════════════════════════════════════════════════════════════
// NAND
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn nand_truth_table() {
    let t = bool_true();
    let f = bool_false();

    // nand(T, T) = F
    assert_eq!(eval_str(&t.nand(&t)), "False", "nand(T, T)");
    // nand(T, F) = T
    assert_eq!(eval_str(&t.nand(&f)), "True", "nand(T, F)");
    // nand(F, T) = T
    assert_eq!(eval_str(&f.nand(&t)), "True", "nand(F, T)");
    // nand(F, F) = T
    assert_eq!(eval_str(&f.nand(&f)), "True", "nand(F, F)");
}

// ═══════════════════════════════════════════════════════════════════════════════
// NOR
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn nor_truth_table() {
    let t = bool_true();
    let f = bool_false();

    // nor(T, T) = F
    assert_eq!(eval_str(&t.nor(&t)), "False", "nor(T, T)");
    // nor(T, F) = F
    assert_eq!(eval_str(&t.nor(&f)), "False", "nor(T, F)");
    // nor(F, T) = F
    assert_eq!(eval_str(&f.nor(&t)), "False", "nor(F, T)");
    // nor(F, F) = T
    assert_eq!(eval_str(&f.nor(&f)), "True", "nor(F, F)");
}

// ═══════════════════════════════════════════════════════════════════════════════
// ITE (if-then-else)
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn ite_selects_then_when_true() {
    let cond = bool_true();
    let then_ = bool_true();
    let else_ = bool_false();

    assert_eq!(eval_str(&cond.ite(&then_, &else_)), "True");
}

#[test]
fn ite_selects_else_when_false() {
    let cond = bool_false();
    let then_ = bool_true();
    let else_ = bool_false();

    assert_eq!(eval_str(&cond.ite(&then_, &else_)), "False");
}

#[test]
fn ite_all_combinations() {
    let t = bool_true();
    let f = bool_false();

    // cond=T → picks then_
    assert_eq!(eval_str(&t.ite(&t, &f)), "True", "ite(T, T, F)");
    assert_eq!(eval_str(&t.ite(&f, &t)), "False", "ite(T, F, T)");
    assert_eq!(eval_str(&t.ite(&t, &t)), "True", "ite(T, T, T)");
    assert_eq!(eval_str(&t.ite(&f, &f)), "False", "ite(T, F, F)");

    // cond=F → picks else_
    assert_eq!(eval_str(&f.ite(&t, &f)), "False", "ite(F, T, F)");
    assert_eq!(eval_str(&f.ite(&f, &t)), "True", "ite(F, F, T)");
    assert_eq!(eval_str(&f.ite(&t, &t)), "True", "ite(F, T, T)");
    assert_eq!(eval_str(&f.ite(&f, &f)), "False", "ite(F, F, F)");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Cross-connective identities
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn xor_equivalent_are_complementary() {
    // xor(a,b) = ¬equivalent(a,b) for all truth-value combinations
    let pairs: Vec<(BoolEx, BoolEx)> = vec![
        (bool_true(), bool_true()),
        (bool_true(), bool_false()),
        (bool_false(), bool_true()),
        (bool_false(), bool_false()),
    ];

    for (a, b) in &pairs {
        let xor_val = eval_str(&a.xor(b));
        let not_equiv = eval_str(&a.equivalent(b).not());
        assert_eq!(xor_val, not_equiv, "xor(a,b) should equal ¬equivalent(a,b)");
    }
}

#[test]
fn nand_from_not_and() {
    // nand(a,b) = ¬(a ∧ b) — verify equivalence with raw not-and
    let pairs: Vec<(BoolEx, BoolEx)> = vec![
        (bool_true(), bool_true()),
        (bool_true(), bool_false()),
        (bool_false(), bool_true()),
        (bool_false(), bool_false()),
    ];

    for (a, b) in &pairs {
        let via_method = eval_str(&a.nand(b));
        let via_manual = eval_str(&a.and(b).not());
        assert_eq!(via_method, via_manual, "nand should equal not-and");
    }
}

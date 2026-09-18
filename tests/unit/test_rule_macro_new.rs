//! Integration tests for the `rule!` macro with new function types added in
//! recent waves (floor/ceiling, special functions, Heaviside/Dirac, binary
//! functions like beta and atan2).

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Wave B: Floor / Ceiling
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rule_with_floor() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let r = rule!(arena, "test_floor", floor(w_) => w_);
        assert_eq!(r.name, "test_floor");
        assert!(!r.pattern.wilds.is_empty(), "should have wildcard w_");
    });
}

#[test]
fn rule_with_ceiling() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let r = rule!(arena, "test_ceiling", ceiling(w_) => w_);
        assert_eq!(r.name, "test_ceiling");
        assert!(!r.pattern.wilds.is_empty(), "should have wildcard w_");
    });
}

#[test]
fn rule_floor_of_integer_pattern() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        // floor(w_ + 1) => w_ (nonsense math, just testing nested expr)
        let r = rule!(arena, "floor_shift", floor(w_ + 1) => w_);
        assert_eq!(r.name, "floor_shift");
        assert!(!r.pattern.wilds.is_empty());
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Wave J: Special functions (unary)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rule_with_gamma() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let r = rule!(arena, "test_gamma", gamma(w_) => w_);
        assert_eq!(r.name, "test_gamma");
        assert!(!r.pattern.wilds.is_empty(), "should have wildcard w_");
    });
}

#[test]
fn rule_with_log_gamma() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let r = rule!(arena, "test_log_gamma", log_gamma(w_) => w_);
        assert_eq!(r.name, "test_log_gamma");
        assert!(!r.pattern.wilds.is_empty(), "should have wildcard w_");
    });
}

#[test]
fn rule_with_digamma() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let r = rule!(arena, "test_digamma", digamma(w_) => w_);
        assert_eq!(r.name, "test_digamma");
        assert!(!r.pattern.wilds.is_empty(), "should have wildcard w_");
    });
}

#[test]
fn rule_with_erf() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let r = rule!(arena, "test_erf", erf(w_) => w_);
        assert_eq!(r.name, "test_erf");
        assert!(!r.pattern.wilds.is_empty(), "should have wildcard w_");
    });
}

#[test]
fn rule_with_erfc() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let r = rule!(arena, "test_erfc", erfc(w_) => w_);
        assert_eq!(r.name, "test_erfc");
        assert!(!r.pattern.wilds.is_empty(), "should have wildcard w_");
    });
}

#[test]
fn rule_erf_complement_identity() {
    // erf(w_) + erfc(w_) => 1 — a real identity!
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let r = rule!(arena, "erf_erfc_sum", erf(w_) + erfc(w_) => 1);
        assert_eq!(r.name, "erf_erfc_sum");
        assert!(!r.pattern.wilds.is_empty());
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Wave δ: Heaviside / DiracDelta
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rule_with_heaviside() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let r = rule!(arena, "test_heaviside", heaviside(w_) => w_);
        assert_eq!(r.name, "test_heaviside");
        assert!(!r.pattern.wilds.is_empty(), "should have wildcard w_");
    });
}

#[test]
fn rule_with_dirac_delta() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let r = rule!(arena, "test_dirac", dirac_delta(w_) => w_);
        assert_eq!(r.name, "test_dirac");
        assert!(!r.pattern.wilds.is_empty(), "should have wildcard w_");
    });
}

#[test]
fn rule_dirac_derivative_pattern() {
    // Pattern with dirac_delta nested inside another function
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let r = rule!(arena, "dirac_nested", exp(dirac_delta(w_)) => 1);
        assert_eq!(r.name, "dirac_nested");
        assert!(!r.pattern.wilds.is_empty());
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Wave S: LambertW
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rule_with_lambertw() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let r = rule!(arena, "test_lambertw", lambertw(w_) => w_);
        assert_eq!(r.name, "test_lambertw");
        assert!(!r.pattern.wilds.is_empty(), "should have wildcard w_");
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Binary functions: beta, atan2
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rule_with_beta() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        // beta(a_, b_) => a_ (nonsense, just testing 2-arg dispatch)
        let r = rule!(arena, "test_beta", beta(a_, b_) => a_);
        assert_eq!(r.name, "test_beta");
        assert!(
            r.pattern.wilds.len() >= 2,
            "should have wildcards a_ and b_"
        );
    });
}

#[test]
fn rule_with_atan2() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        // atan2(y_, x_) => y_ (nonsense, just testing 2-arg dispatch)
        let r = rule!(arena, "test_atan2", atan2(y_, x_) => y_);
        assert_eq!(r.name, "test_atan2");
        assert!(
            r.pattern.wilds.len() >= 2,
            "should have wildcards y_ and x_"
        );
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Combinatorial functions in rule! (1-arg Apply-based with arena methods)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rule_with_fibonacci() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let r = rule!(arena, "test_fib", fibonacci(w_) => w_);
        assert_eq!(r.name, "test_fib");
        assert!(!r.pattern.wilds.is_empty());
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Compound / mixed patterns
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rule_gamma_log_gamma_relation() {
    // ln(gamma(w_)) => log_gamma(w_)   — a real simplification!
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let r = rule!(arena, "ln_gamma", ln(gamma(w_)) => log_gamma(w_));
        assert_eq!(r.name, "ln_gamma");
        assert!(!r.pattern.wilds.is_empty());
    });
}

#[test]
fn rule_heaviside_derivative() {
    // Derivative of Heaviside is Dirac delta (pattern-level only)
    // This just tests that both functions can appear in the same rule.
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let r = rule!(arena, "heaviside_deriv", heaviside(w_) => dirac_delta(w_));
        assert_eq!(r.name, "heaviside_deriv");
        assert!(!r.pattern.wilds.is_empty());
    });
}

#[test]
fn rule_floor_ceiling_identity() {
    // ceiling(w_) => floor(w_) + 1  (wrong in general, but tests both in one rule)
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let r = rule!(arena, "ceil_floor", ceiling(w_) => floor(w_) + 1);
        assert_eq!(r.name, "ceil_floor");
        assert!(!r.pattern.wilds.is_empty());
    });
}

#[test]
fn rule_erf_of_zero_constant() {
    // erf(0) => 0  (mathematically true!)
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let r = rule!(arena, "erf_zero", erf(0) => 0);
        assert_eq!(r.name, "erf_zero");
        assert!(r.pattern.wilds.is_empty(), "no wildcards in this rule");
    });
}

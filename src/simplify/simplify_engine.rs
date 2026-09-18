//! Simplification engine — try multiple strategies, keep shortest.
//!
//! This module provides [`count_ops`] for measuring expression complexity,
//! [`smart_simplify`] which tries multiple simplification strategies
//! and returns the result with the lowest operation count, and
//! [`unified_simplify`] which iterates `smart_simplify` to a fixpoint.

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};
use crate::base::walk;
use crate::transforms::pattern::RawStep;
use num_bigint::BigInt;
use num_traits::Signed;
use rustc_hash::FxHashSet;

// ── Public configuration types ─────────────────────────────────────────

/// Options for configuring the simplification engine.
#[derive(Clone, Debug)]
pub struct SimplifyOpts {
    /// Maximum number of fixpoint iterations (default: 10).
    /// Each iteration runs the full `smart_simplify` strategy set.
    pub max_iterations: usize,
    /// Whether to collect a trace of the strategies and rewrite rules that
    /// changed the expression (default: false).
    ///
    /// The trace is returned by
    /// [`Ex::simplify_traced`](crate::api::expr::Ex::simplify_traced) (which
    /// switches this flag on automatically); `simplify_with` computes the
    /// trace when the flag is set but has no way to return it, so the
    /// flag is harmless there.
    pub trace: bool,
}

impl Default for SimplifyOpts {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            trace: false,
        }
    }
}

impl SimplifyOpts {
    /// Single-pass simplification (no fixpoint iteration).
    pub fn single_pass() -> Self {
        Self {
            max_iterations: 1,
            trace: false,
        }
    }

    /// Set the maximum number of fixpoint iterations.
    #[must_use]
    pub fn max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Request a rewrite trace.
    #[must_use]
    pub fn trace(mut self) -> Self {
        self.trace = true;
        self
    }
}

/// Result of a simplification run.
#[derive(Clone, Debug)]
pub struct SimplifyResult {
    /// The simplified expression.
    pub expr: ExprId,
    /// Trace of the strategies / rules that changed the expression
    /// (empty unless [`SimplifyOpts::trace`] was set).
    pub steps: Vec<RawStep>,
}

/// Count the number of operations (nodes) in an expression.
///
/// Each non-atom node counts as 1. Atoms (numbers, symbols, constants)
/// count as 0. This gives a rough measure of expression complexity.
pub(crate) fn count_ops(arena: &Arena, expr: ExprId) -> usize {
    let post_order = walk::post_order_ids(arena, expr);
    let mut count = 0;
    for &id in &post_order {
        if !arena.node(id).is_atom() {
            count += 1;
        }
    }
    count
}

// ── Type-based strategy gating ─────────────────────────────────────────

/// Flags indicating which node types appear in an expression.
/// Used to skip irrelevant simplification strategies.
struct ExprFlags {
    has_trig: bool,
    has_hyp: bool,
    has_exp_ln: bool,
    has_neg_pow: bool,
    has_add: bool,
    has_apply: bool,
    is_atom: bool,
    node_count: usize,
    /// Expression contains `Abs` nodes — gates refine strategy.
    has_abs: bool,
    /// Expression contains `Sign` nodes — gates refine strategy.
    has_sign: bool,
    /// Expression contains `Floor` or `Ceiling` nodes — gates refine strategy.
    has_floor_ceil: bool,
    /// Expression contains `Pow` nodes (general, not just negative exponent).
    has_pow: bool,
    /// Expression contains `Factorial` or `Binomial` nodes — gates combsimp strategy.
    has_factorial: bool,
    /// Expression contains `Gamma` or `LogGamma` nodes — gates combsimp strategy.
    has_gamma: bool,
    /// Expression contains `Pow` with exponent 1/2 (square root) — gates radsimp strategy.
    has_sqrt: bool,
}

/// Compute expression flags in a single tree walk.
fn compute_flags(arena: &Arena, expr: ExprId) -> ExprFlags {
    let post_order = crate::base::walk::post_order_ids(arena, expr);
    let mut flags = ExprFlags {
        has_trig: false,
        has_hyp: false,
        has_exp_ln: false,
        has_neg_pow: false,
        has_add: false,
        has_apply: false,
        is_atom: post_order.len() == 1 && arena.node(expr).is_atom(),
        node_count: post_order.len(),
        has_abs: false,
        has_sign: false,
        has_floor_ceil: false,
        has_pow: false,
        has_factorial: false,
        has_gamma: false,
        has_sqrt: false,
    };

    for &id in &post_order {
        match arena.node(id) {
            ExprNode::Sin(_)
            | ExprNode::Cos(_)
            | ExprNode::Tan(_)
            | ExprNode::Asin(_)
            | ExprNode::Acos(_)
            | ExprNode::Atan(_) => {
                flags.has_trig = true;
            }
            ExprNode::Sinh(_)
            | ExprNode::Cosh(_)
            | ExprNode::Tanh(_)
            | ExprNode::Asinh(_)
            | ExprNode::Acosh(_)
            | ExprNode::Atanh(_) => {
                flags.has_hyp = true;
            }
            ExprNode::Exp(_) | ExprNode::Ln(_) => {
                flags.has_exp_ln = true;
            }
            ExprNode::Pow(_, exp) => {
                flags.has_pow = true;
                if let Some(r) = arena.as_num(*exp) {
                    if r.is_negative() {
                        flags.has_neg_pow = true;
                    }
                    // Check for sqrt: exponent == 1/2
                    if *r.numer() == BigInt::from(1) && *r.denom() == BigInt::from(2) {
                        flags.has_sqrt = true;
                    }
                }
            }
            ExprNode::Factorial(_) | ExprNode::Binomial(_, _) => {
                flags.has_factorial = true;
            }
            ExprNode::Gamma(_) | ExprNode::LogGamma(_) => {
                flags.has_gamma = true;
            }
            ExprNode::Add(_) => flags.has_add = true,
            ExprNode::Apply(_, _) => flags.has_apply = true,
            ExprNode::Abs(_) => flags.has_abs = true,
            ExprNode::Sign(_) => flags.has_sign = true,
            ExprNode::Floor(_) | ExprNode::Ceiling(_) => flags.has_floor_ceil = true,
            _ => {}
        }
    }

    flags
}

/// The best candidate found so far by [`smart_simplify`], with the name
/// of the strategy that produced it and the rule steps that fired inside
/// that strategy (for tracing).
struct Best {
    expr: ExprId,
    ops: usize,
    strategy: &'static str,
    steps: Vec<RawStep>,
}

impl Best {
    /// Replace the best candidate if `candidate` has a strictly lower op-count.
    fn consider(
        &mut self,
        arena: &Arena,
        candidate: ExprId,
        strategy: &'static str,
        steps: Vec<RawStep>,
    ) {
        let ops = count_ops(arena, candidate);
        tracing::trace!(strategy, ops, "strategy evaluated");
        if ops < self.ops {
            self.expr = candidate;
            self.ops = ops;
            self.strategy = strategy;
            self.steps = steps;
        }
    }

    /// Like [`consider`](Self::consider), but accepts equal op-count candidates too.
    ///
    /// Used for Strategy 1 (eval) so that canonical forms like `-sin(x)` win
    /// over `sin(-x)` even when they have the same number of operations.
    fn consider_or_equal(
        &mut self,
        arena: &Arena,
        candidate: ExprId,
        strategy: &'static str,
        steps: Vec<RawStep>,
    ) {
        let ops = count_ops(arena, candidate);
        tracing::trace!(strategy, ops, "strategy evaluated");
        if ops <= self.ops && candidate != self.expr {
            self.expr = candidate;
            self.ops = ops;
            self.strategy = strategy;
            self.steps = steps;
        }
    }
}

/// Try multiple simplification strategies and return the "simplest" result.
///
/// Strategies tried:
/// 1. eval only
/// 2. eval → simplify (pattern rules)
/// 3. eval → expand → simplify
/// 4. eval → factor_terms → simplify
/// 5. eval → expand_trig → simplify
/// 6. eval → logcombine → simplify
/// 7. eval → cancel with free symbols → simplify
/// 8. eval → refine (assumption-aware, e.g. abs(x)→x when x≥0)
///
/// The result with the lowest `count_ops` is returned.
/// If the best result is more than 1.7× the complexity of the original,
/// the original is returned (to prevent "simplification" that makes things worse).
pub(crate) fn smart_simplify(arena: &mut Arena, expr: ExprId) -> ExprId {
    smart_simplify_traced(arena, expr).0
}

/// [`smart_simplify`] that also reports which strategy produced the result
/// and the rewrite-rule steps that fired inside it.
///
/// The returned `&'static str` is the winning strategy name (`"none"`
/// when the expression was returned unchanged).
pub(crate) fn smart_simplify_traced(
    arena: &mut Arena,
    expr: ExprId,
) -> (ExprId, &'static str, Vec<RawStep>) {
    let flags = compute_flags(arena, expr);

    // Early exit for atoms — nothing to simplify
    if flags.is_atom {
        tracing::debug!("smart_simplify: early exit for atom expression");
        return (expr, "none", Vec::new());
    }

    let original_ops = count_ops(arena, expr);
    let mut best = Best {
        expr,
        ops: original_ops,
        strategy: "none",
        steps: Vec::new(),
    };

    // Strategy 1: eval only (always try — cheap)
    // Use consider_or_equal so that eval's canonical forms (e.g. -sin(x)
    // over sin(-x)) win even at equal op count.  This fixes Bug 14 (odd
    // function parity normalisation).
    let s1 = crate::transforms::eval::eval(arena, expr);
    best.consider_or_equal(arena, s1, "eval", Vec::new());

    // Strategy 2: eval → pattern rules (only if trig/exp/hyp present)
    if flags.has_trig || flags.has_exp_ln || flags.has_hyp {
        let rules = crate::transforms::pattern::basic_rules(arena);
        let s2_eval = crate::transforms::eval::eval(arena, expr);
        let (s2, steps) = crate::transforms::pattern::apply_rules(arena, s2_eval, &rules);
        best.consider_or_equal(arena, s2, "eval+rules", steps);
    } else {
        tracing::debug!("smart_simplify: skipping pattern rules (no trig/exp/hyp nodes)");
    }

    // Strategy 3: eval → expand → simplify (always try — helps with polynomial algebra)
    {
        let rules = crate::transforms::pattern::basic_rules(arena);
        let s3_eval = crate::transforms::eval::eval(arena, expr);
        let s3_expand = crate::transforms::expand::expand(arena, s3_eval);
        let s3_eval2 = crate::transforms::eval::eval(arena, s3_expand);
        let (s3, steps) = crate::transforms::pattern::apply_rules(arena, s3_eval2, &rules);
        best.consider(arena, s3, "eval+expand+rules", steps);
    }

    // Strategy 4: eval → factor_terms (symbolic) → simplify (only if has Add)
    if flags.has_add {
        let rules = crate::transforms::pattern::basic_rules(arena);
        let s4_eval = crate::transforms::eval::eval(arena, expr);
        let (gcd_id, s4_inner) =
            crate::simplify::factor_terms::symbolic_factor_terms_pair(arena, s4_eval);
        let (s4_simplified, steps) =
            crate::transforms::pattern::apply_rules(arena, s4_inner, &rules);
        let s4 = if gcd_id == arena.one {
            s4_simplified
        } else {
            arena.mul(&[gcd_id, s4_simplified])
        };
        best.consider(arena, s4, "eval+factor+rules", steps);
    } else {
        tracing::debug!("smart_simplify: skipping factor_terms (no Add nodes)");
    }

    // Strategy 5: eval → expand_trig → simplify (only if has trig)
    if flags.has_trig {
        let rules = crate::transforms::pattern::basic_rules(arena);
        let s5_eval = crate::transforms::eval::eval(arena, expr);
        let s5_trig = crate::simplify::trig_expand::expand_trig(arena, s5_eval);
        let s5_eval2 = crate::transforms::eval::eval(arena, s5_trig);
        let (s5, steps) = crate::transforms::pattern::apply_rules(arena, s5_eval2, &rules);
        best.consider(arena, s5, "eval+trig_expand+rules", steps);
    } else {
        tracing::debug!("smart_simplify: skipping trig_expand (no trig nodes)");
    }

    // Strategy 5b: eval → fu (Fu's trig algorithm — only if has trig)
    if flags.has_trig {
        let s5b_eval = crate::transforms::eval::eval(arena, expr);
        let s5b_fu = crate::simplify::fu::fu(arena, s5b_eval);
        best.consider(arena, s5b_fu, "eval+fu", Vec::new());
    } else {
        tracing::debug!("smart_simplify: skipping fu (no trig nodes)");
    }

    // Strategy 5c: eval → factor_terms → fu → reassemble (trig + add)
    //
    // Handles cases like 2·sin²(x) + 2·cos²(x) → 2·(sin²+cos²) → 2·1 = 2
    // where the common factor must be extracted before fu can fire.
    if flags.has_trig && flags.has_add {
        let s5c_eval = crate::transforms::eval::eval(arena, expr);
        let (gcd_id, s5c_inner) =
            crate::simplify::factor_terms::symbolic_factor_terms_pair(arena, s5c_eval);
        let s5c_fu = crate::simplify::fu::fu(arena, s5c_inner);
        let s5c = if gcd_id == arena.one {
            s5c_fu
        } else {
            arena.mul(&[gcd_id, s5c_fu])
        };
        best.consider(arena, s5c, "eval+factor+fu", Vec::new());
    } else {
        tracing::debug!("smart_simplify: skipping factor+fu (no trig+add nodes)");
    }

    // Strategy 5d: eval → trig identity rules (sum/difference, double angle,
    // hyperbolic double angle) → eval (only if trig/hyperbolic present)
    if flags.has_trig || flags.has_hyp {
        let rules = crate::simplify::trigsimp::trig_identity_rules(arena);
        let s5d_eval = crate::transforms::eval::eval(arena, expr);
        let (s5d_rules, steps) = crate::transforms::pattern::apply_rules(arena, s5d_eval, &rules);
        let s5d = crate::transforms::eval::eval(arena, s5d_rules);
        best.consider(arena, s5d, "eval+trig_identities", steps);
    }

    // Strategy 6: eval → logcombine → simplify (only if has Ln nodes)
    if flags.has_exp_ln {
        let rules = crate::transforms::pattern::basic_rules(arena);
        let s6_eval = crate::transforms::eval::eval(arena, expr);
        let s6_log = crate::simplify::log_combine::log_combine(arena, s6_eval);
        let (s6, steps) = crate::transforms::pattern::apply_rules(arena, s6_log, &rules);
        best.consider(arena, s6, "eval+logcombine+rules", steps);
    } else {
        tracing::debug!("smart_simplify: skipping logcombine (no exp/ln nodes)");
    }

    // Strategy 7: eval → cancel with free symbols → simplify
    // (Always run — cancel can simplify rational expressions even without
    // visible negative-power nodes, e.g. after substitution or expand.)
    {
        let rules = crate::transforms::pattern::basic_rules(arena);
        let evaled = crate::transforms::eval::eval(arena, expr);
        let free = crate::base::walk::free_symbols(arena, evaled);
        let mut cancel_best = evaled;
        let mut cancel_best_ops = count_ops(arena, evaled);
        let mut cancel_steps: Vec<RawStep> = Vec::new();
        for &sym in &free {
            let cancelled = crate::poly::polybridge::cancel(arena, evaled, sym);
            let cancelled_eval = crate::transforms::eval::eval(arena, cancelled);
            let (cancelled_simp, steps) =
                crate::transforms::pattern::apply_rules(arena, cancelled_eval, &rules);
            let ops = count_ops(arena, cancelled_simp);
            if ops < cancel_best_ops {
                cancel_best = cancelled_simp;
                cancel_best_ops = ops;
                cancel_steps = steps;
            }
        }
        best.consider(arena, cancel_best, "eval+cancel+rules", cancel_steps);
    }

    // Strategy 8: eval → refine (assumption-aware) (only if refinable nodes present)
    if flags.has_abs || flags.has_sign || flags.has_floor_ceil || flags.has_pow {
        let s8_eval = crate::transforms::eval::eval(arena, expr);
        // refine needs &mut AssumptionCache — but smart_simplify only has &mut Arena.
        // We build a temporary cache for this pass.  When called through the public
        // Ex::simplify() path the real cache is used; here we create a fresh one
        // that still reads symbol-level assumptions from the arena.
        let mut temp_assumptions = crate::base::assumptions::AssumptionCache::new();
        let s8 = crate::simplify::refine::refine_full(arena, &mut temp_assumptions, s8_eval);
        best.consider(arena, s8, "eval+refine", Vec::new());
    } else {
        tracing::debug!("smart_simplify: skipping refine (no abs/sign/floor/ceil/pow nodes)");
    }

    // Strategy 9: eval → powsimp → eval (only if has Pow nodes)
    if flags.has_pow {
        let s9_eval = crate::transforms::eval::eval(arena, expr);
        let s9_pow = crate::simplify::powsimp::powsimp(arena, s9_eval);
        let s9 = crate::transforms::eval::eval(arena, s9_pow);
        best.consider(arena, s9, "eval+powsimp", Vec::new());
    } else {
        tracing::debug!("smart_simplify: skipping powsimp (no Pow nodes)");
    }

    // Strategy 10: eval → combsimp → eval (only if has factorial/gamma)
    if flags.has_factorial || flags.has_gamma {
        let s10_eval = crate::transforms::eval::eval(arena, expr);
        let s10_comb = crate::simplify::combsimp::combsimp(arena, s10_eval);
        let s10 = crate::transforms::eval::eval(arena, s10_comb);
        best.consider(arena, s10, "eval+combsimp", Vec::new());
    } else {
        tracing::debug!("smart_simplify: skipping combsimp (no factorial/gamma nodes)");
    }

    // Strategy 11: eval → radsimp (rationalize denominator) (only if has sqrt in denominator)
    if flags.has_sqrt && flags.has_neg_pow {
        let s11_eval = crate::transforms::eval::eval(arena, expr);
        let s11_rad = crate::simplify::radsimp::rationalize_denom(arena, s11_eval);
        let s11 = crate::transforms::eval::eval(arena, s11_rad);
        best.consider(arena, s11, "eval+radsimp", Vec::new());
    } else {
        tracing::debug!("smart_simplify: skipping radsimp (no sqrt+neg_pow nodes)");
    }

    // Strategy 11b: eval → sqrtdenest → eval (only if has sqrt)
    //
    // √(3 + 2√2) → 1 + √2, √(5 − 2√6) → √3 − √2.
    if flags.has_sqrt {
        let s11b_eval = crate::transforms::eval::eval(arena, expr);
        let s11b_den = crate::simplify::radsimp::sqrtdenest(arena, s11b_eval);
        let s11b = crate::transforms::eval::eval(arena, s11b_den);
        best.consider(arena, s11b, "eval+sqrtdenest", Vec::new());
    }

    // Strategy 12: eval → powdenest → powsimp_base → eval (only if has Pow)
    //
    // Full radical simplification chain:
    //   powdenest:    (a·b)^n → a^n · b^n  (e.g., (½·√5)² → ¼·5 = 5/4)
    //   powsimp_base: a^e · b^e → (a·b)^e  (e.g., √2·√3 → √6)
    //
    // Both are applied in sequence because they are complementary:
    // powdenest breaks apart compound bases, powsimp_base recombines
    // separate bases.  The eval passes before and after handle
    // canonicalization of the results (e.g., 5/4 - 5/4 → 0).
    if flags.has_pow {
        let s12_eval = crate::transforms::eval::eval(arena, expr);
        let s12_denest = crate::simplify::powsimp::powdenest(arena, s12_eval);
        let s12_base = crate::simplify::powsimp::powsimp_base(arena, s12_denest);
        let s12 = crate::transforms::eval::eval(arena, s12_base);
        best.consider(arena, s12, "eval+powdenest+powsimp_base+eval", Vec::new());

        // Also try the individual strategies alone (one may help where
        // the other doesn't, and the combined chain may increase op count).
        let s12b_denest = crate::simplify::powsimp::powdenest(arena, s12_eval);
        let s12b = crate::transforms::eval::eval(arena, s12b_denest);
        best.consider(arena, s12b, "eval+powdenest", Vec::new());

        let s12c_base = crate::simplify::powsimp::powsimp_base(arena, s12_eval);
        let s12c = crate::transforms::eval::eval(arena, s12c_base);
        best.consider(arena, s12c, "eval+powsimp_base", Vec::new());
    } else {
        tracing::debug!("smart_simplify: skipping powdenest/powsimp_base (no Pow nodes)");
    }

    tracing::debug!(
        ops_original = original_ops,
        ops_result = best.ops,
        strategy = best.strategy,
        node_count = flags.node_count,
        has_trig = flags.has_trig,
        has_hyp = flags.has_hyp,
        has_exp_ln = flags.has_exp_ln,
        has_neg_pow = flags.has_neg_pow,
        has_add = flags.has_add,
        has_apply = flags.has_apply,
        has_factorial = flags.has_factorial,
        has_gamma = flags.has_gamma,
        has_sqrt = flags.has_sqrt,
        "smart_simplify selected best"
    );

    // Guard: don't return something much more complex than original
    if original_ops > 0 && best.ops as f64 > 1.7 * original_ops as f64 {
        return (expr, "none", Vec::new());
    }

    (best.expr, best.strategy, best.steps)
}

/// Unified simplification engine — iterates [`smart_simplify`] to a fixpoint.
///
/// This is the single engine behind `.simplify()`.  Each iteration runs
/// the full multi-strategy `smart_simplify` (12+ strategies, flag-gated),
/// then checks for convergence.  Safety mechanisms:
///
/// * **Global bloat guard:** the result is never allowed to exceed 2× the
///   op-count of the *original* input (iteration 0), preventing the
///   per-call 1.7× guard from compounding across iterations.
/// * **Cycle detection:** a set of previously-seen `ExprId`s detects
///   oscillation (e.g., expand ↔ factor trading off at equal op-count).
/// * **Fixpoint check:** iteration stops as soon as the expression
///   doesn't change.
///
/// When `opts.trace` is set, every iteration that changes the expression
/// contributes a `strategy:<name>` step followed by the rule-level steps
/// that fired inside the winning strategy.
pub(crate) fn unified_simplify(
    arena: &mut Arena,
    expr: ExprId,
    opts: &SimplifyOpts,
) -> SimplifyResult {
    let original_ops = count_ops(arena, expr);
    let max_ops = 2 * original_ops.max(1); // global bloat ceiling
    let mut current = expr;
    let mut seen = FxHashSet::default();
    seen.insert(current);
    let mut trace: Vec<RawStep> = Vec::new();

    for i in 0..opts.max_iterations {
        let (next, strategy, rule_steps) = smart_simplify_traced(arena, current);

        // Global bloat guard: never exceed 2× the original expression.
        let next_ops = count_ops(arena, next);
        if next_ops > max_ops {
            tracing::debug!(
                iteration = i,
                next_ops,
                max_ops,
                "unified_simplify: global bloat guard triggered, stopping"
            );
            return SimplifyResult {
                expr: current,
                steps: trace,
            };
        }

        // Cycle detection: stop if we've seen this expression before.
        if !seen.insert(next) {
            tracing::debug!(iteration = i, "unified_simplify: cycle detected, stopping");
            // Return the better of current vs next (in case the cycle
            // revisits the optimal form).
            let best = if next_ops <= count_ops(arena, current) {
                if opts.trace && next != current {
                    trace.push(RawStep {
                        rule_name: format!("strategy:{strategy}"),
                        before: current,
                        after: next,
                    });
                    trace.extend(rule_steps);
                }
                next
            } else {
                current
            };
            return SimplifyResult {
                expr: best,
                steps: trace,
            };
        }

        // Fixpoint: expression didn't change.
        if next == current {
            tracing::debug!(iteration = i, "unified_simplify: fixpoint reached");
            return SimplifyResult {
                expr: current,
                steps: trace,
            };
        }

        if opts.trace {
            trace.push(RawStep {
                rule_name: format!("strategy:{strategy}"),
                before: current,
                after: next,
            });
            trace.extend(rule_steps);
        }
        current = next;
    }

    tracing::debug!("unified_simplify: max iterations reached");
    SimplifyResult {
        expr: current,
        steps: trace,
    }
}
/// Legacy wrapper — iterates `smart_simplify` to fixpoint with default options.
#[allow(dead_code)]
pub(crate) fn full_simplify(arena: &mut Arena, expr: ExprId) -> ExprId {
    unified_simplify(arena, expr, &SimplifyOpts::default()).expr
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::arena::Arena;

    fn sym(a: &mut Arena, name: &str) -> ExprId {
        a.symbol(name)
    }
    fn display(a: &Arena, id: ExprId) -> String {
        a.display(id).to_string()
    }

    #[test]
    fn count_ops_atom() {
        let a = Arena::new();
        assert_eq!(count_ops(&a, a.zero), 0);
        assert_eq!(count_ops(&a, a.one), 0);
        assert_eq!(count_ops(&a, a.pi), 0);
    }

    #[test]
    fn count_ops_simple() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let expr = a.add(&[x, two]);
        assert_eq!(count_ops(&a, expr), 1); // one Add
    }

    #[test]
    fn count_ops_nested() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let sin_x = a.sin(x);
        let two = a.int(2);
        let expr = a.pow(sin_x, two);
        assert_eq!(count_ops(&a, expr), 2); // Sin + Pow
    }

    #[test]
    fn smart_simplify_trig_identity() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let sin_x = a.sin(x);
        let cos_x = a.cos(x);
        let two = a.int(2);
        let sin2 = a.pow(sin_x, two);
        let cos2 = a.pow(cos_x, two);
        let expr = a.add(&[sin2, cos2]);
        let result = smart_simplify(&mut a, expr);
        assert_eq!(display(&a, result), "1");
    }

    #[test]
    fn smart_simplify_exp_ln() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let ln_x = a.ln(x);
        let expr = a.exp(ln_x);
        let result = smart_simplify(&mut a, expr);
        assert_eq!(display(&a, result), "x");
    }

    #[test]
    fn smart_simplify_doesnt_bloat() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x is already simple — shouldn't get more complex
        let result = smart_simplify(&mut a, x);
        assert_eq!(result, x);
    }

    #[test]
    fn smart_simplify_eval_catches_sin_zero() {
        let mut a = Arena::new();
        let expr = a.sin(a.zero);
        let result = smart_simplify(&mut a, expr);
        assert_eq!(display(&a, result), "0");
    }

    #[test]
    fn count_ops_complex_expression() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let xy = a.mul(&[x, y]);
        let expr = a.add(&[x2, xy, y]);
        // Pow + Mul + Add = 3
        assert_eq!(count_ops(&a, expr), 3);
    }

    #[test]
    fn smart_simplify_cancel_rational() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let two = a.int(2);
        let one = a.one;
        // (x²-1)/(x-1) should simplify to x+1
        let x2 = a.pow(x, two);
        let numer = a.sub(x2, one);
        let denom = a.sub(x, one);
        let expr = a.div(numer, denom);
        let result = smart_simplify(&mut a, expr);
        let s = a.display(result).to_string();
        assert!(
            s.contains("1") && s.contains("x") && !s.contains("/"),
            "(x²-1)/(x-1) should simplify to x+1, got: {s}"
        );
    }

    /// Regression: Strategy 4 calls `factor_terms_pair` but discards the GCD,
    /// entering only the GCD-stripped inner expression into the candidate pool.
    /// Because `count_ops` is lower for the stripped version, it wins — producing
    /// a result that is NOT mathematically equivalent to the input.
    ///
    /// For `6*x + 12`:
    ///   factor_terms_pair → (gcd=6, inner=x+2)
    ///   Strategy 4 keeps only `x+2`, which has 1 op vs 3 ops for `6*x+12`.
    ///   So smart_simplify returns `x+2` instead of `6*x+12` (or `6*(x+2)`).
    #[test]
    fn smart_simplify_strategy4_discards_gcd() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let six = a.int(6);
        let twelve = a.int(12);
        let six_x = a.mul(&[six, x]);
        let expr = a.add(&[six_x, twelve]); // 6*x + 12

        // Confirm what factor_terms_pair returns
        let eval_expr = crate::transforms::eval::eval(&mut a, expr);
        let (gcd, inner) = crate::simplify::factor_terms::factor_terms_pair(&mut a, eval_expr);
        let inner_s = display(&a, inner);
        assert_eq!(
            gcd,
            num_rational::Ratio::from_integer(num_bigint::BigInt::from(6)),
            "GCD should be 6"
        );
        assert!(
            inner_s.contains("x") && inner_s.contains("2"),
            "inner should be x+2, got: {inner_s}"
        );

        // Now run smart_simplify — the bug causes it to return x+2
        let result = smart_simplify(&mut a, expr);
        let result_s = display(&a, result);

        // The simplified form must still contain the factor 6 (or be
        // equivalent, e.g. "6*(x + 2)" or "6*x + 12").  If it equals
        // just "x + 2" then Strategy 4's GCD was silently dropped.
        let result_ops = count_ops(&a, result);
        let inner_ops = count_ops(&a, inner);
        assert!(
            result_s.contains("6") || result_s.contains("12"),
            "BUG: smart_simplify returned '{}' for 6*x+12 — \
             the GCD factor 6 was discarded by Strategy 4 \
             (result has {} ops vs inner's {} ops)",
            result_s,
            result_ops,
            inner_ops
        );
    }

    #[test]
    fn smart_simplify_cancels_x2_minus_4_over_x_minus_2() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let two = a.int(2);
        let four = a.int(4);
        let x_sq = a.pow(x, two);
        let numer = a.sub(x_sq, four); // x² - 4
        let denom = a.sub(x, two); // x - 2
        let expr = a.div(numer, denom); // (x²-4)/(x-2)
        let result = smart_simplify(&mut a, expr);
        let expected = a.add(&[x, two]); // x + 2
        assert_eq!(
            result,
            expected,
            "(x²-4)/(x-2) should smart_simplify to x+2, got: {}",
            display(&a, result)
        );
    }

    #[test]
    fn full_simplify_cancels_x2_minus_4_over_x_minus_2() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let two = a.int(2);
        let four = a.int(4);
        let x_sq = a.pow(x, two);
        let numer = a.sub(x_sq, four); // x² - 4
        let denom = a.sub(x, two); // x - 2
        let expr = a.div(numer, denom); // (x²-4)/(x-2)
        let result = full_simplify(&mut a, expr);
        let expected = a.add(&[x, two]); // x + 2
        assert_eq!(
            result,
            expected,
            "(x²-4)/(x-2) should full_simplify to x+2, got: {}",
            display(&a, result)
        );
    }

    #[test]
    fn full_simplify_x2_minus_1_over_x_minus_1() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let numer = a.sub(x2, a.one); // x² - 1
        let denom = a.sub(x, a.one); // x - 1
        let expr = a.div(numer, denom);
        let result = full_simplify(&mut a, expr);
        let s = a.display(result).to_string();
        assert!(
            s.contains("1") && s.contains("x") && !s.contains("/"),
            "(x²-1)/(x-1) should full_simplify to x+1, got: {s}"
        );
    }

    #[test]
    fn full_simplify_already_simple() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let two = a.int(2);
        let expr = a.add(&[x, two]); // x + 2 — nothing to cancel
        let result = full_simplify(&mut a, expr);
        assert_eq!(result, expr, "x+2 should stay x+2 through full_simplify");
    }

    #[test]
    fn smart_simplify_preserves_gcd_factor() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let six = a.int(6);
        let twelve = a.int(12);
        let six_x = a.mul(&[six, x]);
        let expr = a.add(&[six_x, twelve]); // 6x + 12
        let result = smart_simplify(&mut a, expr);
        // The result must be mathematically equivalent to 6x + 12
        // It must NOT be x + 2 (which drops the factor of 6)
        let two = a.int(2);
        let x_plus_2 = a.add(&[x, two]);
        assert_ne!(result, x_plus_2, "smart_simplify must not drop GCD factor");
    }

    // ── Tests for compute_flags ────────────────────────────────────────

    #[test]
    fn compute_flags_atom() {
        let a = Arena::new();
        let flags = compute_flags(&a, a.zero);
        assert!(flags.is_atom);
        assert!(!flags.has_trig);
        assert!(!flags.has_hyp);
        assert!(!flags.has_exp_ln);
        assert!(!flags.has_neg_pow);
        assert!(!flags.has_add);
        assert!(!flags.has_apply);
        assert_eq!(flags.node_count, 1);
    }

    #[test]
    fn compute_flags_trig() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let sin_x = a.sin(x);
        let flags = compute_flags(&a, sin_x);
        assert!(flags.has_trig);
        assert!(!flags.has_hyp);
        assert!(!flags.has_exp_ln);
        assert!(!flags.is_atom);
    }

    #[test]
    fn compute_flags_exp_ln() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let ln_x = a.ln(x);
        let exp_ln_x = a.exp(ln_x);
        let flags = compute_flags(&a, exp_ln_x);
        assert!(flags.has_exp_ln);
        assert!(!flags.has_trig);
    }

    #[test]
    fn compute_flags_neg_pow() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let neg_one = a.int(-1);
        let recip = a.pow(x, neg_one);
        let flags = compute_flags(&a, recip);
        assert!(flags.has_neg_pow);
        assert!(!flags.has_trig);
    }

    #[test]
    fn compute_flags_add() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let sum = a.add(&[x, y]);
        let flags = compute_flags(&a, sum);
        assert!(flags.has_add);
        assert!(!flags.has_trig);
    }

    #[test]
    fn compute_flags_hyp() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let sinh_x = a.sinh(x);
        let flags = compute_flags(&a, sinh_x);
        assert!(flags.has_hyp);
        assert!(!flags.has_trig);
    }
}

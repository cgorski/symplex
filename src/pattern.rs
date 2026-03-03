//! Pattern matching and rewrite-rule engine.
//!
//! This module provides structural pattern matching against canonical
//! expression trees, plus a rule-based rewriting system that powers
//! `.simplify()` and `.simplify_trace()`.
//!
//! # Design
//!
//! Patterns are descriptions of expression shapes with **wild** (pattern
//! variable) slots that can bind to any sub-expression.  Matching is
//! purely structural — it operates on the canonical form, so commutativity
//! of `Add`/`Mul` is handled implicitly by the canonical sort order.
//!
//! A [`Rule`] pairs a pattern (the LHS) with a template (the RHS) and
//! an optional condition.  [`apply_rules`] walks an expression bottom-up,
//! trying each rule at every node, and returns the rewritten expression
//! plus a trace of which rules fired.
//!
//! # Example
//!
//! ```text
//! // sin²(w) + cos²(w) → 1
//! let rule = Rule::new(
//!     "pythagorean",
//!     build_pattern_sin2_cos2(&mut arena),
//!     arena.one,
//! );
//! ```

use rustc_hash::FxHashMap;

use crate::arena::Arena;
use crate::node::{ExprId, ExprNode};
use crate::walk;

// ═══════════════════════════════════════════════════════════════════════════
// WildId — pattern variable identifier
// ═══════════════════════════════════════════════════════════════════════════

/// Identifies a pattern variable ("wild") inside a [`Pattern`].
///
/// Wilds are created via [`Arena::wild`] and are represented in the
/// expression tree as special `Symbol` nodes whose names start with `_w`.
/// During matching, a wild binds to whatever sub-expression it is
/// matched against.  If the same `WildId` appears multiple times in a
/// pattern, all occurrences must bind to the **same** `ExprId`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WildId(pub(crate) u32);

// ═══════════════════════════════════════════════════════════════════════════
// Substitution — wild bindings
// ═══════════════════════════════════════════════════════════════════════════

/// A mapping from [`WildId`]s to the [`ExprId`]s they matched.
pub type Substitution = FxHashMap<WildId, ExprId>;

// ═══════════════════════════════════════════════════════════════════════════
// Pattern — what to match
// ═══════════════════════════════════════════════════════════════════════════

/// A pattern describes the shape of an expression to match.
///
/// Patterns are built from the same `ExprId` system as regular
/// expressions, but certain "wild" symbols act as pattern variables
/// that can bind to any sub-expression during matching.
///
/// Matching is structural: patterns match the canonical form of an
/// expression.  Since `Add` and `Mul` are canonically sorted, patterns
/// that respect the canonical order will match correctly without
/// explicit commutative search.
#[derive(Clone, Debug)]
pub struct Pattern {
    /// The root of the pattern expression (built in the arena).
    pub root: ExprId,
    /// Which symbols in the pattern are wilds.
    pub wilds: FxHashMap<ExprId, WildId>,
}

// ═══════════════════════════════════════════════════════════════════════════
// Arena extensions for pattern construction
// ═══════════════════════════════════════════════════════════════════════════

/// Counter for generating unique wild names.
static NEXT_WILD_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

impl Arena {
    /// Create a new wild (pattern variable) symbol.
    ///
    /// Returns the `ExprId` of the wild symbol and its `WildId`.
    /// The wild is represented as a regular symbol with a unique
    /// internal name (e.g., `_w0`, `_w1`, ...).
    pub fn wild(&mut self) -> (ExprId, WildId) {
        let id = NEXT_WILD_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let wid = WildId(id);
        let name = format!("_w{id}");
        let expr_id = self.symbol(&name);
        (expr_id, wid)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Matching
// ═══════════════════════════════════════════════════════════════════════════

/// Try to match `pattern` against `expr`, returning bindings for wilds
/// on success.
///
/// Matching is structural and top-down.  Wilds bind to whatever they
/// encounter.  If the same wild appears multiple times, all occurrences
/// must bind to the same `ExprId` (consistency check).
///
/// Returns `None` if the pattern does not match.
pub(crate) fn match_pattern(
    arena: &Arena,
    pattern: &Pattern,
    expr: ExprId,
) -> Option<Substitution> {
    let mut bindings = Substitution::default();
    if match_recursive(arena, pattern, pattern.root, expr, &mut bindings) {
        Some(bindings)
    } else {
        None
    }
}

/// Recursive structural matching.  Uses the Rust call stack, but
/// pattern depth is always small (patterns are hand-written, typically
/// < 10 nodes deep), so stack overflow is not a concern here.
fn match_recursive(
    arena: &Arena,
    pattern: &Pattern,
    pat_id: ExprId,
    expr_id: ExprId,
    bindings: &mut Substitution,
) -> bool {
    // Is this pattern node a wild?
    if let Some(&wid) = pattern.wilds.get(&pat_id) {
        // Check consistency: if this wild was already bound, it must
        // bind to the same expression.
        if let Some(&existing) = bindings.get(&wid) {
            return existing == expr_id;
        }
        bindings.insert(wid, expr_id);
        return true;
    }

    // Not a wild — structural match.
    let pat_node = arena.node(pat_id);
    let expr_node = arena.node(expr_id);

    match (pat_node.clone(), expr_node.clone()) {
        // Atoms: must be identical ExprIds.
        (ExprNode::Num(a), ExprNode::Num(b)) => a == b,
        (ExprNode::Symbol(a), ExprNode::Symbol(b)) => a == b,
        (ExprNode::Pi, ExprNode::Pi) => true,
        (ExprNode::E, ExprNode::E) => true,
        (ExprNode::ImaginaryUnit, ExprNode::ImaginaryUnit) => true,
        (ExprNode::Infinity, ExprNode::Infinity) => true,
        (ExprNode::NegInfinity, ExprNode::NegInfinity) => true,
        (ExprNode::ComplexInfinity, ExprNode::ComplexInfinity) => true,
        (ExprNode::NaN, ExprNode::NaN) => true,

        // N-ary: Add, Mul — children must match positionally.
        // Since both pattern and expression are canonical (sorted),
        // positional matching IS commutative matching.
        (ExprNode::Add(pat_children), ExprNode::Add(expr_children)) => {
            if pat_children.len() != expr_children.len() {
                return false;
            }
            for (pc, ec) in pat_children.iter().zip(expr_children.iter()) {
                if !match_recursive(arena, pattern, *pc, *ec, bindings) {
                    return false;
                }
            }
            true
        }

        (ExprNode::Mul(pat_children), ExprNode::Mul(expr_children)) => {
            if pat_children.len() != expr_children.len() {
                return false;
            }
            for (pc, ec) in pat_children.iter().zip(expr_children.iter()) {
                if !match_recursive(arena, pattern, *pc, *ec, bindings) {
                    return false;
                }
            }
            true
        }

        // Binary.
        (ExprNode::Pow(pb, pe), ExprNode::Pow(eb, ee)) => {
            match_recursive(arena, pattern, pb, eb, bindings)
                && match_recursive(arena, pattern, pe, ee, bindings)
        }
        (ExprNode::Derivative(pb, pv), ExprNode::Derivative(eb, ev)) => {
            match_recursive(arena, pattern, pb, eb, bindings)
                && match_recursive(arena, pattern, pv, ev, bindings)
        }
        (ExprNode::Integral(pb, pv), ExprNode::Integral(eb, ev)) => {
            match_recursive(arena, pattern, pb, eb, bindings)
                && match_recursive(arena, pattern, pv, ev, bindings)
        }

        // Unary.
        (ExprNode::Neg(pi), ExprNode::Neg(ei)) => match_recursive(arena, pattern, pi, ei, bindings),
        (ExprNode::Sin(pi), ExprNode::Sin(ei)) => match_recursive(arena, pattern, pi, ei, bindings),
        (ExprNode::Cos(pi), ExprNode::Cos(ei)) => match_recursive(arena, pattern, pi, ei, bindings),
        (ExprNode::Tan(pi), ExprNode::Tan(ei)) => match_recursive(arena, pattern, pi, ei, bindings),
        (ExprNode::Exp(pi), ExprNode::Exp(ei)) => match_recursive(arena, pattern, pi, ei, bindings),
        (ExprNode::Ln(pi), ExprNode::Ln(ei)) => match_recursive(arena, pattern, pi, ei, bindings),
        (ExprNode::Sqrt(pi), ExprNode::Sqrt(ei)) => {
            match_recursive(arena, pattern, pi, ei, bindings)
        }
        (ExprNode::Abs(pi), ExprNode::Abs(ei)) => match_recursive(arena, pattern, pi, ei, bindings),
        (ExprNode::Asin(pi), ExprNode::Asin(ei)) => {
            match_recursive(arena, pattern, pi, ei, bindings)
        }
        (ExprNode::Acos(pi), ExprNode::Acos(ei)) => {
            match_recursive(arena, pattern, pi, ei, bindings)
        }
        (ExprNode::Atan(pi), ExprNode::Atan(ei)) => {
            match_recursive(arena, pattern, pi, ei, bindings)
        }
        (ExprNode::Sinh(pi), ExprNode::Sinh(ei)) => {
            match_recursive(arena, pattern, pi, ei, bindings)
        }
        (ExprNode::Cosh(pi), ExprNode::Cosh(ei)) => {
            match_recursive(arena, pattern, pi, ei, bindings)
        }
        (ExprNode::Tanh(pi), ExprNode::Tanh(ei)) => {
            match_recursive(arena, pattern, pi, ei, bindings)
        }

        // Apply: function name must match, then args positionally.
        (ExprNode::Apply(pf, ref pa), ExprNode::Apply(ef, ref ea)) => {
            if pf != ef || pa.len() != ea.len() {
                return false;
            }
            for (pc, ec) in pa.iter().zip(ea.iter()) {
                if !match_recursive(arena, pattern, *pc, *ec, bindings) {
                    return false;
                }
            }
            true
        }

        // Any other combination: no match.
        _ => false,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Template instantiation
// ═══════════════════════════════════════════════════════════════════════════

/// Instantiate a template expression by replacing wilds with their
/// bound values from a [`Substitution`].
///
/// This is used after a successful match to build the replacement
/// expression from the RHS template of a rule.
pub(crate) fn instantiate(
    arena: &mut Arena,
    template: ExprId,
    wilds: &FxHashMap<ExprId, WildId>,
    bindings: &Substitution,
) -> ExprId {
    // Build a replacement map: wild_expr_id → bound_value.
    let replacements: FxHashMap<ExprId, ExprId> = wilds
        .iter()
        .filter_map(|(&expr_id, &wid)| bindings.get(&wid).map(|&val| (expr_id, val)))
        .collect();

    if replacements.is_empty() {
        return template;
    }

    walk::walk_and_rebuild(arena, template, &|_arena, id| {
        replacements.get(&id).copied()
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Rule — named rewrite rule
// ═══════════════════════════════════════════════════════════════════════════

/// A named rewrite rule: pattern (LHS) → template (RHS).
///
/// Optionally carries a condition function that is checked after the
/// pattern matches but before the replacement is applied.
#[derive(Clone)]
pub struct Rule {
    /// Human-readable name for tracing.
    pub name: &'static str,
    /// The pattern to match (LHS).
    pub pattern: Pattern,
    /// The template to instantiate (RHS).
    pub template: ExprId,
    /// Optional condition.  Called with the arena and the bindings
    /// after a successful structural match.  If it returns `false`,
    /// the match is rejected.
    pub condition: Option<fn(&Arena, &Substitution) -> bool>,
}

impl Rule {
    /// Create a rule without a condition.
    pub fn new(name: &'static str, pattern: Pattern, template: ExprId) -> Self {
        Rule {
            name,
            pattern,
            template,
            condition: None,
        }
    }

    /// Try to apply this rule to a single expression node.
    ///
    /// Returns `Some(replacement)` if the rule matched and the condition
    /// (if any) was satisfied; `None` otherwise.
    pub(crate) fn try_apply(&self, arena: &mut Arena, expr: ExprId) -> Option<ExprId> {
        let bindings = match_pattern(arena, &self.pattern, expr)?;

        // Check condition.
        if let Some(cond) = self.condition
            && !cond(arena, &bindings)
        {
            return None;
        }

        // Instantiate the template.
        let result = instantiate(arena, self.template, &self.pattern.wilds, &bindings);
        Some(result)
    }
}

impl std::fmt::Debug for Rule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Rule").field("name", &self.name).finish()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Step — trace entry
// ═══════════════════════════════════════════════════════════════════════════

/// A record of one simplification step.
///
/// Produced by [`apply_rules`] to enable "show your work" functionality.
#[derive(Clone, Debug)]
pub struct Step {
    /// The name of the rule that fired.
    pub rule_name: &'static str,
    /// The sub-expression before the rule was applied.
    pub before: ExprId,
    /// The sub-expression after the rule was applied.
    pub after: ExprId,
}

// ═══════════════════════════════════════════════════════════════════════════
// apply_rules — bottom-up rewriting with tracing
// ═══════════════════════════════════════════════════════════════════════════

/// Apply a list of rules to an expression in a single bottom-up pass.
///
/// At each node (processed leaves-first), the rules are tried in order.
/// The first rule that matches is applied, and the rewritten node
/// replaces the original.  The pass is NOT iterated — for fixpoint
/// rewriting, call this in a loop until no more steps are produced.
///
/// Returns the rewritten expression and a trace of all steps.
pub(crate) fn apply_rules(arena: &mut Arena, expr: ExprId, rules: &[Rule]) -> (ExprId, Vec<Step>) {
    let mut steps: Vec<Step> = Vec::new();
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();

    let post_order = walk::post_order_ids(arena, expr);

    for &id in &post_order {
        // First, rebuild with any already-rewritten children.
        let rebuilt = if arena.node(id).is_atom() {
            id
        } else {
            rebuild_with_cache(arena, id, &cache)
        };

        // Try each rule.
        let mut rewritten = rebuilt;
        for rule in rules {
            if let Some(replacement) = rule.try_apply(arena, rewritten) {
                steps.push(Step {
                    rule_name: rule.name,
                    before: rebuilt,
                    after: replacement,
                });
                rewritten = replacement;
                break; // first match wins
            }
        }

        // Sub-expression matching: if the node is an Add and no rule matched
        // the whole node, try matching rules against subsets of the Add's children.
        if rewritten == rebuilt {
            if let ExprNode::Add(ref children) = arena.node(rebuilt).clone() {
                if children.len() >= 2 {
                    'sub_match: for rule in rules {
                        // Only attempt if the rule's pattern root is an Add.
                        if let ExprNode::Add(ref pat_children) =
                            arena.node(rule.pattern.root).clone()
                        {
                            let k = pat_children.len();
                            if k == 2 && children.len() >= 2 {
                                // Try all pairs of children.
                                for i in 0..children.len() {
                                    for j in (i + 1)..children.len() {
                                        let pair = arena.add(&[children[i], children[j]]);
                                        if let Some(replacement) = rule.try_apply(arena, pair) {
                                            // Build remaining terms.
                                            let mut remaining: smallvec::SmallVec<[ExprId; 6]> =
                                                smallvec::SmallVec::new();
                                            for (idx, &child) in children.iter().enumerate() {
                                                if idx != i && idx != j {
                                                    remaining.push(child);
                                                }
                                            }
                                            remaining.push(replacement);
                                            rewritten = arena.add(&remaining);
                                            steps.push(Step {
                                                rule_name: rule.name,
                                                before: rebuilt,
                                                after: rewritten,
                                            });
                                            break 'sub_match;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        cache.insert(id, rewritten);
    }

    let result = cache.get(&expr).copied().unwrap_or(expr);
    (result, steps)
}

/// Rebuild a node with children looked up from the cache.
///
/// Delegates to [`walk::rebuild_with_cache`] to avoid code duplication.
fn rebuild_with_cache(arena: &mut Arena, id: ExprId, cache: &FxHashMap<ExprId, ExprId>) -> ExprId {
    crate::walk::rebuild_with_cache(arena, id, cache)
}

// ═══════════════════════════════════════════════════════════════════════════
// Built-in rules
// ═══════════════════════════════════════════════════════════════════════════

/// Build the Pythagorean identity rule: `sin²(w) + cos²(w) → 1`.
///
/// Returns the rule.  The pattern and template are interned in the
/// given arena.
pub(crate) fn rule_pythagorean(arena: &mut Arena) -> Rule {
    let (w_expr, w_id) = arena.wild();
    let two = arena.int(2);

    // Pattern: sin(w)^2 + cos(w)^2
    // Build: Pow(Sin(w), 2) and Pow(Cos(w), 2)
    let sin_w = arena.sin(w_expr);
    let cos_w = arena.cos(w_expr);
    let sin_sq = arena.pow(sin_w, two);
    let cos_sq = arena.pow(cos_w, two);
    let pattern_expr = arena.add(&[sin_sq, cos_sq]);

    let mut wilds = FxHashMap::default();
    wilds.insert(w_expr, w_id);

    let pattern = Pattern {
        root: pattern_expr,
        wilds,
    };

    Rule::new("pythagorean", pattern, arena.one)
}

/// Build the inverse function rule: `exp(ln(w)) → w`.
fn rule_exp_ln(arena: &mut Arena) -> Rule {
    let (w_expr, w_id) = arena.wild();
    let ln_w = arena.ln(w_expr);
    let exp_ln_w = arena.exp_fn(ln_w);

    let mut wilds = FxHashMap::default();
    wilds.insert(w_expr, w_id);
    let pattern = Pattern {
        root: exp_ln_w,
        wilds,
    };
    Rule::new("exp_ln", pattern, w_expr)
}

/// Build the inverse function rule: `ln(exp(w)) → w`.
fn rule_ln_exp(arena: &mut Arena) -> Rule {
    let (w_expr, w_id) = arena.wild();
    let exp_w = arena.exp_fn(w_expr);
    let ln_exp_w = arena.ln(exp_w);

    let mut wilds = FxHashMap::default();
    wilds.insert(w_expr, w_id);
    let pattern = Pattern {
        root: ln_exp_w,
        wilds,
    };
    Rule::new("ln_exp", pattern, w_expr)
}

/// Build the idempotent abs rule: `abs(abs(w)) → abs(w)`.
fn rule_abs_abs(arena: &mut Arena) -> Rule {
    let (w_expr, w_id) = arena.wild();
    let abs_w = arena.abs(w_expr);
    let abs_abs_w = arena.abs(abs_w);

    let mut wilds = FxHashMap::default();
    wilds.insert(w_expr, w_id);
    let pattern = Pattern {
        root: abs_abs_w,
        wilds,
    };
    Rule::new("abs_abs", pattern, abs_w)
}

/// Build the rule: `sqrt(w^2) → abs(w)`.
fn rule_sqrt_sq(arena: &mut Arena) -> Rule {
    let (w_expr, w_id) = arena.wild();
    let two = arena.int(2);
    let w_sq = arena.pow(w_expr, two);
    let sqrt_w_sq = arena.sqrt(w_sq);
    let abs_w = arena.abs(w_expr);

    let mut wilds = FxHashMap::default();
    wilds.insert(w_expr, w_id);
    let pattern = Pattern {
        root: sqrt_w_sq,
        wilds,
    };
    Rule::new("sqrt_sq", pattern, abs_w)
}

/// Build the inverse trig rule: `asin(sin(w)) → w`.
fn rule_asin_sin(arena: &mut Arena) -> Rule {
    let (w_expr, w_id) = arena.wild();
    let sin_w = arena.sin(w_expr);
    let asin_sin_w = arena.asin(sin_w);
    let mut wilds = FxHashMap::default();
    wilds.insert(w_expr, w_id);
    let pattern = Pattern {
        root: asin_sin_w,
        wilds,
    };
    Rule::new("asin_sin", pattern, w_expr)
}

/// Build the inverse trig rule: `acos(cos(w)) → w`.
fn rule_acos_cos(arena: &mut Arena) -> Rule {
    let (w_expr, w_id) = arena.wild();
    let cos_w = arena.cos(w_expr);
    let acos_cos_w = arena.acos(cos_w);
    let mut wilds = FxHashMap::default();
    wilds.insert(w_expr, w_id);
    let pattern = Pattern {
        root: acos_cos_w,
        wilds,
    };
    Rule::new("acos_cos", pattern, w_expr)
}

/// Build the inverse trig rule: `atan(tan(w)) → w`.
fn rule_atan_tan(arena: &mut Arena) -> Rule {
    let (w_expr, w_id) = arena.wild();
    let tan_w = arena.tan(w_expr);
    let atan_tan_w = arena.atan(tan_w);
    let mut wilds = FxHashMap::default();
    wilds.insert(w_expr, w_id);
    let pattern = Pattern {
        root: atan_tan_w,
        wilds,
    };
    Rule::new("atan_tan", pattern, w_expr)
}

/// Build the hyperbolic Pythagorean identity: `cosh(w)^2 - sinh(w)^2 → 1`.
fn rule_cosh_sinh_identity(arena: &mut Arena) -> Rule {
    let (w_expr, w_id) = arena.wild();
    let two = arena.int(2);
    let sinh_w = arena.sinh(w_expr);
    let cosh_w = arena.cosh(w_expr);
    let sinh_sq = arena.pow(sinh_w, two);
    let cosh_sq = arena.pow(cosh_w, two);
    let neg_sinh_sq = arena.neg(sinh_sq);
    let pattern_expr = arena.add(&[cosh_sq, neg_sinh_sq]);

    let mut wilds = FxHashMap::default();
    wilds.insert(w_expr, w_id);
    let pattern = Pattern {
        root: pattern_expr,
        wilds,
    };
    Rule::new("cosh_sinh_identity", pattern, arena.one)
}

/// Build a basic set of simplification rules.
pub(crate) fn basic_rules(arena: &mut Arena) -> Vec<Rule> {
    vec![
        rule_pythagorean(arena),
        rule_exp_ln(arena),
        rule_ln_exp(arena),
        rule_abs_abs(arena),
        rule_sqrt_sq(arena),
        rule_asin_sin(arena),
        rule_acos_cos(arena),
        rule_atan_tan(arena),
        rule_cosh_sinh_identity(arena),
    ]
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::Arena;

    fn sym(a: &mut Arena, name: &str) -> ExprId {
        a.symbol(name)
    }

    fn display(a: &Arena, id: ExprId) -> String {
        a.display(id).to_string()
    }

    // ── Wild matching ───────────────────────────────────────────────

    #[test]
    fn wild_matches_anything() {
        let mut a = Arena::new();
        let (w, wid) = a.wild();
        let x = sym(&mut a, "x");

        let pattern = Pattern {
            root: w,
            wilds: {
                let mut m = FxHashMap::default();
                m.insert(w, wid);
                m
            },
        };

        let result = match_pattern(&a, &pattern, x);
        assert!(result.is_some());
        let bindings = result.unwrap();
        assert_eq!(bindings[&wid], x);
    }

    #[test]
    fn wild_consistency_check() {
        let mut a = Arena::new();
        let (w, wid) = a.wild();
        let x = sym(&mut a, "x");

        // Pattern: w + w (same wild twice)
        let pat_expr = a.add(&[w, w]); // canonicalizes to 2*w
        let mut wilds = FxHashMap::default();
        wilds.insert(w, wid);
        let pattern = Pattern {
            root: pat_expr,
            wilds,
        };

        // Expression: x + x = 2*x
        let expr = a.add(&[x, x]); // canonicalizes to 2*x

        let result = match_pattern(&a, &pattern, expr);
        assert!(result.is_some(), "2*w should match 2*x");
        assert_eq!(result.unwrap()[&wid], x);
    }

    #[test]
    fn wild_consistency_fails_on_mismatch() {
        let mut a = Arena::new();
        let (w, wid) = a.wild();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");

        // Pattern: w + w = 2*w
        let pat_expr = a.add(&[w, w]);
        let mut wilds = FxHashMap::default();
        wilds.insert(w, wid);
        let pattern = Pattern {
            root: pat_expr,
            wilds,
        };

        // Expression: x + y (different symbols — 2*w can't match x + y)
        let expr = a.add(&[x, y]);

        let result = match_pattern(&a, &pattern, expr);
        assert!(result.is_none(), "2*w should not match x + y");
    }

    // ── Exact matching ──────────────────────────────────────────────

    #[test]
    fn exact_match_symbol() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");

        let pattern = Pattern {
            root: x,
            wilds: FxHashMap::default(),
        };

        assert!(match_pattern(&a, &pattern, x).is_some());
        let y = sym(&mut a, "y");
        assert!(match_pattern(&a, &pattern, y).is_none());
    }

    #[test]
    fn exact_match_integer() {
        let mut a = Arena::new();
        let two = a.int(2);
        let three = a.int(3);

        let pattern = Pattern {
            root: two,
            wilds: FxHashMap::default(),
        };

        assert!(match_pattern(&a, &pattern, two).is_some());
        assert!(match_pattern(&a, &pattern, three).is_none());
    }

    // ── Structural matching ─────────────────────────────────────────

    #[test]
    fn match_sin_of_wild() {
        let mut a = Arena::new();
        let (w, wid) = a.wild();
        let x = sym(&mut a, "x");

        // Pattern: sin(w)
        let pat = a.sin(w);
        let mut wilds = FxHashMap::default();
        wilds.insert(w, wid);
        let pattern = Pattern { root: pat, wilds };

        // Expression: sin(x)
        let expr = a.sin(x);
        let result = match_pattern(&a, &pattern, expr);
        assert!(result.is_some());
        assert_eq!(result.unwrap()[&wid], x);
    }

    #[test]
    fn match_pow_of_wild() {
        let mut a = Arena::new();
        let (w, wid) = a.wild();
        let x = sym(&mut a, "x");
        let two = a.int(2);

        // Pattern: w^2
        let pat = a.pow(w, two);
        let mut wilds = FxHashMap::default();
        wilds.insert(w, wid);
        let pattern = Pattern { root: pat, wilds };

        // Expression: x^2
        let expr = a.pow(x, two);
        let result = match_pattern(&a, &pattern, expr);
        assert!(result.is_some());
        assert_eq!(result.unwrap()[&wid], x);

        // Non-match: x^3
        let three = a.int(3);
        let expr3 = a.pow(x, three);
        assert!(match_pattern(&a, &pattern, expr3).is_none());
    }

    #[test]
    fn match_fails_different_structure() {
        let mut a = Arena::new();
        let (w, wid) = a.wild();
        let x = sym(&mut a, "x");

        // Pattern: sin(w)
        let pat = a.sin(w);
        let mut wilds = FxHashMap::default();
        wilds.insert(w, wid);
        let pattern = Pattern { root: pat, wilds };

        // Expression: cos(x) — wrong function
        let expr = a.cos(x);
        assert!(match_pattern(&a, &pattern, expr).is_none());
    }

    // ── Instantiation ───────────────────────────────────────────────

    #[test]
    fn instantiate_replaces_wilds() {
        let mut a = Arena::new();
        let (w, wid) = a.wild();
        let x = sym(&mut a, "x");

        // Template: w^2
        let two = a.int(2);
        let template = a.pow(w, two);

        let mut wilds = FxHashMap::default();
        wilds.insert(w, wid);

        let mut bindings = Substitution::default();
        bindings.insert(wid, x);

        let result = instantiate(&mut a, template, &wilds, &bindings);
        assert_eq!(display(&a, result), "x^2");
    }

    // ── Rule application ────────────────────────────────────────────

    #[test]
    fn rule_pythagorean_identity() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);

        // Build sin^2(x) + cos^2(x)
        let sin_x = a.sin(x);
        let cos_x = a.cos(x);
        let sin_sq = a.pow(sin_x, two);
        let cos_sq = a.pow(cos_x, two);
        let expr = a.add(&[sin_sq, cos_sq]);

        // Apply rules.
        let rules = basic_rules(&mut a);
        let (result, steps) = apply_rules(&mut a, expr, &rules);

        assert_eq!(
            display(&a, result),
            "1",
            "sin²(x) + cos²(x) should simplify to 1"
        );
        assert_eq!(steps.len(), 1, "should have one step");
        assert_eq!(steps[0].rule_name, "pythagorean");
    }

    #[test]
    fn rule_pythagorean_with_y() {
        let mut a = Arena::new();
        let y = sym(&mut a, "y");
        let two = a.int(2);

        // Build sin^2(y) + cos^2(y) — different symbol
        let sin_y = a.sin(y);
        let cos_y = a.cos(y);
        let sin_sq = a.pow(sin_y, two);
        let cos_sq = a.pow(cos_y, two);
        let expr = a.add(&[sin_sq, cos_sq]);

        let rules = basic_rules(&mut a);
        let (result, _steps) = apply_rules(&mut a, expr, &rules);
        assert_eq!(display(&a, result), "1");
    }

    #[test]
    fn rule_no_match_returns_unchanged() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.sin(x);

        let rules = basic_rules(&mut a);
        let (result, steps) = apply_rules(&mut a, expr, &rules);

        assert_eq!(result, expr, "no rule should fire on sin(x) alone");
        assert!(steps.is_empty());
    }

    #[test]
    fn rule_pythagorean_in_larger_expression() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let three = a.int(3);

        // Build 3 + sin^2(x) + cos^2(x)
        let sin_x = a.sin(x);
        let cos_x = a.cos(x);
        let sin_sq = a.pow(sin_x, two);
        let cos_sq = a.pow(cos_x, two);
        let expr = a.add(&[three, sin_sq, cos_sq]);

        let rules = basic_rules(&mut a);
        let (result, steps) = apply_rules(&mut a, expr, &rules);

        // Sub-expression matching finds the Pythagorean pair inside
        // the 3-term Add and reduces sin²(x) + cos²(x) → 1, then
        // canonicalization combines 3 + 1 → 4.
        assert_eq!(
            display(&a, result),
            "4",
            "sub-expression matching should find sin²+cos² inside larger Add"
        );
        assert_eq!(steps.len(), 1, "one rule should fire (pythagorean)");
        assert_eq!(steps[0].rule_name, "pythagorean");
    }

    // ── apply_rules tracing ─────────────────────────────────────────

    #[test]
    fn trace_records_all_steps() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);

        let sin_x = a.sin(x);
        let cos_x = a.cos(x);
        let sin_sq = a.pow(sin_x, two);
        let cos_sq = a.pow(cos_x, two);
        let expr = a.add(&[sin_sq, cos_sq]);

        let rules = basic_rules(&mut a);
        let (_result, steps) = apply_rules(&mut a, expr, &rules);

        assert!(!steps.is_empty(), "trace should have at least one entry");
        for step in &steps {
            assert!(!step.rule_name.is_empty());
            // before and after should be different
            assert_ne!(step.before, step.after);
        }
    }

    // ── Two-wild pattern ────────────────────────────────────────────

    #[test]
    fn two_wild_pattern() {
        let mut a = Arena::new();
        let (w1, wid1) = a.wild();
        let (w2, wid2) = a.wild();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");

        // Pattern: w1 + w2
        let pat = a.add(&[w1, w2]);
        let mut wilds = FxHashMap::default();
        wilds.insert(w1, wid1);
        wilds.insert(w2, wid2);
        let pattern = Pattern { root: pat, wilds };

        // Expression: x + y
        let expr = a.add(&[x, y]);
        let result = match_pattern(&a, &pattern, expr);
        assert!(result.is_some());
        let bindings = result.unwrap();
        // The exact binding depends on canonical order — just check both
        // wilds are bound.
        assert!(bindings.contains_key(&wid1));
        assert!(bindings.contains_key(&wid2));
    }
}

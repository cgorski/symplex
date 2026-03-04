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

use num_traits::Signed;
use rustc_hash::FxHashMap;

use crate::arena::Arena;
use crate::assumptions::Props;
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
        (ExprNode::BoolTrue, ExprNode::BoolTrue) => true,
        (ExprNode::BoolFalse, ExprNode::BoolFalse) => true,

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
        (ExprNode::Atan2(py, px), ExprNode::Atan2(ey, ex)) => {
            match_recursive(arena, pattern, py, ey, bindings)
                && match_recursive(arena, pattern, px, ex, bindings)
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
        (ExprNode::Asinh(pi), ExprNode::Asinh(ei)) => {
            match_recursive(arena, pattern, pi, ei, bindings)
        }
        (ExprNode::Acosh(pi), ExprNode::Acosh(ei)) => {
            match_recursive(arena, pattern, pi, ei, bindings)
        }
        (ExprNode::Atanh(pi), ExprNode::Atanh(ei)) => {
            match_recursive(arena, pattern, pi, ei, bindings)
        }
        (ExprNode::Sign(pi), ExprNode::Sign(ei)) => {
            match_recursive(arena, pattern, pi, ei, bindings)
        }
        (ExprNode::Not(pi), ExprNode::Not(ei)) => match_recursive(arena, pattern, pi, ei, bindings),

        // Binary relational.
        (ExprNode::Gt(pa, pb), ExprNode::Gt(ea, eb)) => {
            match_recursive(arena, pattern, pa, ea, bindings)
                && match_recursive(arena, pattern, pb, eb, bindings)
        }
        (ExprNode::Ge(pa, pb), ExprNode::Ge(ea, eb)) => {
            match_recursive(arena, pattern, pa, ea, bindings)
                && match_recursive(arena, pattern, pb, eb, bindings)
        }
        (ExprNode::Eq_(pa, pb), ExprNode::Eq_(ea, eb)) => {
            match_recursive(arena, pattern, pa, ea, bindings)
                && match_recursive(arena, pattern, pb, eb, bindings)
        }
        (ExprNode::Ne(pa, pb), ExprNode::Ne(ea, eb)) => {
            match_recursive(arena, pattern, pa, ea, bindings)
                && match_recursive(arena, pattern, pb, eb, bindings)
        }

        // N-ary logical / piecewise.
        (ExprNode::And(ref pc), ExprNode::And(ref ec)) => {
            if pc.len() != ec.len() {
                return false;
            }
            for (p, e) in pc.iter().zip(ec.iter()) {
                if !match_recursive(arena, pattern, *p, *e, bindings) {
                    return false;
                }
            }
            true
        }
        (ExprNode::Or(ref pc), ExprNode::Or(ref ec)) => {
            if pc.len() != ec.len() {
                return false;
            }
            for (p, e) in pc.iter().zip(ec.iter()) {
                if !match_recursive(arena, pattern, *p, *e, bindings) {
                    return false;
                }
            }
            true
        }
        (ExprNode::Piecewise(ref pc), ExprNode::Piecewise(ref ec)) => {
            if pc.len() != ec.len() {
                return false;
            }
            for (p, e) in pc.iter().zip(ec.iter()) {
                if !match_recursive(arena, pattern, p.0, e.0, bindings) {
                    return false;
                }
                if !match_recursive(arena, pattern, p.1, e.1, bindings) {
                    return false;
                }
            }
            true
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

        // Combinatorial.
        (ExprNode::Factorial(pi), ExprNode::Factorial(ei)) => {
            match_recursive(arena, pattern, pi, ei, bindings)
        }
        (ExprNode::Binomial(pa, pb), ExprNode::Binomial(ea, eb)) => {
            match_recursive(arena, pattern, pa, ea, bindings)
                && match_recursive(arena, pattern, pb, eb, bindings)
        }

        // Floor / Ceiling (unary).
        (ExprNode::Floor(pi), ExprNode::Floor(ei)) => {
            match_recursive(arena, pattern, pi, ei, bindings)
        }
        (ExprNode::Ceiling(pi), ExprNode::Ceiling(ei)) => {
            match_recursive(arena, pattern, pi, ei, bindings)
        }

        // Min / Max (n-ary).
        (ExprNode::Min(ref pc), ExprNode::Min(ref ec)) => {
            if pc.len() != ec.len() {
                return false;
            }
            for (p, e) in pc.iter().zip(ec.iter()) {
                if !match_recursive(arena, pattern, *p, *e, bindings) {
                    return false;
                }
            }
            true
        }
        (ExprNode::Max(ref pc), ExprNode::Max(ref ec)) => {
            if pc.len() != ec.len() {
                return false;
            }
            for (p, e) in pc.iter().zip(ec.iter()) {
                if !match_recursive(arena, pattern, *p, *e, bindings) {
                    return false;
                }
            }
            true
        }

        // Sum / Product_ (4-ary).
        (ExprNode::Sum(pb, pv, pl, ph), ExprNode::Sum(eb, ev, el, eh)) => {
            match_recursive(arena, pattern, pb, eb, bindings)
                && match_recursive(arena, pattern, pv, ev, bindings)
                && match_recursive(arena, pattern, pl, el, bindings)
                && match_recursive(arena, pattern, ph, eh, bindings)
        }
        (ExprNode::Product_(pb, pv, pl, ph), ExprNode::Product_(eb, ev, el, eh)) => {
            match_recursive(arena, pattern, pb, eb, bindings)
                && match_recursive(arena, pattern, pv, ev, bindings)
                && match_recursive(arena, pattern, pl, el, bindings)
                && match_recursive(arena, pattern, ph, eh, bindings)
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
                tracing::debug!(rule = rule.name, "rule fired");
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
        if rewritten == rebuilt
            && let ExprNode::Add(ref children) = arena.node(rebuilt).clone()
            && children.len() >= 2
        {
            'sub_match: for rule in rules {
                // Only attempt if the rule's pattern root is an Add.
                if let ExprNode::Add(ref pat_children) = arena.node(rule.pattern.root).clone() {
                    let k = pat_children.len();
                    if k == 2 && children.len() >= 2 {
                        // Try all pairs of children.
                        for i in 0..children.len() {
                            for j in (i + 1)..children.len() {
                                let pair = arena.add(&[children[i], children[j]]);
                                if let Some(replacement) = rule.try_apply(arena, pair) {
                                    tracing::debug!(
                                        rule = rule.name,
                                        node_type = "Add",
                                        "sub-expression match found"
                                    );
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

        // Mul sub-expression matching: if the node is a Mul and no rule matched
        // the whole node, try matching rules against subsets of the Mul's children.
        if rewritten == rebuilt
            && let ExprNode::Mul(ref children) = arena.node(rebuilt).clone()
            && children.len() >= 2
        {
            'mul_sub_match: for rule in rules {
                // Only attempt if the rule's pattern root is a Mul.
                if let ExprNode::Mul(ref pat_children) = arena.node(rule.pattern.root).clone() {
                    let k = pat_children.len();
                    if k == 2 && children.len() >= 2 {
                        // Try all pairs of children.
                        for i in 0..children.len() {
                            for j in (i + 1)..children.len() {
                                let pair = arena.mul(&[children[i], children[j]]);
                                if let Some(replacement) = rule.try_apply(arena, pair) {
                                    tracing::debug!(
                                        rule = rule.name,
                                        node_type = "Mul",
                                        "sub-expression match found"
                                    );
                                    // Build remaining factors.
                                    let mut remaining: smallvec::SmallVec<[ExprId; 6]> =
                                        smallvec::SmallVec::new();
                                    for (idx, &child) in children.iter().enumerate() {
                                        if idx != i && idx != j {
                                            remaining.push(child);
                                        }
                                    }
                                    remaining.push(replacement);
                                    rewritten = arena.mul(&remaining);
                                    steps.push(Step {
                                        rule_name: rule.name,
                                        before: rebuilt,
                                        after: rewritten,
                                    });
                                    break 'mul_sub_match;
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

/// Helper: build a rule matching `outer(inner(w)) → template(w)`.
fn unary_compose_rule(
    arena: &mut Arena,
    name: &'static str,
    outer_fn: fn(&mut Arena, ExprId) -> ExprId,
    inner_fn: fn(&mut Arena, ExprId) -> ExprId,
    template_fn: fn(&mut Arena, ExprId) -> ExprId,
) -> Rule {
    let (w_expr, w_id) = arena.wild();
    let inner_w = inner_fn(arena, w_expr);
    let pattern_expr = outer_fn(arena, inner_w);
    let template = template_fn(arena, w_expr);
    let mut wilds = FxHashMap::default();
    wilds.insert(w_expr, w_id);
    Rule::new(
        name,
        Pattern {
            root: pattern_expr,
            wilds,
        },
        template,
    )
}

/// Identity template: just returns the wild itself.
fn identity(_arena: &mut Arena, w: ExprId) -> ExprId {
    w
}

/// Build the rule: `sqrt(w^2) → abs(w)`.
fn rule_sqrt_sq(arena: &mut Arena) -> Rule {
    let (w_expr, w_id) = arena.wild();
    let two = arena.int(2);
    let half = arena.rational(1, 2);
    let w_sq = arena.pow(w_expr, two);
    let sqrt_w_sq = arena.pow(w_sq, half); // Pow(Pow(w, 2), 1/2)
    let abs_w = arena.abs(w_expr);

    let mut wilds = FxHashMap::default();
    wilds.insert(w_expr, w_id);
    let pattern = Pattern {
        root: sqrt_w_sq,
        wilds,
    };
    Rule::new("sqrt_sq", pattern, abs_w)
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

/// Power of power: `(w1^w2)^w3 → w1^(w2*w3)`.
///
/// Note: strictly valid for positive bases or integer exponents.
/// May produce incorrect results for negative bases with fractional exponents.
fn rule_pow_pow(arena: &mut Arena) -> Rule {
    let (w1_expr, w1_id) = arena.wild();
    let (w2_expr, w2_id) = arena.wild();
    let (w3_expr, w3_id) = arena.wild();

    let inner_pow = arena.pow(w1_expr, w2_expr);
    let outer_pow = arena.pow(inner_pow, w3_expr);

    let product = arena.mul(&[w2_expr, w3_expr]);
    let template = arena.pow(w1_expr, product);

    let mut wilds = FxHashMap::default();
    wilds.insert(w1_expr, w1_id);
    wilds.insert(w2_expr, w2_id);
    wilds.insert(w3_expr, w3_id);
    let pattern = Pattern {
        root: outer_pow,
        wilds,
    };
    let mut r = Rule::new("pow_pow", pattern, template);
    r.condition = Some(|arena, bindings| {
        // Only fire when at least one bound value is a known integer
        for &val in bindings.values() {
            if let crate::node::ExprNode::Num(nid) = arena.node(val) {
                let r = arena.num(*nid);
                if r.is_integer() {
                    return true;
                }
            }
        }
        false
    });
    r
}

/// Build a basic set of simplification rules.
/// sin(w) / cos(w) → tan(w)
/// Canonical form of sin(x)/cos(x) is Mul([Sin(x), Pow(Cos(x), -1)])
fn rule_sin_div_cos(arena: &mut Arena) -> Rule {
    let (w_expr, w_id) = arena.wild();
    let sin_w = arena.sin(w_expr);
    let cos_w = arena.cos(w_expr);
    let neg_one = arena.int(-1);
    let cos_w_inv = arena.pow(cos_w, neg_one);
    let pattern_expr = arena.mul(&[sin_w, cos_w_inv]);

    let mut wilds = FxHashMap::default();
    wilds.insert(w_expr, w_id);
    let pattern = Pattern {
        root: pattern_expr,
        wilds,
    };
    Rule::new("sin_div_cos", pattern, arena.tan(w_expr))
}

/// cos(w) / sin(w) → 1/tan(w)  (i.e., Pow(tan(w), -1))
fn rule_cos_div_sin(arena: &mut Arena) -> Rule {
    let (w_expr, w_id) = arena.wild();
    let cos_w = arena.cos(w_expr);
    let sin_w = arena.sin(w_expr);
    let neg_one = arena.int(-1);
    let sin_w_inv = arena.pow(sin_w, neg_one);
    let pattern_expr = arena.mul(&[cos_w, sin_w_inv]);
    let tan_w = arena.tan(w_expr);
    let template = arena.pow(tan_w, neg_one);
    let mut wilds = FxHashMap::default();
    wilds.insert(w_expr, w_id);
    let pattern = Pattern {
        root: pattern_expr,
        wilds,
    };
    Rule::new("cos_div_sin", pattern, template)
}

/// sinh(w) / cosh(w) → tanh(w)
fn rule_sinh_div_cosh(arena: &mut Arena) -> Rule {
    let (w_expr, w_id) = arena.wild();
    let sinh_w = arena.sinh(w_expr);
    let cosh_w = arena.cosh(w_expr);
    let neg_one = arena.int(-1);
    let cosh_w_inv = arena.pow(cosh_w, neg_one);
    let pattern_expr = arena.mul(&[sinh_w, cosh_w_inv]);

    let mut wilds = FxHashMap::default();
    wilds.insert(w_expr, w_id);
    let pattern = Pattern {
        root: pattern_expr,
        wilds,
    };
    Rule::new("sinh_div_cosh", pattern, arena.tanh(w_expr))
}

/// exp(a) * exp(b) → exp(a + b)
fn rule_exp_mul(arena: &mut Arena) -> Rule {
    let (a_expr, a_id) = arena.wild();
    let (b_expr, b_id) = arena.wild();
    let exp_a = arena.exp(a_expr);
    let exp_b = arena.exp(b_expr);
    let pattern_expr = arena.mul(&[exp_a, exp_b]);

    let sum = arena.add(&[a_expr, b_expr]);
    let template = arena.exp(sum);

    let mut wilds = FxHashMap::default();
    wilds.insert(a_expr, a_id);
    wilds.insert(b_expr, b_id);
    let pattern = Pattern {
        root: pattern_expr,
        wilds,
    };
    Rule::new("exp_mul", pattern, template)
}

/// exp(a * ln(b)) → b^a  (exp-log denesting)
///
/// Only matches when the argument to exp is a 2-child Mul where one
/// child is ln(something). For 3+ child Mul (like exp(2*x*ln(y))),
/// the pattern system's strict positional matching won't fire.
fn rule_exp_log_denest(arena: &mut Arena) -> Rule {
    let (a_expr, a_id) = arena.wild();
    let (b_expr, b_id) = arena.wild();
    let ln_b = arena.ln(b_expr);
    let product = arena.mul(&[a_expr, ln_b]);
    let pattern_expr = arena.exp(product);
    let template = arena.pow(b_expr, a_expr);

    let mut wilds = FxHashMap::default();
    wilds.insert(a_expr, a_id);
    wilds.insert(b_expr, b_id);
    let pattern = Pattern {
        root: pattern_expr,
        wilds,
    };
    Rule::new("exp_log_denest", pattern, template)
}

// ═══════════════════════════════════════════════════════════════════════════
// Conditional rule helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Condition: every wild-bound value in the substitution is a numeric
/// literal whose value is strictly positive.  Non-numeric bindings
/// (symbols, compound expressions) cause the condition to return `false`
/// because we cannot determine their sign structurally.
fn condition_wild_positive(arena: &Arena, subs: &Substitution) -> bool {
    for &id in subs.values() {
        if let ExprNode::Num(nid) = arena.node(id) {
            if !arena.num(*nid).is_positive() {
                return false;
            }
        } else {
            // Non-numeric: sign unknown → reject.
            return false;
        }
    }
    true
}

/// Condition: every wild-bound value in the substitution is known to be real.
///
/// Numeric literals are always real.  Symbols are checked via their
/// stored assumptions.  The constants π and *e* are real.  Anything
/// else (compound expressions, imaginary unit, etc.) is conservatively
/// rejected.
fn condition_wild_real(arena: &Arena, subs: &Substitution) -> bool {
    for &id in subs.values() {
        match arena.node(id) {
            ExprNode::Num(_) | ExprNode::Pi | ExprNode::E => {
                // These are unconditionally real.
            }
            ExprNode::Symbol(sid) => {
                let assumptions = arena.symbol_assumptions(*sid);
                if assumptions.query(Props::REAL) != Some(true) {
                    return false;
                }
            }
            _ => {
                // Compound or unknown expression: cannot confirm real → reject.
                return false;
            }
        }
    }
    true
}

/// `abs(w) → w` when `w` is a positive numeric literal.
fn rule_abs_positive(arena: &mut Arena) -> Rule {
    let (w_expr, w_id) = arena.wild();
    let abs_w = arena.abs(w_expr);

    let mut wilds = FxHashMap::default();
    wilds.insert(w_expr, w_id);
    let pattern = Pattern { root: abs_w, wilds };

    Rule {
        name: "abs_positive",
        pattern,
        template: w_expr,
        condition: Some(condition_wild_positive),
    }
}

pub(crate) fn basic_rules(arena: &mut Arena) -> Vec<Rule> {
    vec![
        rule_pythagorean(arena),
        unary_compose_rule(arena, "exp_ln", Arena::exp, Arena::ln, identity),
        {
            let mut r = unary_compose_rule(arena, "ln_exp", Arena::ln, Arena::exp, identity);
            r.condition = Some(condition_wild_real);
            r
        },
        unary_compose_rule(arena, "abs_abs", Arena::abs, Arena::abs, Arena::abs),
        rule_sqrt_sq(arena),
        rule_cosh_sinh_identity(arena),
        rule_pow_pow(arena),
        unary_compose_rule(arena, "asinh_sinh", Arena::asinh, Arena::sinh, identity),
        unary_compose_rule(arena, "acosh_cosh", Arena::acosh, Arena::cosh, Arena::abs),
        unary_compose_rule(arena, "atanh_tanh", Arena::atanh, Arena::tanh, identity),
        rule_sin_div_cos(arena),
        rule_cos_div_sin(arena),
        rule_sinh_div_cosh(arena),
        rule_exp_mul(arena),
        rule_exp_log_denest(arena),
        rule_abs_positive(arena),
        unary_compose_rule(arena, "sin_asin", Arena::sin, Arena::asin, identity),
        unary_compose_rule(arena, "cos_acos", Arena::cos, Arena::acos, identity),
        unary_compose_rule(arena, "tan_atan", Arena::tan, Arena::atan, identity),
        unary_compose_rule(arena, "sinh_asinh", Arena::sinh, Arena::asinh, identity),
        unary_compose_rule(arena, "cosh_acosh", Arena::cosh, Arena::acosh, identity),
        unary_compose_rule(arena, "tanh_atanh", Arena::tanh, Arena::atanh, identity),
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

    #[test]
    fn mul_sub_match_exp_combine() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let z = sym(&mut a, "z");

        // Build exp(x) * exp(y) * z
        let exp_x = a.exp(x);
        let exp_y = a.exp(y);
        let expr = a.mul(&[exp_x, exp_y, z]);

        let rules = basic_rules(&mut a);
        let (result, steps) = apply_rules(&mut a, expr, &rules);

        let result_str = display(&a, result);
        // Should contain exp(x + y) and z multiplied together
        assert!(
            result_str.contains("exp("),
            "result should contain exp: got {result_str}"
        );
        assert!(
            result_str.contains("z"),
            "result should contain z: got {result_str}"
        );
        assert!(
            steps.iter().any(|s| s.rule_name == "exp_mul"),
            "exp_mul rule should have fired"
        );
    }

    #[test]
    fn simplify_sin_over_cos() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");

        // Build sin(x) * cos(x)^(-1)
        let sin_x = a.sin(x);
        let cos_x = a.cos(x);
        let neg_one = a.int(-1);
        let cos_x_inv = a.pow(cos_x, neg_one);
        let expr = a.mul(&[sin_x, cos_x_inv]);

        let rules = basic_rules(&mut a);
        let (result, steps) = apply_rules(&mut a, expr, &rules);

        assert_eq!(
            display(&a, result),
            "tan(x)",
            "sin(x)/cos(x) should simplify to tan(x)"
        );
        assert!(steps.iter().any(|s| s.rule_name == "sin_div_cos"));
    }

    #[test]
    fn simplify_sinh_over_cosh() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");

        // Build sinh(x) * cosh(x)^(-1)
        let sinh_x = a.sinh(x);
        let cosh_x = a.cosh(x);
        let neg_one = a.int(-1);
        let cosh_x_inv = a.pow(cosh_x, neg_one);
        let expr = a.mul(&[sinh_x, cosh_x_inv]);

        let rules = basic_rules(&mut a);
        let (result, steps) = apply_rules(&mut a, expr, &rules);

        assert_eq!(
            display(&a, result),
            "tanh(x)",
            "sinh(x)/cosh(x) should simplify to tanh(x)"
        );
        assert!(steps.iter().any(|s| s.rule_name == "sinh_div_cosh"));
    }

    #[test]
    fn simplify_sin_over_cos_in_product() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);

        // Build 2 * sin(x) / cos(x) = 2 * sin(x) * cos(x)^(-1)
        let sin_x = a.sin(x);
        let cos_x = a.cos(x);
        let neg_one = a.int(-1);
        let cos_x_inv = a.pow(cos_x, neg_one);
        let expr = a.mul(&[two, sin_x, cos_x_inv]);

        let rules = basic_rules(&mut a);
        let (result, steps) = apply_rules(&mut a, expr, &rules);

        let result_str = display(&a, result);
        assert!(
            result_str.contains("tan(x)"),
            "2*sin(x)/cos(x) should simplify to 2*tan(x): got {result_str}"
        );
        assert!(
            result_str.contains("2"),
            "result should still contain factor 2: got {result_str}"
        );
        assert!(
            steps.iter().any(|s| s.rule_name == "sin_div_cos"),
            "sin_div_cos rule should have fired"
        );
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

    // ── Conditional rules ───────────────────────────────────────────

    #[test]
    fn conditional_abs_positive() {
        let mut a = Arena::new();
        let five = a.int(5);
        let abs_five = a.abs(five);
        let rules = basic_rules(&mut a);
        let (result, steps) = apply_rules(&mut a, abs_five, &rules);
        assert_eq!(display(&a, result), "5");
        assert!(!steps.is_empty(), "should have fired abs_positive rule");
    }

    #[test]
    fn conditional_abs_negative_unchanged() {
        let mut a = Arena::new();
        let neg_five = a.int(-5);
        let abs_neg = a.abs(neg_five);
        let rules = basic_rules(&mut a);
        let (result, _) = apply_rules(&mut a, abs_neg, &rules);
        // Should NOT fire — -5 is not positive
        assert_eq!(display(&a, result), "abs(-5)");
    }

    #[test]
    fn conditional_abs_symbol_unchanged() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let abs_x = a.abs(x);
        let rules = basic_rules(&mut a);
        let (result, _) = apply_rules(&mut a, abs_x, &rules);
        // Symbol has unknown sign — should not fire
        assert_eq!(display(&a, result), "abs(x)");
    }

    #[test]
    fn simplify_sin_asin() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let asin_x = a.asin(x);
        let sin_asin_x = a.sin(asin_x);
        let rules = basic_rules(&mut a);
        let (result, steps) = apply_rules(&mut a, sin_asin_x, &rules);
        assert_eq!(display(&a, result), "x");
        assert!(steps.iter().any(|s| s.rule_name == "sin_asin"));
    }

    #[test]
    fn simplify_cos_acos() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let acos_x = a.acos(x);
        let cos_acos_x = a.cos(acos_x);
        let rules = basic_rules(&mut a);
        let (result, steps) = apply_rules(&mut a, cos_acos_x, &rules);
        assert_eq!(display(&a, result), "x");
        assert!(steps.iter().any(|s| s.rule_name == "cos_acos"));
    }

    #[test]
    fn simplify_tan_atan() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let atan_x = a.atan(x);
        let tan_atan_x = a.tan(atan_x);
        let rules = basic_rules(&mut a);
        let (result, steps) = apply_rules(&mut a, tan_atan_x, &rules);
        assert_eq!(display(&a, result), "x");
        assert!(steps.iter().any(|s| s.rule_name == "tan_atan"));
    }

    #[test]
    fn simplify_sinh_asinh() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let asinh_x = a.asinh(x);
        let sinh_asinh_x = a.sinh(asinh_x);
        let rules = basic_rules(&mut a);
        let (result, steps) = apply_rules(&mut a, sinh_asinh_x, &rules);
        assert_eq!(display(&a, result), "x");
        assert!(steps.iter().any(|s| s.rule_name == "sinh_asinh"));
    }

    #[test]
    fn pow_pow_blocked_for_fractional_exponents() {
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        let half = arena.rational(1, 2);
        let third = arena.rational(1, 3);
        let inner = arena.pow(x, half);
        let expr = arena.pow(inner, third); // (x^(1/2))^(1/3)
        let rules = basic_rules(&mut arena);
        let (result, _) = apply_rules(&mut arena, expr, &rules);
        // Should NOT simplify to x^(1/6) because no exponent is integer
        assert_eq!(
            result, expr,
            "pow_pow should not fire for fractional exponents"
        );
    }

    #[test]
    fn pow_pow_fires_for_integer_exponent() {
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        let two = arena.int(2);
        let three = arena.int(3);
        let inner = arena.pow(x, two);
        let expr = arena.pow(inner, three); // (x^2)^3
        let rules = basic_rules(&mut arena);
        let (result, _) = apply_rules(&mut arena, expr, &rules);
        let six = arena.int(6);
        let expected = arena.pow(x, six); // x^6
        assert_eq!(
            result, expected,
            "pow_pow should fire for integer exponents"
        );
    }

    #[test]
    fn acosh_cosh_gives_abs() {
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        let cosh_x = arena.cosh(x);
        let expr = arena.acosh(cosh_x);
        let rules = basic_rules(&mut arena);
        let (result, _) = apply_rules(&mut arena, expr, &rules);
        let expected = arena.abs(x);
        assert_eq!(result, expected, "acosh(cosh(x)) should give |x|");
    }

    #[test]
    fn asin_sin_no_longer_simplifies() {
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        let sin_x = arena.sin(x);
        let expr = arena.asin(sin_x);
        let rules = basic_rules(&mut arena);
        let (result, _) = apply_rules(&mut arena, expr, &rules);
        assert_eq!(result, expr, "asin(sin(x)) should stay (rule removed)");
    }

    #[test]
    fn exp_log_denest_simplifies() {
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        let three = arena.int(3);
        let ln_x = arena.ln(x);
        let product = arena.mul(&[three, ln_x]);
        let expr = arena.exp(product); // exp(3*ln(x))
        let rules = basic_rules(&mut arena);
        let (result, _) = apply_rules(&mut arena, expr, &rules);
        let expected = arena.pow(x, three); // x^3
        assert_eq!(result, expected, "exp(3*ln(x)) should simplify to x^3");
    }
}

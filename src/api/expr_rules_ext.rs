//! Public rewrite-rule engine and simplification extensions on [`Ex`].
//!
//! This module is the user-facing half of the pattern-matching engine in
//! `crate::transforms::pattern` (crate-internal).  It provides:
//!
//! * [`Rule`] — a named rewrite rule built from two expressions
//!   (`lhs → rhs`) whose symbols ending in `_` are wildcards, optionally
//!   with a guard closure or a closure-computed right-hand side;
//! * [`RuleSet`] — an ordered collection of rules;
//! * [`Bindings`] — the wildcard → sub-expression map handed to guards;
//! * [`RewriteOpts`] / [`RewriteStrategy`] — traversal configuration;
//! * [`Step`] — one entry of a rewrite / simplification trace;
//! * the `Ex` methods [`rewrite`](Ex::rewrite), [`rewrite_once`](Ex::rewrite_once),
//!   [`rewrite_traced`](Ex::rewrite_traced), [`rewrite_with`](Ex::rewrite_with),
//!   [`simplify_with_rules`](Ex::simplify_with_rules),
//!   [`simplify_traced`](Ex::simplify_traced) and
//!   [`subs_algebraic`](Ex::subs_algebraic).
//!
//! # Wildcard conventions
//!
//! | Name        | Meaning                                                        |
//! |-------------|----------------------------------------------------------------|
//! | `a_`        | matches any single sub-expression (in `Add`/`Mul`: one term, or the remaining terms if it is the last plain wildcard) |
//! | `rest__`    | *sequence* wildcard: absorbs the remaining terms of an `Add`/`Mul`, possibly none (binding to `0`/`1`) |
//!
//! `Add` and `Mul` are matched associatively and commutatively with a
//! bounded backtracking search (see the internal `MATCH_BUDGET` constant);
//! every other node is matched structurally.
//!
//! # Example
//!
//! ```
//! use symplex::prelude::*;
//! use symplex::macros::{Rule, RuleSet};
//!
//! let ctx = Context::new();
//! let (x, a) = (ctx.symbol("x"), ctx.symbol("a_"));
//! // sin(a_)^2 → 1 - cos(a_)^2
//! let rule = Rule::new("sin_sq", &a.sin().powi(2), &(1 - &a.cos().powi(2)));
//! let rules = RuleSet::from_rules(vec![rule]);
//! let expr = &x.sin().powi(2) + 3;
//! let result = expr.rewrite(&rules);
//! assert_eq!(format!("{result}"), "-cos(x)^2 + 4");
//! ```

use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::api::context::Context;
use crate::api::expr::{Ex, Expr, Numeric, SimplifyOpts};
use crate::base::arena::Arena;
use crate::base::errors::SymplexError;
use crate::base::node::{ExprId, ExprNode};
use crate::base::walk;
use crate::transforms::expand::ExpandOpts;
use crate::transforms::pattern::{
    self, MATCH_BUDGET, MatchResult, Pattern, RawStep, Substitution, WildId,
};

// ═══════════════════════════════════════════════════════════════════════════
// Step
// ═══════════════════════════════════════════════════════════════════════════

/// One step of a rewrite or simplification trace.
///
/// `before` and `after` are the sub-expression that was rewritten; for
/// simplification traces, strategy-level entries are named
/// `strategy:<name>` and span the whole expression, while rule-level
/// entries carry the rule name.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::macros::RuleSet;
///
/// let ctx = Context::new();
/// let x = ctx.symbol("x");
/// let expr = &x.sin().powi(2) + &x.cos().powi(2);
/// let (result, steps) = expr.rewrite_traced(&RuleSet::standard(&ctx));
/// assert_eq!(format!("{result}"), "1");
/// assert_eq!(steps[0].rule_name, "pythagorean");
/// assert_eq!(format!("{}", steps[0].after), "1");
/// ```
#[derive(Clone, Debug)]
pub struct Step {
    /// The name of the rule (or simplification strategy) that fired.
    pub rule_name: String,
    /// The sub-expression before the rewrite.
    pub before: Ex,
    /// The sub-expression after the rewrite.
    pub after: Ex,
}

impl Step {
    fn from_raw(template: &Ex, raw: RawStep) -> Step {
        Step {
            rule_name: raw.rule_name,
            before: template.wrap(raw.before),
            after: template.wrap(raw.after),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Bindings
// ═══════════════════════════════════════════════════════════════════════════

/// Wildcard bindings produced by a successful match: wildcard name →
/// matched sub-expression.
///
/// Names are the symbol names used in the rule's left-hand side
/// (including the trailing underscore), e.g. `"a_"` or `"rest__"`.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::macros::Rule;
///
/// let ctx = Context::new();
/// let (x, y, a, b) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("a_"), ctx.symbol("b_"));
/// let rule = Rule::new("swap", &a.pow(&b), &b.pow(&a));
/// let bindings = rule.matches(&x.pow(&y)).unwrap();
/// assert_eq!(bindings.get("a_"), Some(&x));
/// assert_eq!(bindings.get("b_"), Some(&y));
/// assert_eq!(bindings.len(), 2);
/// ```
#[derive(Clone, Debug, Default)]
pub struct Bindings {
    map: FxHashMap<String, Ex>,
}

impl Bindings {
    /// Look up the sub-expression bound to wildcard `name`.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Ex> {
        self.map.get(name)
    }

    /// Number of bound wildcards.
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// `true` if no wildcard is bound.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Iterate over `(name, value)` pairs (unspecified order).
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Ex)> {
        self.map.iter().map(|(k, v)| (k.as_str(), v))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// RewriteOpts / RewriteStrategy
// ═══════════════════════════════════════════════════════════════════════════

/// Traversal order used by [`Ex::rewrite_with`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RewriteStrategy {
    /// Leaves first: children are rewritten before their parent, one rule
    /// application per node per pass.  Passes repeat until no rule fires.
    #[default]
    BottomUp,
    /// Root first: rules are applied at a node (repeatedly, until none
    /// fires) *before* its children are visited.  Passes repeat until no
    /// rule fires.
    TopDown,
    /// Innermost-first normalisation: like [`BottomUp`](Self::BottomUp),
    /// but every replacement is itself fully normalised before the walk
    /// continues upward, so each pass produces a rule-normal form.
    Innermost,
}

/// Options controlling [`Ex::rewrite_with`].
///
/// # Examples
///
/// ```
/// use symplex::macros::{RewriteOpts, RewriteStrategy};
///
/// let opts = RewriteOpts { max_iterations: 5, strategy: RewriteStrategy::TopDown };
/// assert_eq!(opts.max_iterations, 5);
/// assert_eq!(RewriteOpts::default().strategy, RewriteStrategy::BottomUp);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RewriteOpts {
    /// Maximum number of full passes over the expression (default `50`).
    /// Rewriting stops early as soon as a pass changes nothing.
    pub max_iterations: usize,
    /// Traversal order (default [`RewriteStrategy::BottomUp`]).
    pub strategy: RewriteStrategy,
}

impl Default for RewriteOpts {
    fn default() -> Self {
        RewriteOpts {
            max_iterations: 50,
            strategy: RewriteStrategy::BottomUp,
        }
    }
}

impl RewriteOpts {
    /// Exactly one pass, bottom-up.
    #[must_use]
    pub fn single_pass() -> Self {
        RewriteOpts {
            max_iterations: 1,
            strategy: RewriteStrategy::BottomUp,
        }
    }

    /// Builder: set the iteration cap.
    #[must_use]
    pub fn max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Builder: set the traversal strategy.
    #[must_use]
    pub fn strategy(mut self, strategy: RewriteStrategy) -> Self {
        self.strategy = strategy;
        self
    }
}

/// Tree-size ceiling for rewriting: a pass whose result has more than this
/// many nodes (counted with multiplicity, i.e. the unfolded tree rather
/// than the hash-consed DAG) stops the fixpoint iteration and the previous
/// result is kept.  Guards against rules that grow expressions without
/// bound.
pub const MAX_REWRITE_OPS: usize = 100_000;

/// Maximum number of successive rule applications at a single node in
/// the `TopDown` / `Innermost` strategies before the walk moves on.
const MAX_NODE_REWRITES: usize = 32;

/// Maximum nesting of replacement normalisation in the `Innermost` strategy.
const MAX_INNERMOST_NESTING: usize = 8;

// ═══════════════════════════════════════════════════════════════════════════
// Rule
// ═══════════════════════════════════════════════════════════════════════════

type GuardFn = Arc<dyn Fn(&Bindings) -> bool + Send + Sync>;
type RhsFn = Arc<dyn Fn(&Bindings) -> Option<Ex> + Send + Sync>;

#[derive(Clone)]
enum Rhs {
    Template(ExprId),
    Closure(RhsFn),
}

#[derive(Clone)]
enum Guard {
    None,
    Arena(fn(&Arena, &Substitution) -> bool),
    Closure(GuardFn),
}

/// A named rewrite rule `lhs → rhs` over expressions of one [`Context`].
///
/// Symbols in `lhs` whose names end in `_` are wildcards; a name ending
/// in `__` is a *sequence* wildcard that absorbs the remaining terms of
/// an `Add`/`Mul` (see the `Ex::rewrite` docs).  The right-hand side is
/// either a template expression (wildcards are substituted) or a closure
/// receiving the [`Bindings`].
///
/// Rules are `Clone + Send + Sync`.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::macros::Rule;
///
/// let ctx = Context::new();
/// let (x, a) = (ctx.symbol("x"), ctx.symbol("a_"));
///
/// // Template rule: exp(ln(a_)) → a_
/// let r = Rule::new("exp_ln", &a.ln().exp(), &a);
/// assert_eq!(r.apply(&x.ln().exp()), Some(x.clone()));
/// assert_eq!(r.apply(&x.exp()), None);
///
/// // Guarded rule: only fire when the bound value is a number.
/// let g = Rule::new_with_guard("sin_num", &a.sin(), &ctx.int(0), |b| {
///     b.get("a_").is_some_and(|v| v.expr_type() == ExprType::Number)
/// });
/// assert!(g.apply(&x.sin()).is_none());
/// assert_eq!(g.apply(&ctx.int(7).sin()), Some(ctx.int(0)));
/// ```
#[derive(Clone)]
pub struct Rule {
    name: String,
    lhs: Ex,
    pattern: Pattern,
    wild_names: FxHashMap<WildId, String>,
    rhs: Rhs,
    guard: Guard,
}

impl std::fmt::Debug for Rule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Rule")
            .field("name", &self.name)
            .field("lhs", &self.lhs)
            .finish_non_exhaustive()
    }
}

impl Rule {
    /// Build a template rule `lhs → rhs`.
    ///
    /// Wildcards in `rhs` that do not occur in `lhs` are left as literal
    /// symbols in the output; use [`try_new`](Self::try_new) to reject
    /// such rules.
    ///
    /// # Panics
    ///
    /// Panics if `lhs` and `rhs` belong to different contexts.
    #[must_use]
    pub fn new(name: impl Into<String>, lhs: &Ex, rhs: &Ex) -> Rule {
        let rhs_id = lhs.checked_id(rhs);
        let (pattern, wild_names) = Self::compile_lhs(lhs);
        Rule {
            name: name.into(),
            lhs: lhs.clone(),
            pattern,
            wild_names,
            rhs: Rhs::Template(rhs_id),
            guard: Guard::None,
        }
    }

    /// Like [`new`](Self::new), but returns an error when `rhs` mentions
    /// a wildcard that is not bound by `lhs`, or when `lhs` has no
    /// structure at all (a bare wildcard would match everything).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::macros::Rule;
    ///
    /// let ctx = Context::new();
    /// let (a, b) = (ctx.symbol("a_"), ctx.symbol("b_"));
    /// assert!(Rule::try_new("bad", &a.sin(), &b).is_err());
    /// assert!(Rule::try_new("ok", &a.sin(), &a).is_ok());
    /// ```
    pub fn try_new(name: impl Into<String>, lhs: &Ex, rhs: &Ex) -> Result<Rule, SymplexError> {
        let rule = Rule::new(name, lhs, rhs);
        if rule.pattern.wilds.contains_key(&rule.pattern.root) {
            return Err(SymplexError::InvalidArgument {
                operation: "Rule::try_new",
                reason: "left-hand side is a bare wildcard and would match every expression"
                    .to_string(),
            });
        }
        let lhs_names: Vec<&String> = rule.wild_names.values().collect();
        let unbound = {
            let inner = lhs.inner.read();
            let rhs_id = rule.lhs.checked_id(rhs);
            walk::post_order_ids(&inner.arena, rhs_id)
                .into_iter()
                .filter(|&id| pattern::is_wild_symbol(&inner.arena, id))
                .filter_map(|id| match inner.arena.node(id) {
                    ExprNode::Symbol(sid) => Some(inner.arena.symbol_name(*sid).to_string()),
                    _ => None,
                })
                .find(|n| !lhs_names.contains(&n))
        };
        if let Some(n) = unbound {
            return Err(SymplexError::InvalidArgument {
                operation: "Rule::try_new",
                reason: format!(
                    "wildcard `{n}` appears in the right-hand side but not in the left-hand side"
                ),
            });
        }
        Ok(rule)
    }

    /// Build a template rule with a guard: the rewrite fires only when
    /// `guard(&bindings)` returns `true`.
    ///
    /// The guard runs without holding the context lock, so it may call
    /// any `Ex` method (e.g. `is_positive()`, `is_number()`).
    ///
    /// # Panics
    ///
    /// Panics if `lhs` and `rhs` belong to different contexts.
    #[must_use]
    pub fn new_with_guard(
        name: impl Into<String>,
        lhs: &Ex,
        rhs: &Ex,
        guard: impl Fn(&Bindings) -> bool + Send + Sync + 'static,
    ) -> Rule {
        let mut rule = Rule::new(name, lhs, rhs);
        rule.guard = Guard::Closure(Arc::new(guard));
        rule
    }

    /// Build a rule whose right-hand side is computed by a closure.
    ///
    /// Returning `None` from the closure means "does not apply" (the next
    /// match / rule is tried).  The returned expression must belong to
    /// the same context as `lhs`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::macros::{Rule, RuleSet};
    ///
    /// let ctx = Context::new();
    /// let (x, a) = (ctx.symbol("x"), ctx.symbol("a_"));
    /// // Evaluate ln of perfect powers of e: ln(exp(a_)) → a_ only if a_ is a number.
    /// let r = Rule::new_fn("ln_exp_num", &a.exp().ln(), |b| {
    ///     let v = b.get("a_")?;
    ///     (v.expr_type() == ExprType::Number).then(|| v.clone())
    /// });
    /// let rules = RuleSet::from_rules(vec![r]);
    /// assert_eq!(format!("{}", ctx.int(3).exp().ln().rewrite(&rules)), "3");
    /// assert_eq!(format!("{}", x.exp().ln().rewrite(&rules)), "ln(exp(x))");
    /// ```
    #[must_use]
    pub fn new_fn(
        name: impl Into<String>,
        lhs: &Ex,
        f: impl Fn(&Bindings) -> Option<Ex> + Send + Sync + 'static,
    ) -> Rule {
        let (pattern, wild_names) = Self::compile_lhs(lhs);
        Rule {
            name: name.into(),
            lhs: lhs.clone(),
            pattern,
            wild_names,
            rhs: Rhs::Closure(Arc::new(f)),
            guard: Guard::None,
        }
    }

    /// Wrap an arena-level rule produced by the [`rule!`](crate::rule)
    /// macro (built inside `ctx.with_arena_mut(|arena| ...)`).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::macros::{Rule, RuleSet};
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let raw = ctx.with_arena_mut(|arena| rule!(arena, "pyth", sin(w_)^2 + cos(w_)^2 => 1));
    /// let rule = Rule::from_macro_rule(&ctx, raw);
    /// let rules = RuleSet::from_rules(vec![rule]);
    /// let expr = &x.sin().powi(2) + &x.cos().powi(2) + 2;
    /// assert_eq!(format!("{}", expr.rewrite(&rules)), "3");
    /// ```
    #[must_use]
    pub fn from_macro_rule(ctx: &Context, rule: pattern::Rule) -> Rule {
        let pattern::Rule {
            name,
            pattern,
            template,
            condition,
        } = rule;
        let lhs = Ex::from_raw_parts(ctx.id, Arc::clone(&ctx.inner), pattern.root);
        let wild_names = {
            let inner = ctx.inner.read();
            pattern
                .wilds
                .iter()
                .map(|(&id, &wid)| {
                    let name = match inner.arena.node(id) {
                        ExprNode::Symbol(sid) => inner.arena.symbol_name(*sid).to_string(),
                        _ => format!("_w{}", wid.0),
                    };
                    (wid, name)
                })
                .collect()
        };
        Rule {
            name: name.to_string(),
            lhs,
            pattern,
            wild_names,
            rhs: Rhs::Template(template),
            guard: match condition {
                Some(f) => Guard::Arena(f),
                None => Guard::None,
            },
        }
    }

    fn compile_lhs(lhs: &Ex) -> (Pattern, FxHashMap<WildId, String>) {
        let inner = lhs.inner.read();
        pattern::pattern_from_expr(&inner.arena, lhs.raw_id())
    }

    /// The rule's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The left-hand side (pattern) expression.
    #[must_use]
    pub fn lhs(&self) -> &Ex {
        &self.lhs
    }

    /// Names of the wildcards bound by this rule's left-hand side.
    #[must_use]
    pub fn wildcards(&self) -> Vec<String> {
        let mut v: Vec<String> = self.wild_names.values().cloned().collect();
        v.sort();
        v
    }

    /// Match the rule's left-hand side against the *whole* of `expr`
    /// (no traversal, no partial `Add`/`Mul` match) and return the
    /// bindings on success.  Guards are honoured.
    #[must_use]
    pub fn matches(&self, expr: &Ex) -> Option<Bindings> {
        let id = self.lhs.checked_id(expr);
        let matches = {
            let mut inner = self.lhs.inner.write();
            pattern::match_all(&mut inner.arena, &self.pattern, id, false, 8, MATCH_BUDGET)
        };
        matches.into_iter().find_map(|m| {
            self.check_guard(&m)
                .map(|b| b.unwrap_or_else(|| self.bindings_of(&m.bindings)))
        })
    }

    /// Apply the rule at the root of `expr` (no traversal).
    ///
    /// For an `Add`/`Mul` subject the pattern may match a subset of the
    /// terms; the remaining terms are re-attached to the replacement.
    /// Returns `None` if the rule does not apply.
    #[must_use]
    pub fn apply(&self, expr: &Ex) -> Option<Ex> {
        let id = self.lhs.checked_id(expr);
        self.apply_id(id, true).map(|r| self.lhs.wrap(r))
    }

    /// Convert arena-level bindings to [`Bindings`].
    fn bindings_of(&self, subs: &Substitution) -> Bindings {
        let mut map = FxHashMap::default();
        for (&wid, &id) in subs {
            let name = self
                .wild_names
                .get(&wid)
                .cloned()
                .unwrap_or_else(|| format!("_w{}", wid.0));
            map.insert(name, self.lhs.wrap(id));
        }
        Bindings { map }
    }

    /// Evaluate the guard for one match.  Returns `None` when rejected,
    /// `Some(bindings)` (possibly not yet materialised) when accepted.
    fn check_guard(&self, m: &MatchResult) -> Option<Option<Bindings>> {
        match &self.guard {
            Guard::None => Some(None),
            Guard::Arena(f) => {
                let inner = self.lhs.inner.read();
                if f(&inner.arena, &m.bindings) {
                    Some(None)
                } else {
                    None
                }
            }
            Guard::Closure(g) => {
                let b = self.bindings_of(&m.bindings);
                if g(&b) { Some(Some(b)) } else { None }
            }
        }
    }

    /// Core: try the rule at arena node `id`.  Locks are released around
    /// user closures.
    pub(crate) fn apply_id(&self, id: ExprId, allow_partial: bool) -> Option<ExprId> {
        let max_results = match (&self.guard, &self.rhs) {
            (Guard::None, Rhs::Template(_)) => 1,
            _ => 8,
        };
        let matches = {
            let mut inner = self.lhs.inner.write();
            pattern::match_all(
                &mut inner.arena,
                &self.pattern,
                id,
                allow_partial,
                max_results,
                MATCH_BUDGET,
            )
        };
        for m in matches {
            let Some(guard_bindings) = self.check_guard(&m) else {
                continue;
            };
            let replacement = match &self.rhs {
                Rhs::Template(t) => {
                    let mut inner = self.lhs.inner.write();
                    pattern::instantiate(&mut inner.arena, *t, &self.pattern.wilds, &m.bindings)
                }
                Rhs::Closure(f) => {
                    let b = guard_bindings.unwrap_or_else(|| self.bindings_of(&m.bindings));
                    match f(&b) {
                        Some(r) => self.lhs.checked_id(&r),
                        None => continue,
                    }
                }
            };
            let mut inner = self.lhs.inner.write();
            let result = pattern::reattach_leftover(&mut inner.arena, id, replacement, &m.leftover);
            return Some(result);
        }
        None
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// RuleSet
// ═══════════════════════════════════════════════════════════════════════════

/// An ordered collection of [`Rule`]s.
///
/// At each node the rules are tried in order and the first one that
/// applies wins.  `RuleSet` is `Clone + Send + Sync`.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::macros::{Rule, RuleSet};
///
/// let ctx = Context::new();
/// let (x, a) = (ctx.symbol("x"), ctx.symbol("a_"));
/// let rules = RuleSet::new()
///     .with(Rule::new("cos_sq", &a.cos().powi(2), &(1 - &a.sin().powi(2))))
///     .with(Rule::new("tan", &(&a.sin() / &a.cos()), &a.tan()));
/// assert_eq!(rules.len(), 2);
/// let expr = &x.cos().powi(2) + &x.sin() / &x.cos();
/// assert_eq!(format!("{}", expr.rewrite(&rules)), "-sin(x)^2 + tan(x) + 1");
/// ```
#[derive(Clone, Debug, Default)]
pub struct RuleSet {
    rules: Vec<Rule>,
}

impl RuleSet {
    /// An empty rule set.
    #[must_use]
    pub fn new() -> Self {
        RuleSet::default()
    }

    /// Build from a vector of rules (order is preserved).
    #[must_use]
    pub fn from_rules(rules: Vec<Rule>) -> Self {
        RuleSet { rules }
    }

    /// Wrap arena-level rules produced by the [`rule!`](crate::rule) macro.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::macros::RuleSet;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let raw = ctx.with_arena_mut(|arena| {
    ///     vec![
    ///         rule!(arena, "exp_ln", exp(ln(w_)) => w_),
    ///         rule!(arena, "sin_asin", sin(asin(w_)) => w_),
    ///     ]
    /// });
    /// let rules = RuleSet::from_macro_rules(&ctx, raw);
    /// assert_eq!(format!("{}", (&x.ln().exp() + &x.asin().sin()).rewrite(&rules)), "2*x");
    /// ```
    #[must_use]
    pub fn from_macro_rules(ctx: &Context, rules: Vec<pattern::Rule>) -> Self {
        RuleSet {
            rules: rules
                .into_iter()
                .map(|r| Rule::from_macro_rule(ctx, r))
                .collect(),
        }
    }

    /// The built-in simplification rules used by
    /// [`simplify`](Ex::simplify) (Pythagorean identities, `exp(ln x) → x`,
    /// `sin/cos → tan`, inverse-function compositions, …), as a rule set
    /// for `ctx`.
    #[must_use]
    pub fn standard(ctx: &Context) -> Self {
        let raw = {
            let mut inner = ctx.inner.write();
            pattern::basic_rules(&mut inner.arena)
        };
        RuleSet::from_macro_rules(ctx, raw)
    }

    /// Append a rule.
    pub fn push(&mut self, rule: Rule) {
        self.rules.push(rule);
    }

    /// Builder-style append.
    #[must_use]
    pub fn with(mut self, rule: Rule) -> Self {
        self.rules.push(rule);
        self
    }

    /// Append all rules of `other`.
    pub fn extend(&mut self, other: &RuleSet) {
        self.rules.extend(other.rules.iter().cloned());
    }

    /// Number of rules.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// `true` if the set has no rules.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Iterate over the rules in order.
    pub fn iter(&self) -> impl Iterator<Item = &Rule> {
        self.rules.iter()
    }

    /// The rules as a slice.
    #[must_use]
    pub fn rules(&self) -> &[Rule] {
        &self.rules
    }

    /// Try every rule in order at arena node `id`; first hit wins.
    fn apply_first(&self, id: ExprId) -> Option<(ExprId, &str)> {
        for rule in &self.rules {
            if let Some(r) = rule.apply_id(id, true)
                && r != id
            {
                return Some((r, rule.name.as_str()));
            }
        }
        None
    }
}

impl FromIterator<Rule> for RuleSet {
    fn from_iter<I: IntoIterator<Item = Rule>>(iter: I) -> Self {
        RuleSet {
            rules: iter.into_iter().collect(),
        }
    }
}

impl From<Vec<Rule>> for RuleSet {
    fn from(rules: Vec<Rule>) -> Self {
        RuleSet { rules }
    }
}

impl std::ops::Index<usize> for RuleSet {
    type Output = Rule;
    fn index(&self, i: usize) -> &Rule {
        &self.rules[i]
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Rewrite driver
// ═══════════════════════════════════════════════════════════════════════════

/// Internal traversal state shared by the strategies.
struct Driver<'a> {
    template: &'a Ex,
    rules: &'a RuleSet,
    steps: Vec<RawStep>,
    trace: bool,
}

impl Driver<'_> {
    fn with_arena<R>(&self, f: impl FnOnce(&Arena) -> R) -> R {
        let inner = self.template.inner.read();
        f(&inner.arena)
    }

    fn with_arena_mut<R>(&self, f: impl FnOnce(&mut Arena) -> R) -> R {
        let mut inner = self.template.inner.write();
        f(&mut inner.arena)
    }

    fn record(&mut self, name: &str, before: ExprId, after: ExprId) {
        if self.trace {
            self.steps.push(RawStep {
                rule_name: name.to_string(),
                before,
                after,
            });
        }
    }

    /// Apply the first matching rule at `id`, recording a step.
    fn rewrite_at(&mut self, id: ExprId) -> Option<ExprId> {
        let (new, name) = self.rules.apply_first(id)?;
        let name = name.to_string();
        self.record(&name, id, new);
        Some(new)
    }

    /// Apply rules at `id` repeatedly (bounded) until none fires.
    fn rewrite_at_fixpoint(&mut self, id: ExprId) -> ExprId {
        let mut current = id;
        for _ in 0..MAX_NODE_REWRITES {
            match self.rewrite_at(current) {
                Some(new) if new != current => current = new,
                _ => break,
            }
        }
        current
    }

    /// One bottom-up pass.
    fn pass_bottom_up(&mut self, root: ExprId) -> ExprId {
        let post_order = self.with_arena(|a| walk::post_order_ids(a, root));
        let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();
        for &id in &post_order {
            let rebuilt = self.with_arena_mut(|a| {
                if a.node(id).is_atom() {
                    id
                } else {
                    walk::rebuild_with_cache(a, id, &cache)
                }
            });
            let rewritten = self.rewrite_at(rebuilt).unwrap_or(rebuilt);
            cache.insert(id, rewritten);
        }
        cache.get(&root).copied().unwrap_or(root)
    }

    /// One innermost-first pass: like bottom-up, but every replacement is
    /// normalised (bounded nesting) before the walk moves on.
    fn pass_innermost(&mut self, root: ExprId, nesting: usize) -> ExprId {
        let post_order = self.with_arena(|a| walk::post_order_ids(a, root));
        let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();
        for &id in &post_order {
            let rebuilt = self.with_arena_mut(|a| {
                if a.node(id).is_atom() {
                    id
                } else {
                    walk::rebuild_with_cache(a, id, &cache)
                }
            });
            let mut current = rebuilt;
            for _ in 0..MAX_NODE_REWRITES {
                let Some(new) = self.rewrite_at(current) else {
                    break;
                };
                if new == current {
                    break;
                }
                current = if nesting < MAX_INNERMOST_NESTING {
                    self.pass_innermost(new, nesting + 1)
                } else {
                    new
                };
            }
            cache.insert(id, current);
        }
        cache.get(&root).copied().unwrap_or(root)
    }

    /// One top-down pass (iterative, explicit stack).
    fn pass_top_down(&mut self, root: ExprId) -> ExprId {
        // original id → root-rewritten id
        let mut root_rw: FxHashMap<ExprId, ExprId> = FxHashMap::default();
        // root-rewritten id → fully processed id
        let mut done: FxHashMap<ExprId, ExprId> = FxHashMap::default();

        let r0 = self.rewrite_at_fixpoint(root);
        root_rw.insert(root, r0);
        let mut stack: Vec<(ExprId, bool)> = vec![(r0, false)];

        while let Some(&(rid, expanded)) = stack.last() {
            if done.contains_key(&rid) {
                stack.pop();
                continue;
            }
            if !expanded {
                if let Some(top) = stack.last_mut() {
                    top.1 = true;
                }
                let children = self.with_arena(|a| a.children(rid));
                for &c in children.iter().rev() {
                    let rc = match root_rw.get(&c) {
                        Some(&rc) => rc,
                        None => {
                            let rc = self.rewrite_at_fixpoint(c);
                            root_rw.insert(c, rc);
                            rc
                        }
                    };
                    if !done.contains_key(&rc) {
                        stack.push((rc, false));
                    }
                }
            } else {
                stack.pop();
                let children = self.with_arena(|a| a.children(rid));
                let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();
                for &c in &children {
                    let rc = root_rw.get(&c).copied().unwrap_or(c);
                    let fc = done.get(&rc).copied().unwrap_or(rc);
                    cache.insert(c, fc);
                }
                let rebuilt = self.with_arena_mut(|a| {
                    if a.node(rid).is_atom() {
                        rid
                    } else {
                        walk::rebuild_with_cache(a, rid, &cache)
                    }
                });
                done.insert(rid, rebuilt);
            }
        }
        done.get(&r0).copied().unwrap_or(r0)
    }

    fn run(&mut self, root: ExprId, opts: &RewriteOpts) -> ExprId {
        let mut current = root;
        for _ in 0..opts.max_iterations {
            let steps_before = self.steps.len();
            let next = match opts.strategy {
                RewriteStrategy::BottomUp => self.pass_bottom_up(current),
                RewriteStrategy::TopDown => self.pass_top_down(current),
                RewriteStrategy::Innermost => self.pass_innermost(current, 0),
            };
            if next == current {
                break;
            }
            let size = self.with_arena(|a| pattern::tree_size_capped(a, next, MAX_REWRITE_OPS + 1));
            if size > MAX_REWRITE_OPS {
                tracing::debug!(size, "rewrite: tree-size guard triggered, stopping");
                self.steps.truncate(steps_before);
                break;
            }
            current = next;
        }
        current
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// impl Expr<Numeric> — rewriting
// ═══════════════════════════════════════════════════════════════════════════

impl Expr<Numeric> {
    /// Rewrite with `rules` to a fixpoint (bottom-up, at most
    /// [`RewriteOpts::default()`]`.max_iterations` passes).
    ///
    /// Every rule is a value-preserving identity supplied by the caller;
    /// the engine itself never changes the value of an expression beyond
    /// what the rules state.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::macros::{Rule, RuleSet};
    ///
    /// let ctx = Context::new();
    /// let (x, y, a, b) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("a_"), ctx.symbol("b_"));
    /// // ln(a_) + ln(b_) → ln(a_ * b_)
    /// let rules = RuleSet::from_rules(vec![
    ///     Rule::new("ln_add", &(&a.ln() + &b.ln()), &(&a * &b).ln()),
    /// ]);
    /// let expr = &x.ln() + &y.ln() + &(&x + 1).ln();
    /// assert_eq!(format!("{}", expr.rewrite(&rules)), "ln(x*y*(x + 1))");
    /// ```
    #[must_use = "returns the rewritten form; does not modify in place"]
    pub fn rewrite(&self, rules: &RuleSet) -> Ex {
        self.rewrite_with(rules, &RewriteOpts::default())
    }

    /// A single bottom-up pass of `rules` (at most one rule application
    /// per node).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::macros::{Rule, RuleSet};
    ///
    /// let ctx = Context::new();
    /// let (x, a) = (ctx.symbol("x"), ctx.symbol("a_"));
    /// let rules = RuleSet::from_rules(vec![Rule::new("dbl", &a.sin(), &(&a.sin() * 2))]);
    /// // One pass: sin(x) → 2*sin(x); a second pass would give 4*sin(x).
    /// assert_eq!(format!("{}", x.sin().rewrite_once(&rules)), "2*sin(x)");
    /// ```
    #[must_use = "returns the rewritten form; does not modify in place"]
    pub fn rewrite_once(&self, rules: &RuleSet) -> Ex {
        self.rewrite_with(rules, &RewriteOpts::single_pass())
    }

    /// Like [`rewrite`](Self::rewrite), also returning every rule
    /// application as a [`Step`] in the order it happened.
    #[must_use = "returns the rewritten form and trace; does not modify in place"]
    pub fn rewrite_traced(&self, rules: &RuleSet) -> (Ex, Vec<Step>) {
        self.rewrite_with_traced(rules, &RewriteOpts::default())
    }

    /// Rewrite with explicit [`RewriteOpts`] (strategy and iteration cap).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::macros::{RewriteOpts, RewriteStrategy, Rule, RuleSet};
    ///
    /// let ctx = Context::new();
    /// let (x, a) = (ctx.symbol("x"), ctx.symbol("a_"));
    /// // abs(abs(a_)) style nesting: exp(exp(a_)) → exp(a_)
    /// let rules = RuleSet::from_rules(vec![Rule::new("flatten", &a.exp().exp(), &a.exp())]);
    /// let expr = x.exp().exp().exp().exp();
    /// let opts = RewriteOpts::default().strategy(RewriteStrategy::TopDown);
    /// assert_eq!(format!("{}", expr.rewrite_with(&rules, &opts)), "exp(x)");
    /// ```
    #[must_use = "returns the rewritten form; does not modify in place"]
    pub fn rewrite_with(&self, rules: &RuleSet, opts: &RewriteOpts) -> Ex {
        let mut driver = Driver {
            template: self,
            rules,
            steps: Vec::new(),
            trace: false,
        };
        let id = driver.run(self.raw_id(), opts);
        self.wrap(id)
    }

    /// [`rewrite_with`](Self::rewrite_with) plus the trace of steps.
    #[must_use = "returns the rewritten form and trace; does not modify in place"]
    pub fn rewrite_with_traced(&self, rules: &RuleSet, opts: &RewriteOpts) -> (Ex, Vec<Step>) {
        let mut driver = Driver {
            template: self,
            rules,
            steps: Vec::new(),
            trace: true,
        };
        let id = driver.run(self.raw_id(), opts);
        let steps = driver
            .steps
            .into_iter()
            .map(|s| Step::from_raw(self, s))
            .collect();
        (self.wrap(id), steps)
    }

    /// Standard [`simplify`](Self::simplify) interleaved with the user
    /// rules `extra`, iterated to a fixpoint (at most
    /// [`SimplifyOpts::default()`]`.max_iterations` rounds).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::macros::{Rule, RuleSet};
    ///
    /// let ctx = Context::new();
    /// let (x, a) = (ctx.symbol("x"), ctx.symbol("a_"));
    /// // A user-supplied identity: sinh(a_) → (exp(a_) - exp(-a_)) / 2
    /// let sinh_def = (&a.exp() - &(-&a).exp()) / 2;
    /// let extra = RuleSet::from_rules(vec![Rule::new("sinh_def", &a.sinh(), &sinh_def)]);
    /// let expr = &x.sinh() - &(&x.exp() - &(-&x).exp()) / 2;
    /// assert_eq!(format!("{}", expr.simplify_with_rules(&extra)), "0");
    /// ```
    #[must_use = "returns the simplified form; does not modify in place"]
    pub fn simplify_with_rules(&self, extra: &RuleSet) -> Ex {
        let max = SimplifyOpts::default().max_iterations.max(1);
        let mut current = self.clone();
        for _ in 0..max {
            let simplified = current.simplify();
            let rewritten = simplified.rewrite(extra);
            if rewritten == current {
                return rewritten;
            }
            current = rewritten;
        }
        current
    }

    /// [`simplify_with`](Self::simplify_with) that also returns the trace
    /// of strategies and rules that changed the expression.
    ///
    /// The `trace` flag of `opts` is switched on automatically.  Each
    /// fixpoint iteration contributes one `strategy:<name>` step (before /
    /// after the whole expression) followed by the rule-level steps that
    /// fired inside the winning strategy.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let expr = &x.sin().powi(2) + &x.cos().powi(2);
    /// let (result, steps) = expr.simplify_traced(&SimplifyOpts::default());
    /// assert_eq!(format!("{result}"), "1");
    /// assert!(steps.iter().any(|s| s.rule_name.starts_with("strategy:")));
    /// assert!(steps.iter().any(|s| s.rule_name == "pythagorean"));
    /// ```
    #[must_use = "returns the simplified form and trace; does not modify in place"]
    pub fn simplify_traced(&self, opts: &SimplifyOpts) -> (Ex, Vec<Step>) {
        let opts = opts.clone().trace();
        let result = {
            let mut inner = self.inner.write();
            crate::simplify::simplify_engine::unified_simplify(
                &mut inner.arena,
                self.raw_id(),
                &opts,
            )
        };
        let steps = result
            .steps
            .into_iter()
            .map(|s| Step::from_raw(self, s))
            .collect();
        (self.wrap(result.expr), steps)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// impl Expr<Numeric> — simplification gap-fill (0.2)
// ═══════════════════════════════════════════════════════════════════════════

impl Expr<Numeric> {
    /// Apply an arena-level transformation under the write lock.
    fn transform(&self, f: impl FnOnce(&mut Arena, ExprId) -> ExprId) -> Ex {
        let id = {
            let mut inner = self.inner.write();
            f(&mut inner.arena, self.raw_id())
        };
        self.wrap(id)
    }

    // ── Expansion ──────────────────────────────────────────────────

    /// Expansion with explicit hints — see [`ExpandOpts`].
    ///
    /// [`expand`](Self::expand) is `expand_with(&ExpandOpts::default())`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::macros::ExpandOpts;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// let expr = &(&x + &y).sin() * &(&x + 1);
    /// // Only distribute the product, leave sin(x + y) alone:
    /// let mul_only = expr.expand_with(&ExpandOpts::none().with_mul(true));
    /// assert_eq!(format!("{mul_only}"), "x*sin(x + y) + sin(x + y)");
    /// // Trig expansion too:
    /// let all = expr.expand_with(&ExpandOpts::default().trig(true));
    /// assert!(format!("{all}").contains("cos(y)"), "{all}");
    /// ```
    #[must_use = "returns the expanded form; does not modify in place"]
    pub fn expand_with(&self, opts: &ExpandOpts) -> Ex {
        let opts = *opts;
        self.transform(move |a, id| crate::transforms::expand::expand_with(a, id, &opts))
    }

    /// Distribute powers over products: `(x·y)^e → x^e·y^e`.
    ///
    /// Only fires when the identity is guaranteed — integer `e`, or all
    /// factors known non-negative — unless `force` is set.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y, a) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("a"));
    /// let expr = (&x * &y).pow(&a);
    /// assert_eq!(format!("{}", expr.expand_power_base(false)), "(x*y)^a");
    /// assert_eq!(format!("{}", expr.expand_power_base(true)), "x^a*y^a");
    /// let p = ctx.symbol_with("p", &[Assumption::Positive]);
    /// let q = ctx.symbol_with("q", &[Assumption::Positive]);
    /// assert_eq!(format!("{}", (&p * &q).pow(&a).expand_power_base(false)), "p^a*q^a");
    /// ```
    #[must_use = "returns the expanded form; does not modify in place"]
    pub fn expand_power_base(&self, force: bool) -> Ex {
        let opts = ExpandOpts::none().power_base(true).force(force);
        self.expand_with(&opts)
    }

    /// Split sums in exponents: `x^(a+b) → x^a·x^b`, `exp(a+b) → exp(a)·exp(b)`.
    ///
    /// Guarded (base `e`, positive base, same-sign numeric or integer
    /// summands) unless `force` is set.  Note that the canonical form
    /// merges equal bases in a product, so `x^a·x^b` immediately folds
    /// back to `x^(a+b)`; the split is only observable for `exp`, whose
    /// factors are kept apart (they are recombined by
    /// [`simplify_powers`](Self::simplify_powers) / `exp_mul`).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, a, b) = (ctx.symbol("x"), ctx.symbol("a"), ctx.symbol("b"));
    /// assert_eq!(format!("{}", (&a + &b).exp().expand_power_exp(false)), "exp(a)*exp(b)");
    /// // x^a * x^b is re-merged by canonicalisation:
    /// assert_eq!(format!("{}", x.pow(&(&a + &b)).expand_power_exp(true)), "x^(a + b)");
    /// ```
    #[must_use = "returns the expanded form; does not modify in place"]
    pub fn expand_power_exp(&self, force: bool) -> Ex {
        let opts = ExpandOpts::none().power_exp(true).force(force);
        self.expand_with(&opts)
    }

    /// Expand non-negative integer powers of sums only (multinomial
    /// theorem); products are *not* distributed.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// let expr = (&x + &y).powi(3);
    /// assert_eq!(expr.expand_multinomial(), expr.expand());
    /// assert_eq!(expr.expand_multinomial().term_count(), 4);
    /// let prod = &x * &(&x + 1);
    /// assert_eq!(format!("{}", prod.expand_multinomial()), "x*(x + 1)");
    /// ```
    #[must_use = "returns the expanded form; does not modify in place"]
    pub fn expand_multinomial(&self) -> Ex {
        self.expand_with(&ExpandOpts::none().multinomial(true))
    }

    /// Expand logarithms, honouring the positivity guard unless `force`
    /// is set ([`expand_log`](Self::expand_log) is the forced form).
    ///
    /// `ln(a·b) → ln a + ln b` and `ln(a^n) → n·ln a` are exact for
    /// positive real `a`, `b` (and real `n`); for other arguments they can
    /// be off by a multiple of `2πi`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// let expr = (&x * &y).ln();
    /// assert_eq!(format!("{}", expr.expand_log_with(false)), "ln(x*y)");
    /// assert_eq!(format!("{}", expr.expand_log_with(true)), "ln(x) + ln(y)");
    /// let p = ctx.symbol_with("p", &[Assumption::Positive]);
    /// let q = ctx.symbol_with("q", &[Assumption::Positive]);
    /// assert_eq!(format!("{}", (&p * &q).ln().expand_log_with(false)), "ln(p) + ln(q)");
    /// ```
    #[must_use = "returns the expanded form; does not modify in place"]
    pub fn expand_log_with(&self, force: bool) -> Ex {
        self.transform(move |a, id| crate::simplify::log_expand::expand_log_with(a, id, force))
    }

    /// Combine logarithms, honouring the positivity guard unless `force`
    /// is set ([`log_combine`](Self::log_combine) is the forced form).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// let expr = &x.ln() + &y.ln();
    /// assert_eq!(format!("{}", expr.log_combine_with(false)), "ln(x) + ln(y)");
    /// assert_eq!(format!("{}", expr.log_combine_with(true)), "ln(x*y)");
    /// let p = ctx.symbol_with("p", &[Assumption::Positive]);
    /// assert_eq!(format!("{}", (&p.ln() * 2).log_combine_with(false)), "ln(p^2)");
    /// ```
    #[must_use = "returns the combined form; does not modify in place"]
    pub fn log_combine_with(&self, force: bool) -> Ex {
        self.transform(move |a, id| crate::simplify::log_combine::log_combine_with(a, id, force))
    }

    // ── Radicals, powers, signs ────────────────────────────────────

    /// Denest square roots of the form `√(a + b√c)` with rational `a`,
    /// `b`, `c` whenever `a² − b²c` is a perfect square.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let sqrt2 = ctx.int(2).sqrt();
    /// let e = (&sqrt2 * 2 + 3).sqrt();          // √(3 + 2√2)
    /// assert_eq!(format!("{}", e.sqrtdenest()), "sqrt(2) + 1");
    /// let f = (5 - &ctx.int(6).sqrt() * 2).sqrt(); // √(5 − 2√6)
    /// assert_eq!(format!("{}", f.sqrtdenest()), "sqrt(3) - sqrt(2)");
    /// ```
    #[must_use = "returns the denested form; does not modify in place"]
    pub fn sqrtdenest(&self) -> Ex {
        self.transform(crate::simplify::radsimp::sqrtdenest)
    }

    /// Sign normalisation: extract `−1` from sums with a negative leading
    /// term when they occur inside products or integer powers.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    /// assert_eq!(format!("{}", (&y - &x).powi(2).signsimp()), "(x - y)^2");
    /// assert_eq!(format!("{}", (&y - &x).powi(3).signsimp()), "-(x - y)^3");
    /// assert_eq!(format!("{}", (&z * &(&y - &x)).signsimp()), "-z*(x - y)");
    /// ```
    #[must_use = "returns the normalised form; does not modify in place"]
    pub fn signsimp(&self) -> Ex {
        self.transform(crate::simplify::factor_terms::signsimp)
    }

    /// Denest powers, each rewrite only when it is an identity:
    ///
    /// | Rewrite                    | Condition (any of)                                   |
    /// |----------------------------|------------------------------------------------------|
    /// | `(a·b)^e → a^e·b^e`        | `e ∈ ℤ`; all factors known non-negative; `force`    |
    /// | `(x^a)^b → x^(a·b)`        | `b ∈ ℤ`; `x > 0` and `a` real; `force`               |
    /// | `√(x²) → x`                | `x ≥ 0`; `force`                                     |
    /// | `√(x²) → ∣x∣`              | `x` real (not known non-real)                        |
    ///
    /// `(x^a)^b = exp(b·Log(exp(a·Log x)))` equals `x^(ab)` exactly when
    /// `Im(a·Log x) ∈ (−π, π]` — guaranteed for `x > 0` and real `a` — or
    /// when `b` is an integer.  `√(x²) = |x|` needs `x` real (`√(i²) = i`).
    /// With `force = true` every symbol is treated as positive (like SymPy's
    /// `powdenest(force=True)`).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, a, b) = (ctx.symbol("x"), ctx.symbol("a"), ctx.symbol("b"));
    /// let nested = x.pow(&a).pow(&b);
    /// assert_eq!(format!("{}", nested.powdenest(false)), "(x^a)^b");
    /// assert_eq!(format!("{}", nested.powdenest(true)), "x^(a*b)");
    /// assert_eq!(format!("{}", x.pow(&a).powi(3).powdenest(false)), "x^(3*a)");
    /// assert_eq!(format!("{}", x.powi(2).sqrt().powdenest(false)), "abs(x)");
    /// let p = ctx.symbol_with("p", &[Assumption::Positive]);
    /// assert_eq!(format!("{}", p.powi(2).sqrt().powdenest(false)), "p");
    /// ```
    #[must_use = "returns the denested form; does not modify in place"]
    pub fn powdenest(&self, force: bool) -> Ex {
        self.transform(move |a, id| crate::simplify::powsimp::powdenest_with(a, id, force))
    }

    // ── Collecting ─────────────────────────────────────────────────

    /// Recursively collect every sum by the given variables, in order.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    /// let expr = &x * &y + &x * &z + &y.powi(2) * &x + &y.powi(2) * &z;
    /// let c = expr.rcollect(&[&x, &y]);
    /// assert_eq!(format!("{c}"), "z*y^2 + x*(y^2 + y + z)");
    /// ```
    #[must_use = "returns the collected form; does not modify in place"]
    pub fn rcollect(&self, vars: &[&Ex]) -> Ex {
        let ids: Vec<ExprId> = vars.iter().map(|v| self.checked_id(v)).collect();
        self.transform(move |a, id| crate::simplify::factor_terms::rcollect(a, id, &ids))
    }

    /// Factor the rational content out of sums inside products and powers:
    /// `z·(2x + 4y) → 2·z·(x + 2y)`.
    ///
    /// A *top-level* sum is returned unchanged, because the canonical form
    /// distributes a numeric coefficient over a sum (`2·(x + 2y)` is not
    /// representable); use [`factor_terms`](Self::factor_terms) to obtain
    /// the `(2, x + 2y)` pair.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    /// let expr = &z * &(&x * 2 + &y * 4);
    /// assert_eq!(format!("{}", expr.collect_const()), "2*z*(x + 2*y)");
    /// let sq = (&x * 2 + &y * 4).powi(2);
    /// assert_eq!(format!("{}", sq.collect_const()), "4*(x + 2*y)^2");
    /// ```
    #[must_use = "returns the collected form; does not modify in place"]
    pub fn collect_const(&self) -> Ex {
        self.transform(crate::simplify::factor_terms::collect_const)
    }

    // ── Numeric recognition ────────────────────────────────────────

    /// Recognise a numerical expression as `q·c`, `q + c` or `c^(p/q)` for
    /// one of the given constants `c` (or as a plain rational), within
    /// `tolerance`.  Falls back to [`simplify_numeric`](Self::simplify_numeric).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let pi = ctx.pi();
    /// let e = ctx.e();
    /// let v = ctx.rational(628318530717958, 100000000000000); // ≈ 2π
    /// assert_eq!(format!("{}", v.nsimplify_with_constants(&[&pi, &e], 1e-9)), "2*pi");
    /// let w = ctx.rational(3718281828459045, 1000000000000000); // ≈ 1 + e
    /// assert_eq!(format!("{}", w.nsimplify_with_constants(&[&pi, &e], 1e-9)), "1 + E");
    /// ```
    #[must_use = "returns the recognised form; does not modify in place"]
    pub fn nsimplify_with_constants(&self, constants: &[&Ex], tolerance: f64) -> Ex {
        let ids: Vec<ExprId> = constants.iter().map(|c| self.checked_id(c)).collect();
        self.transform(move |a, id| {
            crate::simplify::nsimplify::nsimplify_with_constants(a, id, &ids, tolerance)
        })
    }

    /// [`nsimplify_with_constants`](Self::nsimplify_with_constants) with the
    /// default table `π, e, √2, √3, √5, ln 2, φ, γ`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let v = ctx.rational(1414213562373095, 1000000000000000); // ≈ √2
    /// assert_eq!(format!("{}", v.nsimplify(1e-9)), "sqrt(2)");
    /// let g = ctx.rational(5772156649015329, 10000000000000000); // ≈ γ
    /// assert_eq!(format!("{}", g.nsimplify(1e-9)), "EulerGamma");
    /// ```
    #[must_use = "returns the recognised form; does not modify in place"]
    pub fn nsimplify(&self, tolerance: f64) -> Ex {
        self.transform(move |a, id| {
            let consts = crate::simplify::nsimplify::default_constants(a);
            crate::simplify::nsimplify::nsimplify_with_constants(a, id, &consts, tolerance)
        })
    }

    // ── Variable separation ────────────────────────────────────────

    /// Additive separation: partition the terms of a sum by the variables
    /// they depend on (terms with the same dependency set are summed).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// let expr = &x.sin() + &y.powi(2) + &x * &y + 3;
    /// let groups = expr.separate_vars_additive(&[&x, &y]);
    /// assert_eq!(groups.len(), 4);
    /// let mixed = groups.iter().find(|(deps, _)| deps.len() == 2).unwrap();
    /// assert_eq!(format!("{}", mixed.1), "x*y");
    /// ```
    #[must_use]
    pub fn separate_vars_additive(&self, vars: &[&Ex]) -> Vec<(Vec<Ex>, Ex)> {
        let ids: Vec<ExprId> = vars.iter().map(|v| self.checked_id(v)).collect();
        let raw = {
            let mut inner = self.inner.write();
            crate::domains::separatevars::separatevars_additive(
                &mut inner.arena,
                self.raw_id(),
                &ids,
            )
        };
        raw.into_iter()
            .map(|(deps, sum)| {
                (
                    deps.into_iter().map(|d| self.wrap(d)).collect(),
                    self.wrap(sum),
                )
            })
            .collect()
    }

    /// Multiplicative separation into one factor per variable.
    ///
    /// Returns `Some(pairs)` with one `(var, factor)` entry per variable in
    /// `vars` (factor `1` when the expression does not depend on it) such
    /// that the product of all factors equals `self`; a factor free of
    /// every variable is folded into the first variable's entry.  Returns
    /// `None` when some factor depends on two or more variables.  Sums are
    /// first written as `content·(…)` so that `x·y + x·z` separates.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// let expr = &x.sin() * &y.powi(2) * 2;
    /// let d = expr.separate_vars_dict(&[&x, &y]).unwrap();
    /// assert_eq!(format!("{}", d[0].1), "2*sin(x)");
    /// assert_eq!(format!("{}", d[1].1), "y^2");
    /// assert!((&x + &y).separate_vars_dict(&[&x, &y]).is_none());
    /// // x*y + x*z = x*(y + z)
    /// let z = ctx.symbol("z");
    /// let e = &x * &y + &x * &z;
    /// assert!(e.separate_vars_dict(&[&x, &y, &z]).is_none(), "y + z mixes y and z");
    /// let d = e.separate_vars_dict(&[&x, &y]).unwrap(); // z is a parameter here
    /// assert_eq!(format!("{}", d[0].1), "x");
    /// assert_eq!(format!("{}", d[1].1), "y + z");
    /// ```
    #[must_use]
    pub fn separate_vars_dict(&self, vars: &[&Ex]) -> Option<Vec<(Ex, Ex)>> {
        let ids: Vec<ExprId> = vars.iter().map(|v| self.checked_id(v)).collect();
        let factors = {
            let mut inner = self.inner.write();
            crate::domains::separatevars::separatevars_dict(&mut inner.arena, self.raw_id(), &ids)?
        };
        Some(
            vars.iter()
                .zip(factors)
                .map(|(v, f)| ((*v).clone(), self.wrap(f)))
                .collect(),
        )
    }

    // ── Algebraic substitution ─────────────────────────────────────

    /// Algebraic substitution `old → new`: unlike [`subs`](Self::subs),
    /// `old` is recognised inside powers, products and sums.
    ///
    /// | `self`        | `old`     | result      |
    /// |---------------|-----------|-------------|
    /// | `x^4`         | `x^2`     | `y^2`       |
    /// | `x^3`         | `x^2`     | `x·y`       |
    /// | `1/x^2`       | `x^2`     | `1/y`       |
    /// | `2·x·y·z`     | `x·y`     | `2·w·z`     |
    /// | `a + b + c`   | `a + b`   | `c + d`     |
    /// | `exp(2x)`     | `exp(x)`  | `t^2`       |
    ///
    /// Every rewrite is an identity in `old`, so substituting `old` back
    /// for `new` recovers the value of `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// assert_eq!(format!("{}", x.powi(4).subs_algebraic(&x.powi(2), &y)), "y^2");
    /// assert_eq!(format!("{}", x.powi(3).subs_algebraic(&x.powi(2), &y)), "x*y");
    /// assert_eq!(format!("{}", x.powi(-2).subs_algebraic(&x.powi(2), &y)), "1/y");
    /// ```
    #[must_use = "returns the substituted form; does not modify in place"]
    pub fn subs_algebraic(&self, old: &Ex, new: &Ex) -> Ex {
        let old_id = self.checked_id(old);
        let new_id = self.checked_id(new);
        self.transform(move |a, id| pattern::subs_algebraic(a, id, old_id, new_id))
    }
}

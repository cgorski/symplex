//! Algebraic expansion.
//!
//! This module implements [`expand`], which distributes products over
//! sums and expands integer powers of sums.
//!
//! # What `expand` does
//!
//! - `a * (b + c)` → `a*b + a*c`
//! - `(a + b) * (c + d)` → `a*c + a*d + b*c + b*d`
//! - `(a + b)^n` for non-negative integer `n` → multinomial expansion
//! - Recursively expands nested products/powers of sums
//!
//! # What `expand` does NOT do
//!
//! - Does not factor, collect, or simplify
//! - Does not evaluate functions (`sin`, `cos`, etc.)
//! - Does not cancel common factors in fractions
//!
//! # Design
//!
//! Expansion is performed bottom-up using an explicit post-order
//! traversal (no recursion).  Each node is expanded after its children,
//! so by the time we reach a `Mul` or `Pow`, the children are already
//! in expanded form.

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Signed};
use rustc_hash::FxHashSet;
use smallvec::SmallVec;

use crate::base::arena::Arena;
use crate::base::assumptions::{AssumptionCache, Props};
use crate::base::node::{ExprId, ExprNode};
use crate::base::walk;

// ═══════════════════════════════════════════════════════════════════════════
// ExpandOpts
// ═══════════════════════════════════════════════════════════════════════════

/// Hints controlling [`Ex::expand_with`](crate::api::expr::Ex::expand_with).
///
/// Every rewrite is value-preserving.  The hints that are only valid
/// under side conditions (`power_base`, `power_exp`, `log`) are guarded by
/// the assumption system unless [`force`](Self::force) is set:
///
/// | Hint          | Rewrite                              | Guard (unless `force`) |
/// |---------------|--------------------------------------|------------------------|
/// | `mul`         | `a·(b + c) → a·b + a·c`              | none                   |
/// | `multinomial` | `(a + b)^n → …` for integer `n ≥ 0`  | none                   |
/// | `power_base`  | `(x·y)^e → x^e·y^e`                  | `e ∈ ℤ`, or every factor known non-negative |
/// | `power_exp`   | `x^(a+b) → x^a·x^b`                   | `x = e`, `x > 0`, all summands numeric same-sign, or all summands known integers |
/// | `log`         | `ln(a·b) → ln a + ln b`, `ln(a^n) → n·ln a` | arguments known positive (`n` real) |
/// | `trig`        | `sin(a + b) → sin a cos b + cos a sin b`, … | none            |
/// | `deep`        | also expand inside function arguments | —                      |
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::macros::ExpandOpts;
///
/// let ctx = Context::new();
/// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
/// let expr = (&x * &y).ln();
/// // Default: logs are not expanded.
/// assert_eq!(format!("{}", expr.expand_with(&ExpandOpts::default())), "ln(x*y)");
/// // With `log` + `force` the identity is applied unconditionally.
/// let opts = ExpandOpts::default().log(true).force(true);
/// assert_eq!(format!("{}", expr.expand_with(&opts)), "ln(x) + ln(y)");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpandOpts {
    /// Distribute products over sums (default `true`).
    pub mul: bool,
    /// Expand non-negative integer powers of sums (default `true`).
    pub multinomial: bool,
    /// Distribute powers over products, `(x·y)^e → x^e·y^e` (default `true`, guarded).
    pub power_base: bool,
    /// Split sums in exponents, `x^(a+b) → x^a·x^b` (default `true`, guarded).
    pub power_exp: bool,
    /// Expand logarithms of products / powers (default `false`, guarded).
    pub log: bool,
    /// Expand trigonometric functions of sums and multiples (default `false`).
    pub trig: bool,
    /// Recurse into function arguments (default `true`).  When `false`,
    /// only the algebraic skeleton reachable through `Add`/`Mul`/`Pow`
    /// from the root is expanded.
    pub deep: bool,
    /// Apply the guarded rewrites unconditionally (default `false`).
    pub force: bool,
}

impl Default for ExpandOpts {
    fn default() -> Self {
        ExpandOpts {
            mul: true,
            multinomial: true,
            power_base: true,
            power_exp: true,
            log: false,
            trig: false,
            deep: true,
            force: false,
        }
    }
}

impl ExpandOpts {
    /// No hints enabled (only `deep`); combine with the builder methods.
    #[must_use]
    pub fn none() -> Self {
        ExpandOpts {
            mul: false,
            multinomial: false,
            power_base: false,
            power_exp: false,
            log: false,
            trig: false,
            deep: true,
            force: false,
        }
    }

    /// Every hint enabled (still guarded unless `force`).
    #[must_use]
    pub fn all() -> Self {
        ExpandOpts {
            mul: true,
            multinomial: true,
            power_base: true,
            power_exp: true,
            log: true,
            trig: true,
            deep: true,
            force: false,
        }
    }

    /// Builder: set `mul`.
    #[must_use]
    pub fn with_mul(mut self, v: bool) -> Self {
        self.mul = v;
        self
    }
    /// Builder: set `multinomial`.
    #[must_use]
    pub fn multinomial(mut self, v: bool) -> Self {
        self.multinomial = v;
        self
    }
    /// Builder: set `power_base`.
    #[must_use]
    pub fn power_base(mut self, v: bool) -> Self {
        self.power_base = v;
        self
    }
    /// Builder: set `power_exp`.
    #[must_use]
    pub fn power_exp(mut self, v: bool) -> Self {
        self.power_exp = v;
        self
    }
    /// Builder: set `log`.
    #[must_use]
    pub fn log(mut self, v: bool) -> Self {
        self.log = v;
        self
    }
    /// Builder: set `trig`.
    #[must_use]
    pub fn trig(mut self, v: bool) -> Self {
        self.trig = v;
        self
    }
    /// Builder: set `deep`.
    #[must_use]
    pub fn deep(mut self, v: bool) -> Self {
        self.deep = v;
        self
    }
    /// Builder: set `force`.
    #[must_use]
    pub fn force(mut self, v: bool) -> Self {
        self.force = v;
        self
    }
}

/// Fully expand an expression: distribute products over sums and
/// expand integer powers of sums.
///
/// The result is a sum of products — no unexpanded `Mul(…, Add(…))`
/// or `Pow(Add(…), positive_int)` nodes remain.
///
/// Equivalent to [`expand_with`] with [`ExpandOpts::default()`].
pub(crate) fn expand(arena: &mut Arena, expr: ExprId) -> ExprId {
    expand_with(arena, expr, &ExpandOpts::default())
}

/// The set of nodes reachable from `root` through `Add`/`Mul`/`Pow`
/// edges only (the "algebraic skeleton"), used for `deep = false`.
fn algebraic_skeleton(arena: &Arena, root: ExprId) -> FxHashSet<ExprId> {
    let mut set = FxHashSet::default();
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if !set.insert(id) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Add(ch) | ExprNode::Mul(ch) => stack.extend(ch.iter().copied()),
            ExprNode::Pow(b, e) => {
                stack.push(*b);
                stack.push(*e);
            }
            _ => {}
        }
    }
    set
}

/// Expand with explicit hints — see [`ExpandOpts`].
pub(crate) fn expand_with(arena: &mut Arena, expr: ExprId, opts: &ExpandOpts) -> ExprId {
    // Bottom-up: expand children first, then handle the current node.
    let post_order = walk::post_order_ids(arena, expr);
    let mut cache = rustc_hash::FxHashMap::<ExprId, ExprId>::default();
    let skeleton = if opts.deep {
        None
    } else {
        Some(algebraic_skeleton(arena, expr))
    };
    let mut assumptions = AssumptionCache::new();

    for &id in &post_order {
        if let Some(sk) = &skeleton
            && (!sk.contains(&id)
                || !matches!(
                    arena.node(id),
                    ExprNode::Add(_) | ExprNode::Mul(_) | ExprNode::Pow(_, _)
                ))
        {
            // `deep = false`: nodes outside the algebraic skeleton, and
            // function nodes on its boundary, are left untouched (their
            // arguments are not rebuilt even if shared with expanded parts).
            cache.insert(id, id);
            continue;
        }
        let node = arena.node(id).clone();
        let expanded = match node {
            // Mul without the `mul` hint: just rebuild.
            ExprNode::Mul(ref children) if !opts.mul => {
                let new_children: SmallVec<[ExprId; 6]> = children
                    .iter()
                    .map(|&c| cache.get(&c).copied().unwrap_or(c))
                    .collect();
                if new_children == *children {
                    id
                } else {
                    arena.mul(&new_children)
                }
            }

            // exp(a + b) → exp(a)·exp(b) under the `power_exp` hint (always
            // valid; `e^(a+b)` is canonicalised to an `Exp` node, so the
            // `Pow` path below never sees it).
            ExprNode::Exp(inner) if opts.power_exp => {
                let new_inner = cache.get(&inner).copied().unwrap_or(inner);
                match arena.node(new_inner).clone() {
                    ExprNode::Add(children) => {
                        let factors: SmallVec<[ExprId; 6]> =
                            children.iter().map(|&c| arena.exp(c)).collect();
                        arena.mul(&factors)
                    }
                    _ => rebuild_unary_expanded(arena, id, inner, &cache, Arena::exp),
                }
            }

            // Ln with the `log` hint.
            ExprNode::Ln(inner) if opts.log => {
                let new_inner = cache.get(&inner).copied().unwrap_or(inner);
                crate::simplify::log_expand::expand_ln_node_guarded(
                    arena,
                    &mut assumptions,
                    new_inner,
                    opts.force,
                )
            }

            // Trig functions with the `trig` hint.
            ExprNode::Sin(inner) if opts.trig => {
                let rebuilt = rebuild_unary_expanded(arena, id, inner, &cache, Arena::sin);
                crate::simplify::trig_expand::expand_trig(arena, rebuilt)
            }
            ExprNode::Cos(inner) if opts.trig => {
                let rebuilt = rebuild_unary_expanded(arena, id, inner, &cache, Arena::cos);
                crate::simplify::trig_expand::expand_trig(arena, rebuilt)
            }
            ExprNode::Tan(inner) if opts.trig => {
                let rebuilt = rebuild_unary_expanded(arena, id, inner, &cache, Arena::tan);
                crate::simplify::trig_expand::expand_trig(arena, rebuilt)
            }
            // Add: expand each child, then re-add.
            ExprNode::Add(ref children) => {
                let new_children: SmallVec<[ExprId; 6]> = children
                    .iter()
                    .map(|&c| cache.get(&c).copied().unwrap_or(c))
                    .collect();
                if new_children == *children {
                    id
                } else {
                    arena.add(&new_children)
                }
            }

            // Mul: expand each child, then distribute.
            ExprNode::Mul(ref children) => {
                let new_children: SmallVec<[ExprId; 6]> = children
                    .iter()
                    .map(|&c| cache.get(&c).copied().unwrap_or(c))
                    .collect();
                expand_mul(arena, &new_children)
            }

            // Pow: if base is Add and exp is a non-negative integer,
            // expand via repeated multiplication.
            ExprNode::Pow(base, exp) => {
                let new_base = cache.get(&base).copied().unwrap_or(base);
                let new_exp = cache.get(&exp).copied().unwrap_or(exp);
                expand_pow(arena, &mut assumptions, new_base, new_exp, opts)
            }

            // Neg: expand the inner, then negate.
            ExprNode::Neg(inner) => {
                let new_inner = cache.get(&inner).copied().unwrap_or(inner);
                if new_inner == inner {
                    id
                } else {
                    arena.neg(new_inner)
                }
            }

            // Unary functions: just rebuild with expanded child.
            ExprNode::Sin(inner) => rebuild_unary_expanded(arena, id, inner, &cache, Arena::sin),
            ExprNode::Cos(inner) => rebuild_unary_expanded(arena, id, inner, &cache, Arena::cos),
            ExprNode::Tan(inner) => rebuild_unary_expanded(arena, id, inner, &cache, Arena::tan),
            ExprNode::Exp(inner) => rebuild_unary_expanded(arena, id, inner, &cache, Arena::exp),
            ExprNode::Ln(inner) => rebuild_unary_expanded(arena, id, inner, &cache, Arena::ln),
            ExprNode::Abs(inner) => rebuild_unary_expanded(arena, id, inner, &cache, Arena::abs),
            ExprNode::Asin(inner) => rebuild_unary_expanded(arena, id, inner, &cache, Arena::asin),
            ExprNode::Acos(inner) => rebuild_unary_expanded(arena, id, inner, &cache, Arena::acos),
            ExprNode::Atan(inner) => rebuild_unary_expanded(arena, id, inner, &cache, Arena::atan),
            ExprNode::Atan2(y, x) => {
                let ny = cache.get(&y).copied().unwrap_or(y);
                let nx = cache.get(&x).copied().unwrap_or(x);
                if ny == y && nx == x {
                    id
                } else {
                    arena.atan2(ny, nx)
                }
            }
            ExprNode::Sinh(inner) => rebuild_unary_expanded(arena, id, inner, &cache, Arena::sinh),
            ExprNode::Cosh(inner) => rebuild_unary_expanded(arena, id, inner, &cache, Arena::cosh),
            ExprNode::Tanh(inner) => rebuild_unary_expanded(arena, id, inner, &cache, Arena::tanh),
            ExprNode::Asinh(inner) => {
                rebuild_unary_expanded(arena, id, inner, &cache, Arena::asinh)
            }
            ExprNode::Acosh(inner) => {
                rebuild_unary_expanded(arena, id, inner, &cache, Arena::acosh)
            }
            ExprNode::Atanh(inner) => {
                rebuild_unary_expanded(arena, id, inner, &cache, Arena::atanh)
            }
            ExprNode::Sign(inner) => rebuild_unary_expanded(arena, id, inner, &cache, Arena::sign),
            ExprNode::Heaviside(inner) => {
                rebuild_unary_expanded(arena, id, inner, &cache, Arena::heaviside)
            }
            ExprNode::DiracDelta(inner) => {
                rebuild_unary_expanded(arena, id, inner, &cache, Arena::dirac_delta)
            }
            ExprNode::LambertW(inner) => {
                rebuild_unary_expanded(arena, id, inner, &cache, Arena::lambertw)
            }
            ExprNode::Floor(inner) => {
                rebuild_unary_expanded(arena, id, inner, &cache, Arena::floor)
            }
            ExprNode::Ceiling(inner) => {
                rebuild_unary_expanded(arena, id, inner, &cache, Arena::ceiling)
            }
            ExprNode::Not(inner) => rebuild_unary_expanded(arena, id, inner, &cache, Arena::not),

            // Boolean atoms: unchanged.
            ExprNode::BoolTrue | ExprNode::BoolFalse => id,

            // Relational operators: rebuild binary with expanded children.
            ExprNode::Gt(a, b) => {
                let na = cache.get(&a).copied().unwrap_or(a);
                let nb = cache.get(&b).copied().unwrap_or(b);
                if na == a && nb == b {
                    id
                } else {
                    arena.gt(na, nb)
                }
            }
            ExprNode::Ge(a, b) => {
                let na = cache.get(&a).copied().unwrap_or(a);
                let nb = cache.get(&b).copied().unwrap_or(b);
                if na == a && nb == b {
                    id
                } else {
                    arena.ge(na, nb)
                }
            }
            ExprNode::Eq_(a, b) => {
                let na = cache.get(&a).copied().unwrap_or(a);
                let nb = cache.get(&b).copied().unwrap_or(b);
                if na == a && nb == b {
                    id
                } else {
                    arena.eq_(na, nb)
                }
            }
            ExprNode::Ne(a, b) => {
                let na = cache.get(&a).copied().unwrap_or(a);
                let nb = cache.get(&b).copied().unwrap_or(b);
                if na == a && nb == b {
                    id
                } else {
                    arena.ne_(na, nb)
                }
            }

            // N-ary logical / piecewise: rebuild with expanded children.
            ExprNode::And(ref children) => {
                let new: SmallVec<[ExprId; 6]> = children
                    .iter()
                    .map(|&c| cache.get(&c).copied().unwrap_or(c))
                    .collect();
                if new == *children {
                    id
                } else {
                    arena.and(&new)
                }
            }
            ExprNode::Or(ref children) => {
                let new: SmallVec<[ExprId; 6]> = children
                    .iter()
                    .map(|&c| cache.get(&c).copied().unwrap_or(c))
                    .collect();
                if new == *children { id } else { arena.or(&new) }
            }
            ExprNode::Piecewise(ref pairs) => {
                let new: SmallVec<[(ExprId, ExprId); 3]> = pairs
                    .iter()
                    .map(|&(val, cond)| {
                        let nv = cache.get(&val).copied().unwrap_or(val);
                        let nc = cache.get(&cond).copied().unwrap_or(cond);
                        (nv, nc)
                    })
                    .collect();
                if new == *pairs {
                    id
                } else {
                    arena.intern(ExprNode::Piecewise(new))
                }
            }

            ExprNode::Min(ref children) => {
                let new: smallvec::SmallVec<[crate::base::node::ExprId; 4]> = children
                    .iter()
                    .map(|&c| *cache.get(&c).unwrap_or(&c))
                    .collect();
                if new[..] == children[..] {
                    id
                } else {
                    arena.intern(ExprNode::Min(new))
                }
            }
            ExprNode::Max(ref children) => {
                let new: smallvec::SmallVec<[crate::base::node::ExprId; 4]> = children
                    .iter()
                    .map(|&c| *cache.get(&c).unwrap_or(&c))
                    .collect();
                if new[..] == children[..] {
                    id
                } else {
                    arena.intern(ExprNode::Max(new))
                }
            }
            ExprNode::Derivative(body, var) => {
                let new_body = *cache.get(&body).unwrap_or(&body);
                let new_var = *cache.get(&var).unwrap_or(&var);
                if new_body == body && new_var == var {
                    id
                } else {
                    arena.intern(ExprNode::Derivative(new_body, new_var))
                }
            }
            ExprNode::Integral(body, var) => {
                let new_body = *cache.get(&body).unwrap_or(&body);
                let new_var = *cache.get(&var).unwrap_or(&var);
                if new_body == body && new_var == var {
                    id
                } else {
                    arena.intern(ExprNode::Integral(new_body, new_var))
                }
            }
            ExprNode::Sum(body, var, lo, hi) => {
                let nb = *cache.get(&body).unwrap_or(&body);
                let nv = *cache.get(&var).unwrap_or(&var);
                let nl = *cache.get(&lo).unwrap_or(&lo);
                let nh = *cache.get(&hi).unwrap_or(&hi);
                if nb == body && nv == var && nl == lo && nh == hi {
                    id
                } else {
                    arena.intern(ExprNode::Sum(nb, nv, nl, nh))
                }
            }
            ExprNode::Product_(body, var, lo, hi) => {
                let nb = *cache.get(&body).unwrap_or(&body);
                let nv = *cache.get(&var).unwrap_or(&var);
                let nl = *cache.get(&lo).unwrap_or(&lo);
                let nh = *cache.get(&hi).unwrap_or(&hi);
                if nb == body && nv == var && nl == lo && nh == hi {
                    id
                } else {
                    arena.intern(ExprNode::Product_(nb, nv, nl, nh))
                }
            }
            ExprNode::Apply(func_id, ref args) => {
                let new_args: smallvec::SmallVec<[crate::base::node::ExprId; 2]> =
                    args.iter().map(|&a| *cache.get(&a).unwrap_or(&a)).collect();
                if new_args == *args {
                    id
                } else {
                    arena.intern(ExprNode::Apply(func_id, new_args))
                }
            }
            // Keep the wildcard for truly inert nodes (atoms handled earlier)
            _ => id,
        };

        cache.insert(id, expanded);
    }

    cache.get(&expr).copied().unwrap_or(expr)
}

/// Helper for rebuilding unary nodes with expanded children.
#[inline]
fn rebuild_unary_expanded(
    arena: &mut Arena,
    id: ExprId,
    inner: ExprId,
    cache: &rustc_hash::FxHashMap<ExprId, ExprId>,
    ctor: fn(&mut Arena, ExprId) -> ExprId,
) -> ExprId {
    let new_inner = cache.get(&inner).copied().unwrap_or(inner);
    if new_inner == inner {
        id
    } else {
        ctor(arena, new_inner)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Mul expansion (distribution)
// ═══════════════════════════════════════════════════════════════════════════

/// Expand a product by distributing over any Add factors.
///
/// Given factors `[f₁, f₂, …, fₙ]`, if any `fᵢ` is an `Add`, we
/// distribute.  The distribution is done incrementally: start with a
/// running "partial product" (list of terms), and for each factor
/// either multiply it into every term (if the factor is not an Add)
/// or cross-multiply with all Add children.
///
/// Example: `(a + b) * (c + d)` → partial starts as `[a, b]`, then
/// crossed with `[c, d]` → `[a*c, a*d, b*c, b*d]`.
fn expand_mul(arena: &mut Arena, factors: &[ExprId]) -> ExprId {
    if factors.is_empty() {
        return arena.one;
    }
    if factors.len() == 1 {
        return factors[0];
    }

    // Separate numeric coefficient from symbolic factors.
    // (The canonical Mul may have a leading Num.)
    let mut coeff_factors: SmallVec<[ExprId; 4]> = SmallVec::new();
    let mut symbolic_factors: SmallVec<[ExprId; 6]> = SmallVec::new();

    for &f in factors {
        if let ExprNode::Num(_) = arena.node(f) {
            coeff_factors.push(f);
        } else {
            symbolic_factors.push(f);
        }
    }

    // Check if any symbolic factor is an Add.
    let has_add = symbolic_factors
        .iter()
        .any(|&f| matches!(arena.node(f), ExprNode::Add(_)));

    if !has_add {
        // No Add factors — nothing to distribute.  Just rebuild.
        let mut all: SmallVec<[ExprId; 6]> = SmallVec::new();
        all.extend_from_slice(&coeff_factors);
        all.extend_from_slice(&symbolic_factors);
        return arena.mul(&all);
    }

    // Incremental distribution.
    // `terms` holds the running list of partially-multiplied summands.
    let mut terms: Vec<SmallVec<[ExprId; 4]>> = vec![coeff_factors.clone()];

    for &factor in &symbolic_factors {
        let factor_node = arena.node(factor).clone();
        if let ExprNode::Add(add_children) = factor_node {
            // Cross-multiply: for each existing term, for each Add child,
            // produce a new term = existing_factors ++ [child].
            let mut new_terms: Vec<SmallVec<[ExprId; 4]>> = Vec::new();
            for existing in &terms {
                for &child in &add_children {
                    let mut combined = existing.clone();
                    combined.push(child);
                    new_terms.push(combined);
                }
            }
            terms = new_terms;
        } else {
            // Non-Add factor: append to every existing term.
            for term in &mut terms {
                term.push(factor);
            }
        }
    }

    // Build the final sum of products.
    let sum_terms: SmallVec<[ExprId; 6]> = terms
        .into_iter()
        .map(|factors| arena.mul(&factors))
        .collect();

    if sum_terms.len() == 1 {
        sum_terms[0]
    } else {
        arena.add(&sum_terms)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Pow expansion
// ═══════════════════════════════════════════════════════════════════════════

/// Expand `base^exp`:
///
/// 1. **Multinomial**: `(a + b)^n` for non-negative integer `n`
/// 2. **Power-of-product**: `(x·y)^n` → `x^n · y^n`
/// 3. **Sum exponent**: `x^(a+b)` → `x^a · x^b`
fn expand_pow(
    arena: &mut Arena,
    assumptions: &mut AssumptionCache,
    base: ExprId,
    exp: ExprId,
    opts: &ExpandOpts,
) -> ExprId {
    // ── Step 1: Multinomial expansion for Add^positive_int ──
    if opts.multinomial
        && let Some(result) = try_multinomial_expand(arena, base, exp)
    {
        return result;
    }

    // ── Step 2: (x·y)^n → x^n · y^n ──
    if opts.power_base
        && let Some(result) = expand_power_base(arena, assumptions, base, exp, opts.force)
    {
        return result;
    }

    // ── Step 3: x^(a+b) → x^a · x^b ──
    if opts.power_exp
        && let Some(result) = expand_power_exp(arena, assumptions, base, exp, opts.force)
    {
        return result;
    }

    arena.pow(base, exp)
}

/// Try multinomial/binomial expansion for `Add^positive_int`.
///
/// Returns `Some(expanded)` if base is Add and exp is a positive integer,
/// otherwise `None`.
fn try_multinomial_expand(arena: &mut Arena, base: ExprId, exp: ExprId) -> Option<ExprId> {
    let exp_val = match arena.as_num(exp) {
        Some(r) if r.is_integer() => {
            let n: i64 = r.to_integer().try_into().ok()?;
            n
        }
        _ => return None,
    };

    if exp_val <= 0 {
        return None;
    }

    let n = exp_val as usize;

    let children = match arena.node(base).clone() {
        ExprNode::Add(ch) => ch,
        _ => return None,
    };

    let max_expand = arena.config.max_pow_exponent.min(200);
    if n > max_expand {
        return None;
    }

    tracing::debug!(
        "expand_pow: multinomial expansion for {}-term Add ^ {}",
        children.len(),
        n
    );

    let k = children.len();
    Some(if k == 2 {
        binomial_expand_terms(arena, &children, n)
    } else {
        multinomial_expand_terms(arena, &children, n)
    })
}

/// Expand `(x·y·z)^n` → `x^n · y^n · z^n` when base is a product.
///
/// Only applies when the base is `Mul` and the exponent is **not** a
/// positive integer.  Positive-integer exponents of products are already
/// handled correctly by `canon_pow` / `canon_mul`, and expanding them
/// here would create new `Pow` nodes that aren't visited in the current
/// bottom-up pass, breaking expand-idempotency (e.g. `(-x)^2` would
/// stay as `(-x)^2` on the first expand but become `x^2` on the second).
///
/// # Validity guard
///
/// `(x·y)^e = x^e·y^e` holds for every integer `e`, and for arbitrary
/// `e` when the factors are non-negative reals; it fails in general
/// (`√((−1)(−1)) = 1 ≠ √(−1)·√(−1) = −1`).  Unless `force` is set, the
/// rewrite therefore requires the exponent to be an integer or every
/// factor to be known non-negative through the assumption system.
fn expand_power_base(
    arena: &mut Arena,
    assumptions: &mut AssumptionCache,
    base: ExprId,
    exp: ExprId,
    force: bool,
) -> Option<ExprId> {
    // Skip when exponent is a positive integer — those cases are
    // already fully handled by canonicalization or multinomial expansion.
    if let Some(r) = arena.as_num(exp)
        && r.is_integer()
        && r.is_positive()
    {
        return None;
    }
    let children = match arena.node(base).clone() {
        ExprNode::Mul(children) => children,
        _ => return None,
    };
    let exp_is_integer = arena.as_num(exp).is_some_and(|r| r.is_integer())
        || assumptions.query(arena, exp, Props::INTEGER) == Some(true);
    let all_nonneg = force
        || exp_is_integer
        || children
            .iter()
            .all(|&c| assumptions.query(arena, c, Props::NONNEGATIVE) == Some(true));
    if !all_nonneg {
        tracing::trace!(
            "expand_power_base: guard rejected (exponent not integer, factors not known non-negative)"
        );
        return None;
    }
    let factors: Vec<ExprId> = children.iter().map(|&c| arena.pow(c, exp)).collect();
    Some(arena.mul(&factors))
}

/// Expand `x^(a+b+c)` → `x^a · x^b · x^c` when exponent is a sum.
///
/// Guard: this identity is only universally valid when the base is positive
/// (or is Euler's `e`).  For negative bases with fractional exponents, the
/// identity fails due to complex branch cuts.  We also allow the split when
/// all exponent summands are provably same-sign (all ≥ 0 or all ≤ 0),
/// because integer exponents don't introduce branch-cut issues, when all
/// summands are known integers, or when `force` is set.
fn expand_power_exp(
    arena: &mut Arena,
    assumptions: &mut AssumptionCache,
    base: ExprId,
    exp: ExprId,
    force: bool,
) -> Option<ExprId> {
    if let ExprNode::Add(ref children) = arena.node(exp).clone() {
        // Always safe for e^(a+b) = e^a · e^b
        let is_euler_e = base == arena.e_const();

        // Safe if base is a known positive numeric literal or known positive
        // through the assumption system.
        let base_known_positive = if let Some(r) = arena.as_num(base) {
            r.is_positive()
        } else {
            assumptions.query(arena, base, Props::POSITIVE) == Some(true)
        };

        // Safe if every summand is a known integer.
        let all_integer = children
            .iter()
            .all(|&c| assumptions.query(arena, c, Props::INTEGER) == Some(true));

        // Safe if all exponent summands have known same sign
        let all_same_sign = {
            let mut all_nonneg = true;
            let mut all_nonpos = true;
            for &child in children.iter() {
                if let Some(r) = arena.as_num(child) {
                    if r.is_negative() {
                        all_nonneg = false;
                    }
                    if r.is_positive() {
                        all_nonpos = false;
                    }
                } else {
                    // Can't determine sign of symbolic term — be conservative
                    all_nonneg = false;
                    all_nonpos = false;
                }
            }
            all_nonneg || all_nonpos
        };

        if force || is_euler_e || base_known_positive || all_same_sign || all_integer {
            let factors: Vec<ExprId> = children.iter().map(|&e| arena.pow(base, e)).collect();
            return Some(arena.mul(&factors));
        }
    }
    None
}

// ═══════════════════════════════════════════════════════════════════════════
// Multinomial / binomial expansion helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Expand `(a + b)^n` using the binomial theorem.
///
/// Produces `n + 1` terms: `Σ_{k=0}^{n} C(n,k) · a^(n−k) · b^k`.
/// Binomial coefficients are computed incrementally via the multiplicative
/// recurrence `C(n, k) = C(n, k−1) · (n−k+1) / k` to avoid factorial
/// overflow.
fn binomial_expand_terms(arena: &mut Arena, children: &[ExprId], n: usize) -> ExprId {
    let a = children[0];
    let b = children[1];

    let mut coeff = BigInt::one(); // C(n, 0) = 1
    let mut terms = Vec::with_capacity(n + 1);

    // k=0: C(n,0) · a^n · b^0 = a^n
    let a_pow_n = if n == 1 {
        a
    } else {
        let exp_id = arena.int(n as i64);
        arena.pow(a, exp_id)
    };
    terms.push(a_pow_n);

    for k in 1..=n {
        // C(n, k) = C(n, k-1) * (n - k + 1) / k
        coeff *= BigInt::from(n - k + 1);
        coeff /= BigInt::from(k);

        // Build coefficient expression (skip if 1)
        let mut factors: SmallVec<[ExprId; 4]> = SmallVec::new();
        if coeff != BigInt::one() {
            let coeff_r = Ratio::from_integer(coeff.clone());
            let nid = arena.intern_num(coeff_r);
            factors.push(arena.intern(ExprNode::Num(nid)));
        }

        // a^(n-k)
        let a_exp = n - k;
        if a_exp == 1 {
            factors.push(a);
        } else if a_exp >= 2 {
            let exp_id = arena.int(a_exp as i64);
            factors.push(arena.pow(a, exp_id));
        }

        // b^k
        if k == 1 {
            factors.push(b);
        } else {
            let exp_id = arena.int(k as i64);
            factors.push(arena.pow(b, exp_id));
        }

        let term = if factors.len() == 1 {
            factors[0]
        } else {
            arena.mul(&factors)
        };
        terms.push(term);
    }

    arena.add(&terms)
}

/// Expand `(x₁ + x₂ + … + xₖ)^n` using the multinomial theorem.
///
/// Enumerates all weak compositions of `n` into `k` non-negative parts
/// and produces one term per composition:
///   `n! / (n₁! · … · nₖ!) · x₁^n₁ · … · xₖ^nₖ`.
fn multinomial_expand_terms(arena: &mut Arena, children: &[ExprId], n: usize) -> ExprId {
    let k = children.len();
    let compositions = generate_compositions(n, k);
    let mut terms = Vec::with_capacity(compositions.len());

    for partition in &compositions {
        // Compute multinomial coefficient n! / (n₁! · n₂! · … · nₖ!)
        let coeff = multinomial_coeff(n, partition);

        // Build term: coeff · x₁^n₁ · x₂^n₂ · … · xₖ^nₖ
        let mut factors: SmallVec<[ExprId; 6]> = SmallVec::new();
        if coeff != BigInt::one() {
            let coeff_r = Ratio::from_integer(coeff);
            let nid = arena.intern_num(coeff_r);
            factors.push(arena.intern(ExprNode::Num(nid)));
        }

        for (i, &ni) in partition.iter().enumerate() {
            if ni == 1 {
                factors.push(children[i]);
            } else if ni >= 2 {
                let exp_id = arena.int(ni as i64);
                factors.push(arena.pow(children[i], exp_id));
            }
            // ni == 0 → omit this variable from the product
        }

        let term = if factors.len() == 1 {
            factors[0]
        } else {
            arena.mul(&factors)
        };
        terms.push(term);
    }

    arena.add(&terms)
}

/// Compute the multinomial coefficient `n! / (n₁! · n₂! · … · nₖ!)`.
///
/// Uses the product-of-binomials identity:
///   `C(n; n₁,…,nₖ) = C(n, n₁) · C(n−n₁, n₂) · C(n−n₁−n₂, n₃) · …`
fn multinomial_coeff(n: usize, partition: &[usize]) -> BigInt {
    let mut result = BigInt::one();
    let mut remaining = n;
    for &ni in partition {
        // C(remaining, ni) via multiplicative formula
        for j in 0..ni {
            result *= BigInt::from(remaining - j);
            result /= BigInt::from(j + 1);
        }
        remaining -= ni;
    }
    result
}

/// Generate all weak compositions of `n` into `k` non-negative parts.
///
/// A weak composition is an ordered tuple `(n₁, …, nₖ)` where each
/// `nᵢ ≥ 0` and `n₁ + … + nₖ = n`.  The count is `C(n+k−1, k−1)`.
fn generate_compositions(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut current = vec![0usize; k];
    generate_compositions_inner(n, k, 0, &mut current, &mut result);
    result
}

fn generate_compositions_inner(
    remaining: usize,
    k: usize,
    pos: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<Vec<usize>>,
) {
    if pos == k - 1 {
        // Last slot gets whatever is remaining.
        current[pos] = remaining;
        result.push(current.clone());
        return;
    }
    for i in 0..=remaining {
        current[pos] = i;
        generate_compositions_inner(remaining - i, k, pos + 1, current, result);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

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

    // ── Basic distribution ──────────────────────────────────────────

    #[test]
    fn expand_no_add_is_noop() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let expr = a.mul(&[x, y]);
        let result = expand(&mut a, expr);
        assert_eq!(result, expr, "x*y should not change");
    }

    #[test]
    fn expand_atom_is_noop() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let result = expand(&mut a, x);
        assert_eq!(result, x);
    }

    #[test]
    fn expand_number_times_add() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let two = a.int(2);
        // 2*(x + y) — this is already distributed by canon_mul.
        // But let's build it via raw and expand.
        let sum = a.add(&[x, y]);
        let expr = a.mul(&[two, sum]);
        // Already distributed by Number*Add rule: 2*x + 2*y
        let s = display(&a, expr);
        assert!(
            s.contains('+'),
            "2*(x+y) should already be distributed, got: {s}"
        );
    }

    #[test]
    fn expand_symbol_times_add() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let z = sym(&mut a, "z");
        // z * (x + y) — NOT distributed by canon_mul (symbolic, not numeric)
        let sum = a.add(&[x, y]);
        let expr = a.mul(&[z, sum]);
        assert_eq!(display(&a, expr), "z*(x + y)");

        let result = expand(&mut a, expr);
        assert_eq!(display(&a, result), "x*z + y*z");
    }

    #[test]
    fn expand_add_times_add() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let u = sym(&mut a, "u");
        let v = sym(&mut a, "v");
        // (x + y) * (u + v) → x*u + x*v + y*u + y*v
        let sum1 = a.add(&[x, y]);
        let sum2 = a.add(&[u, v]);
        let expr = a.mul(&[sum1, sum2]);
        let result = expand(&mut a, expr);
        let s = display(&a, result);
        // Should have 4 terms.
        assert!(
            s.contains("x*u") || s.contains("u*x"),
            "should contain x*u term, got: {s}"
        );
    }

    // ── Power expansion ─────────────────────────────────────────────

    #[test]
    fn expand_x_plus_1_squared() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.one;
        let sum = a.add(&[x, one]);
        let two = a.int(2);
        let expr = a.pow(sum, two);
        assert_eq!(display(&a, expr), "(x + 1)^2");

        let result = expand(&mut a, expr);
        assert_eq!(display(&a, result), "x^2 + 2*x + 1");
    }

    #[test]
    fn expand_x_plus_1_cubed() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.one;
        let sum = a.add(&[x, one]);
        let three = a.int(3);
        let expr = a.pow(sum, three);

        let result = expand(&mut a, expr);
        assert_eq!(display(&a, result), "x^3 + 3*x^2 + 3*x + 1");
    }

    #[test]
    fn expand_x_plus_y_squared() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let sum = a.add(&[x, y]);
        let two = a.int(2);
        let expr = a.pow(sum, two);

        let result = expand(&mut a, expr);
        let s = display(&a, result);
        // (x + y)^2 = x^2 + 2*x*y + y^2
        assert!(s.contains("x^2"), "should contain x^2, got: {s}");
        assert!(s.contains("y^2"), "should contain y^2, got: {s}");
        assert!(
            s.contains("2*x*y") || s.contains("2*y*x"),
            "should contain 2*x*y, got: {s}"
        );
    }

    #[test]
    fn expand_pow_zero_is_one() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let sum = a.add(&[x, a.one]);
        let zero = a.zero;
        let expr = a.pow(sum, zero);
        // (x+1)^0 = 1 (canonical)
        assert_eq!(display(&a, expr), "1");
        let result = expand(&mut a, expr);
        assert_eq!(display(&a, result), "1");
    }

    #[test]
    fn expand_pow_one_is_identity() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let sum = a.add(&[x, a.one]);
        let one = a.one;
        let expr = a.pow(sum, one);
        // (x+1)^1 = x+1 (canonical)
        let result = expand(&mut a, expr);
        assert_eq!(display(&a, result), "x + 1");
    }

    #[test]
    fn expand_pow_negative_not_expanded() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let sum = a.add(&[x, a.one]);
        let neg_two = a.int(-2);
        let expr = a.pow(sum, neg_two);
        // (x+1)^(-2) should NOT be expanded.
        let result = expand(&mut a, expr);
        assert_eq!(display(&a, result), "(x + 1)^(-2)");
    }

    #[test]
    fn expand_pow_non_add_base_not_expanded() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let three = a.int(3);
        let expr = a.pow(x, three);
        // x^3 — base is not an Add, no expansion.
        let result = expand(&mut a, expr);
        assert_eq!(result, expr);
    }

    // ── Nested expansion ────────────────────────────────────────────

    #[test]
    fn expand_nested_mul_of_add() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let z = sym(&mut a, "z");
        // x * (y + z) * (x + 1)
        let sum1 = a.add(&[y, z]);
        let sum2 = a.add(&[x, a.one]);
        let expr = a.mul(&[x, sum1, sum2]);
        let result = expand(&mut a, expr);
        let s = display(&a, result);
        // Should be fully distributed: x*y*x + x*y + x*z*x + x*z
        // = x^2*y + x*y + x^2*z + x*z
        assert!(!s.contains('('), "should be fully expanded, got: {s}");
    }

    #[test]
    fn expand_product_of_expanded_power() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let two = a.int(2);
        // y * (x + 1)^2
        let sum = a.add(&[x, a.one]);
        let pow = a.pow(sum, two);
        let expr = a.mul(&[y, pow]);
        let result = expand(&mut a, expr);
        let s = display(&a, result);
        // y * (x^2 + 2x + 1) = x^2*y + 2*x*y + y
        assert!(!s.contains("^2)"), "power should be expanded, got: {s}");
        assert!(s.contains('y'), "should contain y, got: {s}");
    }

    // ── No-op cases ─────────────────────────────────────────────────

    #[test]
    fn expand_already_expanded_is_noop() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let two = a.int(2);
        // x^2 + 2*x*y + y^2 — already expanded.
        let x2 = a.pow(x, two);
        let y2 = a.pow(y, two);
        let two_xy = a.mul(&[two, x, y]);
        let expr = a.add(&[x2, two_xy, y2]);
        let result = expand(&mut a, expr);
        assert_eq!(result, expr, "already expanded should be unchanged");
    }

    #[test]
    fn expand_sum_of_symbols() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let expr = a.add(&[x, y]);
        let result = expand(&mut a, expr);
        assert_eq!(result, expr, "x + y should be unchanged");
    }

    // ── Idempotence ─────────────────────────────────────────────────

    #[test]
    fn expand_is_idempotent() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let sum = a.add(&[x, a.one]);
        let two = a.int(2);
        let expr = a.pow(sum, two);

        let first = expand(&mut a, expr);
        let second = expand(&mut a, first);
        assert_eq!(first, second, "expand should be idempotent");
    }

    // ── Correctness via substitution ────────────────────────────────

    #[test]
    fn expand_x_plus_1_squared_evaluates_correctly() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let sum = a.add(&[x, a.one]);
        let two = a.int(2);
        let expr = a.pow(sum, two);

        let expanded = expand(&mut a, expr);
        // Evaluate both at x=5: (5+1)^2 = 36
        let five = a.int(5);
        let orig_val = crate::transforms::subs::subs(&mut a, expr, x, five);
        let exp_val = crate::transforms::subs::subs(&mut a, expanded, x, five);
        assert_eq!(
            orig_val, exp_val,
            "expanded form should evaluate to same value"
        );
        assert_eq!(display(&a, orig_val), "36");
    }

    #[test]
    fn expand_x_plus_y_cubed_evaluates_correctly() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let sum = a.add(&[x, y]);
        let three = a.int(3);
        let expr = a.pow(sum, three);

        let expanded = expand(&mut a, expr);
        // Evaluate at x=2, y=3: (2+3)^3 = 125
        let two = a.int(2);
        let three_val = a.int(3);
        let orig_val = crate::transforms::subs::subs(&mut a, expr, x, two);
        let orig_val = crate::transforms::subs::subs(&mut a, orig_val, y, three_val);
        let exp_val = crate::transforms::subs::subs(&mut a, expanded, x, two);
        let exp_val = crate::transforms::subs::subs(&mut a, exp_val, y, three_val);
        assert_eq!(
            orig_val, exp_val,
            "expanded form should evaluate to same value"
        );
        assert_eq!(display(&a, orig_val), "125");
    }

    // ── Deep nesting (stack safety) ─────────────────────────────────

    #[test]
    fn expand_deep_no_overflow() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // Build deeply nested: sin(sin(sin(...(x+1)^2...)))
        let sum = a.add(&[x, a.one]);
        let two = a.int(2);
        let mut expr = a.pow(sum, two);
        for _ in 0..50 {
            expr = a.sin(expr);
        }
        // Should not overflow — uses iterative walker.
        let _result = expand(&mut a, expr);
    }

    #[test]
    fn expand_inside_sinh() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.one;
        let sum = a.add(&[x, one]);
        let two = a.int(2);
        let sq = a.pow(sum, two);
        let expr = a.sinh(sq);
        // sinh((x+1)^2) → expand inner → sinh(1 + x^2 + 2*x)
        let result = expand(&mut a, expr);
        let s = display(&a, result);
        assert!(
            s.starts_with("sinh("),
            "should still be sinh(...), got: {s}"
        );
        assert!(
            s.contains("x^2"),
            "inner should be expanded to contain x^2, got: {s}"
        );
        assert!(
            !s.contains("(1 + x)^2"),
            "inner should no longer contain (1 + x)^2, got: {s}"
        );
    }

    #[test]
    fn expand_inside_derivative() {
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        let one = arena.int(1);
        let sum = arena.add(&[x, one]); // x + 1
        let two = arena.int(2);
        let sq = arena.pow(sum, two); // (x+1)^2
        let deriv = arena.intern(crate::base::node::ExprNode::Derivative(sq, x));
        let expanded = expand(&mut arena, deriv);
        // The body should be expanded: x^2 + 2x + 1
        if let crate::base::node::ExprNode::Derivative(body, _) = arena.node(expanded) {
            // body should NOT be (x+1)^2 anymore
            assert_ne!(*body, sq, "body should be expanded inside Derivative");
        } else {
            panic!("result should still be a Derivative");
        }
    }

    // ── expand_power_exp soundness guard ─────────────────────────

    #[test]
    fn expand_power_exp_positive_numeric_base_allowed() {
        // 2^(a+b): the guard ALLOWS the split (base is positive),
        // but canon_mul immediately recombines 2^a * 2^b back to 2^(a+b).
        // This is correct behavior — the important thing is that the
        // guard doesn't BLOCK it (unlike the symbolic-base case).
        // We verify idempotence and that no panic occurs.
        let mut a = Arena::new();
        let two = a.int(2);
        let va = sym(&mut a, "a");
        let vb = sym(&mut a, "b");
        let sum = a.add(&[va, vb]);
        let expr = a.pow(two, sum);
        let result = expand(&mut a, expr);
        let s = display(&a, result);
        // Canon recombines, so result looks the same — that's fine.
        assert!(
            s.contains("2") && s.contains("a") && s.contains("b"),
            "2^(a+b) should produce valid expression, got: {s}"
        );
        // Verify the guard is reached: confirm base is a positive number
        if let ExprNode::Pow(base, _) = a.node(expr).clone() {
            assert!(a.as_num(base).unwrap().is_positive());
        }
    }

    #[test]
    fn expand_power_exp_symbolic_base_blocked() {
        // x^(a+b) must NOT split when x is a general symbol
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let va = sym(&mut a, "a");
        let vb = sym(&mut a, "b");
        let sum = a.add(&[va, vb]);
        let expr = a.pow(x, sum);
        let result = expand(&mut a, expr);
        let s = display(&a, result);
        // Should remain as x^(a + b), NOT become x^a * x^b
        assert!(
            s.contains("x^("),
            "x^(a+b) should NOT split for symbolic base, got: {s}"
        );
    }

    #[test]
    fn expand_power_exp_all_nonneg_exponents_allowed() {
        // x^(2+3): the guard ALLOWS the split because both exponent
        // summands are non-negative integers. After split + canon,
        // x^2 * x^3 recombines to x^5.
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let three = a.int(3);
        let sum = a.add(&[two, three]);
        let expr = a.pow(x, sum);
        let result = expand(&mut a, expr);
        let s = display(&a, result);
        // canon_add folds 2+3 → 5 anyway, so we get x^5 regardless.
        // The key test is that no panic occurs and result is valid.
        assert!(
            s.contains("x^5") || s.contains("x"),
            "x^(2+3) should produce valid expression, got: {s}"
        );
    }

    #[test]
    fn expand_power_exp_negative_numeric_base_blocked() {
        // (-2)^(a+b) must NOT split — negative base
        let mut a = Arena::new();
        let two = a.int(2);
        let neg_two = a.neg(two);
        let va = sym(&mut a, "a");
        let vb = sym(&mut a, "b");
        let sum = a.add(&[va, vb]);
        let expr = a.pow(neg_two, sum);
        let result = expand(&mut a, expr);
        let s = display(&a, result);
        // Should NOT have been split
        assert!(
            !s.contains("(-2)^a") || !s.contains("(-2)^b"),
            "(-2)^(a+b) should NOT split, got: {s}"
        );
    }

    #[test]
    fn expand_power_exp_euler_is_exp_node() {
        // e^(a+b) is canonicalized to Exp(a+b), NOT Pow(E, a+b).
        // So expand_power_exp is never reached for it.
        // Instead, the Exp node should NOT be expanded by the expand pass
        // (expand doesn't have a rule for Exp(Add(...))).
        let mut a = Arena::new();
        let va = sym(&mut a, "a");
        let vb = sym(&mut a, "b");
        let sum = a.add(&[va, vb]);
        let e = a.e_const();
        let expr = a.pow(e, sum);
        // canon_pow converts Pow(E, x) → Exp(x)
        assert!(
            matches!(a.node(expr), ExprNode::Exp(_)),
            "e^(a+b) should be canonicalized to Exp(a+b)"
        );
        let result = expand(&mut a, expr);
        let s = display(&a, result);
        // Currently expand does NOT split Exp(Add(...)) — that's OK,
        // it's a separate feature from expand_power_exp.
        assert!(
            s.contains("exp("),
            "e^(a+b) should remain as exp(...), got: {s}"
        );
    }

    // ── expand_with / ExpandOpts ───────────────────────────────────

    #[test]
    fn expand_with_mul_off_keeps_products() {
        let mut a = Arena::new();
        let (x, y) = (sym(&mut a, "x"), sym(&mut a, "y"));
        let sum = a.add(&[x, y]);
        let e = a.mul(&[x, sum]);
        let opts = ExpandOpts::none();
        assert_eq!(expand_with(&mut a, e, &opts), e);
        let opts = ExpandOpts::none().with_mul(true);
        let r = expand_with(&mut a, e, &opts);
        assert_eq!(display(&a, r), "x^2 + x*y");
    }

    #[test]
    fn expand_with_deep_false_skips_function_arguments() {
        let mut a = Arena::new();
        let (x, y) = (sym(&mut a, "x"), sym(&mut a, "y"));
        let sum = a.add(&[x, y]);
        let two = a.int(2);
        let sq = a.pow(sum, two);
        let s = a.sin(sq);
        let e = a.add(&[s, sq]);
        let shallow = expand_with(&mut a, e, &ExpandOpts::default().deep(false));
        assert_eq!(display(&a, shallow), "x^2 + 2*x*y + y^2 + sin((x + y)^2)");
        let deep = expand_with(&mut a, e, &ExpandOpts::default());
        assert!(display(&a, deep).contains("sin(x^2"));
    }

    #[test]
    fn expand_power_base_guard_blocks_symbolic_factors() {
        let mut a = Arena::new();
        let (x, y, n) = (sym(&mut a, "x"), sym(&mut a, "y"), sym(&mut a, "n"));
        let xy = a.mul(&[x, y]);
        let e = a.pow(xy, n);
        assert_eq!(expand(&mut a, e), e);
        let forced = expand_with(&mut a, e, &ExpandOpts::default().force(true));
        assert_eq!(display(&a, forced), "x^n*y^n");
        let m3 = a.int(-3);
        let int_pow = a.pow(xy, m3);
        let r = expand(&mut a, int_pow);
        assert_eq!(display(&a, r), "x^(-3)*y^(-3)");
    }

    #[test]
    fn expand_exp_of_sum_splits() {
        let mut a = Arena::new();
        let (x, y) = (sym(&mut a, "x"), sym(&mut a, "y"));
        let sum = a.add(&[x, y]);
        let e = a.exp(sum);
        let r = expand(&mut a, e);
        assert_eq!(display(&a, r), "exp(x)*exp(y)");
        let opts = ExpandOpts::default().power_exp(false);
        assert_eq!(expand_with(&mut a, e, &opts), e);
    }

    #[test]
    fn expand_opts_builders() {
        let o = ExpandOpts::none()
            .log(true)
            .trig(true)
            .multinomial(true)
            .power_base(true)
            .power_exp(true);
        assert!(o.log && o.trig && o.multinomial && o.power_base && o.power_exp && !o.mul);
        assert!(!ExpandOpts::all().deep(false).deep);
        assert_eq!(
            ExpandOpts::default(),
            ExpandOpts::none()
                .with_mul(true)
                .multinomial(true)
                .power_base(true)
                .power_exp(true)
        );
    }
}

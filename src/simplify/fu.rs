//! Trig simplification using the algorithm from "Automated and readable
//! simplification of trigonometric expressions" (2006).
//!
//! Implements named transforms TR0–TR14, TRmorrie, and TRpower, organized
//! into rule lists with a greedy tree search.  The [`fu`] function
//! orchestrates all transforms and picks the best result by
//! `(trig_count, op_count)` measure.

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::Q;
use crate::base::walk;
use crate::simplify::simplify_engine::count_ops;

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::One;
use rustc_hash::FxHashMap;

// ═══════════════════════════════════════════════════════════════════════════
// Walk helper
// ═══════════════════════════════════════════════════════════════════════════

/// Walk an expression bottom-up, applying `transform` at each node.
///
/// Like [`walk::walk_and_rebuild`] but the transform closure receives
/// `&mut Arena` so it can construct new nodes.  Returns `Some(replacement)`
/// to replace the node, or `None` to keep it (with rebuilt children).
fn walk_transform(
    arena: &mut Arena,
    expr: ExprId,
    transform: impl Fn(&mut Arena, ExprId) -> Option<ExprId>,
) -> ExprId {
    let post_order = walk::post_order_ids(arena, expr);
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();

    for &id in &post_order {
        let rebuilt = walk::rebuild_with_cache(arena, id, &cache);
        let result = transform(arena, rebuilt).unwrap_or(rebuilt);
        cache.insert(id, result);
    }

    cache.get(&expr).copied().unwrap_or(expr)
}

// ═══════════════════════════════════════════════════════════════════════════
// Measure
// ═══════════════════════════════════════════════════════════════════════════

/// The number of trig function *occurrences* in the expression tree —
/// Fu's `L` measure, counted with multiplicity as SymPy does.  The arena is
/// a DAG, so `16·cos⁷x − 24·cos⁵x + 10·cos³x − cos x` shares one `cos(x)`
/// node; counting nodes once would score it as *one* trig function and
/// prefer it to Morrie's `sin(8x)/(8·sin x)` (two).  Computed bottom-up over
/// the post-order so shared subtrees are still visited once.
fn trig_count(arena: &Arena, expr: ExprId) -> usize {
    let order = walk::post_order_ids(arena, expr);
    let mut counts: rustc_hash::FxHashMap<ExprId, usize> = rustc_hash::FxHashMap::default();
    for &id in &order {
        let own = usize::from(matches!(
            arena.node(id),
            ExprNode::Sin(_) | ExprNode::Cos(_) | ExprNode::Tan(_)
        ));
        let below: usize = arena
            .children(id)
            .iter()
            .map(|c| counts.get(c).copied().unwrap_or(0))
            .sum();
        counts.insert(id, own + below);
    }
    counts.get(&expr).copied().unwrap_or(0)
}

/// Measure function for comparing candidate simplifications.
///
/// Returns `(trig_count, op_count)` — compared lexicographically to pick
/// the form with the fewest trig functions, breaking ties by total
/// operation count.
fn measure(arena: &Arena, expr: ExprId) -> (usize, usize) {
    (trig_count(arena, expr), count_ops(arena, expr))
}

/// Pick the best expression from candidates by measure.
fn pick_best(arena: &Arena, candidates: &[ExprId]) -> ExprId {
    candidates
        .iter()
        .copied()
        .min_by_key(|&e| measure(arena, e))
        .unwrap_or(candidates[0])
}

// ═══════════════════════════════════════════════════════════════════════════
// Predicate helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Check if expression tree contains any trig node.
fn has_trig(arena: &Arena, expr: ExprId) -> bool {
    walk::post_order_ids(arena, expr).iter().any(|&id| {
        matches!(
            arena.node(id),
            ExprNode::Sin(_) | ExprNode::Cos(_) | ExprNode::Tan(_)
        )
    })
}

/// Check if expression tree contains a `Tan` node.
fn has_tan(arena: &Arena, expr: ExprId) -> bool {
    walk::post_order_ids(arena, expr)
        .iter()
        .any(|&id| matches!(arena.node(id), ExprNode::Tan(_)))
}

/// Check if expression tree contains a `Sin` or `Cos` node.
fn has_sin_or_cos(arena: &Arena, expr: ExprId) -> bool {
    walk::post_order_ids(arena, expr)
        .iter()
        .any(|&id| matches!(arena.node(id), ExprNode::Sin(_) | ExprNode::Cos(_)))
}

/// Check if an expression is a numeric integer with a given value.
fn is_integer_val(arena: &Arena, id: ExprId, val: i64) -> bool {
    arena
        .as_num(id)
        .is_some_and(|r| r.is_integer() && *r == Ratio::from_integer(BigInt::from(val)))
}

/// Extract the base argument `x` from `2*x` (i.e. `Mul([2, x])`).
///
/// Returns `Some(x)` if the expression is `Mul([2, x])` for some `x`.
fn extract_double_angle_arg(arena: &Arena, expr: ExprId) -> Option<ExprId> {
    if let ExprNode::Mul(ref children) = arena.node(expr).clone()
        && children.len() == 2
    {
        if is_integer_val(arena, children[0], 2) {
            return Some(children[1]);
        }
        if is_integer_val(arena, children[1], 2) {
            return Some(children[0]);
        }
    }
    None
}

// ═══════════════════════════════════════════════════════════════════════════
// TR0: Normalize via eval + pattern rules
// ═══════════════════════════════════════════════════════════════════════════

/// TR0: Normalize via eval and basic pattern-rule simplification.
///
/// Delegates to `eval::eval` then `pattern::apply_rules` with the
/// standard rule set (Pythagorean identity, exp/ln cancellation, etc.).
fn tr0(arena: &mut Arena, expr: ExprId) -> ExprId {
    let evaled = crate::transforms::eval::eval(arena, expr);
    let rules = crate::transforms::pattern::basic_rules(arena);
    let (result, _) = crate::transforms::pattern::apply_rules(arena, evaled, &rules);
    result
}

// ═══════════════════════════════════════════════════════════════════════════
// TR1: sec(x) → 1/cos(x), csc(x) → 1/sin(x)
// ═══════════════════════════════════════════════════════════════════════════

/// TR1: Rewrite reciprocal trig functions.
///
/// In this system, `sec` and `csc` don't have dedicated nodes — they are
/// already represented as `cos(x)^(-1)` and `sin(x)^(-1)`.  This
/// transform is effectively a no-op but is included for algorithm
/// completeness.
fn tr1(_arena: &mut Arena, expr: ExprId) -> ExprId {
    expr
}

// ═══════════════════════════════════════════════════════════════════════════
// TR2: tan(x) → sin(x)/cos(x)
// ═══════════════════════════════════════════════════════════════════════════

/// TR2: Expand tan to sin/cos ratio.
///
/// `tan(x) → sin(x)/cos(x)`
fn tr2(arena: &mut Arena, expr: ExprId) -> ExprId {
    walk_transform(arena, expr, |arena, id| {
        let node = arena.node(id).clone();
        if let ExprNode::Tan(inner) = node {
            let sin = arena.sin(inner);
            let cos = arena.cos(inner);
            Some(arena.div(sin, cos))
        } else {
            None
        }
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// TR2i: sin(x)/cos(x) → tan(x)
// ═══════════════════════════════════════════════════════════════════════════

/// TR2i: Contract sin/cos ratios back to tan.
///
/// `sin(x)/cos(x) → tan(x)` — detects `Mul` nodes containing `Sin(x)`
/// and `Pow(Cos(x), -1)` with matching arguments.
fn tr2i(arena: &mut Arena, expr: ExprId) -> ExprId {
    walk_transform(arena, expr, |arena, id| {
        let node = arena.node(id).clone();
        if let ExprNode::Mul(ref children) = node {
            let mut sin_factors: Vec<(usize, ExprId)> = Vec::new();
            let mut cos_inv_factors: Vec<(usize, ExprId)> = Vec::new();

            for (idx, &child) in children.iter().enumerate() {
                match arena.node(child).clone() {
                    ExprNode::Sin(arg) => sin_factors.push((idx, arg)),
                    ExprNode::Pow(base, exp) => {
                        if is_integer_val(arena, exp, -1)
                            && let ExprNode::Cos(arg) = arena.node(base).clone()
                        {
                            cos_inv_factors.push((idx, arg));
                        }
                    }
                    _ => {}
                }
            }

            // Match sin(x) * cos(x)^(-1) → tan(x)
            for &(si, sin_arg) in &sin_factors {
                for &(ci, cos_arg) in &cos_inv_factors {
                    if sin_arg == cos_arg {
                        let tan_x = arena.tan(sin_arg);
                        let remaining: Vec<ExprId> = children
                            .iter()
                            .enumerate()
                            .filter(|&(idx, _)| idx != si && idx != ci)
                            .map(|(_, &c)| c)
                            .collect();
                        if remaining.is_empty() {
                            return Some(tan_x);
                        }
                        let mut all = remaining;
                        all.push(tan_x);
                        return Some(arena.mul(&all));
                    }
                }
            }
        }
        None
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// TR3 / TR4: Evaluate at special angles
// ═══════════════════════════════════════════════════════════════════════════

/// TR3 / TR4: Evaluate trig at special angles by delegating to eval.
///
/// `sin(π/4) → √2/2`, `cos(0) → 1`, etc.
fn tr3(arena: &mut Arena, expr: ExprId) -> ExprId {
    crate::transforms::eval::eval(arena, expr)
}

// ═══════════════════════════════════════════════════════════════════════════
// TR5: sin²(x) → (1 - cos(2x))/2
// ═══════════════════════════════════════════════════════════════════════════

/// TR5: Power-reducing identity for sin².
///
/// `sin²(x) → (1 - cos(2x))/2`
fn tr5(arena: &mut Arena, expr: ExprId) -> ExprId {
    walk_transform(arena, expr, |arena, id| {
        let node = arena.node(id).clone();
        if let ExprNode::Pow(base, exp) = node {
            if !is_integer_val(arena, exp, 2) {
                return None;
            }
            if let ExprNode::Sin(inner) = arena.node(base).clone() {
                let two = arena.int(2);
                let two_x = arena.mul(&[two, inner]);
                let cos_2x = arena.cos(two_x);
                let one = arena.one;
                let one_minus_cos = arena.sub(one, cos_2x);
                let half = arena.rational(1, 2);
                return Some(arena.mul(&[half, one_minus_cos]));
            }
        }
        None
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// TR5i: (1 - cos(2x))/2 → sin²(x) — inverse of TR5
// ═══════════════════════════════════════════════════════════════════════════

/// TR5i: Inverse power-reducing identity.
///
/// Recognises `1/2 - cos(2x)/2` and replaces with `sin²(x)`.
/// Also recognises `1/2 + cos(2x)/2` and replaces with `cos²(x)`.
fn tr5i(arena: &mut Arena, expr: ExprId) -> ExprId {
    walk_transform(arena, expr, |arena, id| {
        let node = arena.node(id).clone();
        if let ExprNode::Add(ref children) = node {
            let half_pos = Ratio::new(BigInt::from(1), BigInt::from(2));
            let half_neg = Ratio::new(BigInt::from(-1), BigInt::from(2));

            // Find constant 1/2 term and cos(2x) terms with ±1/2 coefficient
            let mut const_half_idx: Option<usize> = None;
            let mut cos_2x_entries: Vec<(usize, ExprId, bool)> = Vec::new(); // (idx, base_arg, is_negative)

            for (idx, &child) in children.iter().enumerate() {
                let (coeff, term) = arena.as_coeff_term(child);
                if term == arena.one && coeff == half_pos {
                    const_half_idx = Some(idx);
                }
                if let ExprNode::Cos(inner) = arena.node(term).clone()
                    && let Some(base_arg) = extract_double_angle_arg(arena, inner)
                {
                    if coeff == half_neg {
                        cos_2x_entries.push((idx, base_arg, true));
                    } else if coeff == half_pos {
                        cos_2x_entries.push((idx, base_arg, false));
                    }
                }
            }

            if let Some(half_idx) = const_half_idx
                && let Some(&(cos_idx, base_arg, is_negative)) = cos_2x_entries.first()
            {
                if is_negative {
                    // 1/2 - cos(2x)/2 → sin²(x)
                    let sin_x = arena.sin(base_arg);
                    let two = arena.int(2);
                    let result = arena.pow(sin_x, two);
                    return Some(replace_pair_in_add(
                        arena, children, half_idx, cos_idx, result,
                    ));
                } else {
                    // 1/2 + cos(2x)/2 → cos²(x)
                    let cos_x = arena.cos(base_arg);
                    let two = arena.int(2);
                    let result = arena.pow(cos_x, two);
                    return Some(replace_pair_in_add(
                        arena, children, half_idx, cos_idx, result,
                    ));
                }
            }
        }
        None
    })
}

/// Replace two children in an Add with a single replacement.
fn replace_pair_in_add(
    arena: &mut Arena,
    children: &[ExprId],
    idx1: usize,
    idx2: usize,
    replacement: ExprId,
) -> ExprId {
    let remaining: Vec<ExprId> = children
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != idx1 && i != idx2)
        .map(|(_, &c)| c)
        .collect();
    if remaining.is_empty() {
        return replacement;
    }
    let mut new_children = remaining;
    new_children.push(replacement);
    arena.add(&new_children)
}

// ═══════════════════════════════════════════════════════════════════════════
// TR6: cos²(x) → (1 + cos(2x))/2
// ═══════════════════════════════════════════════════════════════════════════

/// TR6: Power-reducing identity for cos².
///
/// `cos²(x) → (1 + cos(2x))/2`
fn tr6(arena: &mut Arena, expr: ExprId) -> ExprId {
    walk_transform(arena, expr, |arena, id| {
        let node = arena.node(id).clone();
        if let ExprNode::Pow(base, exp) = node {
            if !is_integer_val(arena, exp, 2) {
                return None;
            }
            if let ExprNode::Cos(inner) = arena.node(base).clone() {
                let two = arena.int(2);
                let two_x = arena.mul(&[two, inner]);
                let cos_2x = arena.cos(two_x);
                let one = arena.one;
                let one_plus_cos = arena.add(&[one, cos_2x]);
                let half = arena.rational(1, 2);
                return Some(arena.mul(&[half, one_plus_cos]));
            }
        }
        None
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// TR7: Lower power of cos (alias for TR6)
// ═══════════════════════════════════════════════════════════════════════════

/// TR7: Lower power of cos — same transformation as TR6.
fn tr7(arena: &mut Arena, expr: ExprId) -> ExprId {
    tr6(arena, expr)
}

// ═══════════════════════════════════════════════════════════════════════════
// TR8: Product-to-sum (delegates to trig_combine)
// ═══════════════════════════════════════════════════════════════════════════

/// TR8: Product-to-sum identities for trig.
///
/// `sin(a)·cos(b) → [sin(a+b) + sin(a-b)]/2`, etc.
/// Delegates to the existing [`trig_combine`](crate::simplify::trig_combine) module.
fn tr8(arena: &mut Arena, expr: ExprId) -> ExprId {
    crate::simplify::trig_combine::trig_combine(arena, expr)
}

// ═══════════════════════════════════════════════════════════════════════════
// TR9: Sum-to-product
// ═══════════════════════════════════════════════════════════════════════════

/// TR9: Sum-to-product identities.
///
/// - `sin(a) + sin(b) → 2·sin((a+b)/2)·cos((a-b)/2)`
/// - `sin(a) - sin(b) → 2·cos((a+b)/2)·sin((a-b)/2)`
/// - `cos(a) + cos(b) → 2·cos((a+b)/2)·cos((a-b)/2)`
/// - `cos(a) - cos(b) → -2·sin((a+b)/2)·sin((a-b)/2)`
fn tr9(arena: &mut Arena, expr: ExprId) -> ExprId {
    walk_transform(arena, expr, |arena, id| {
        let node = arena.node(id).clone();
        if let ExprNode::Add(ref children) = node {
            // Collect sin/cos terms with unit or negated coefficients.
            let mut trig_terms: Vec<(usize, ExprId, bool, i8)> = Vec::new(); // (idx, arg, is_sin, sign)

            for (idx, &child) in children.iter().enumerate() {
                let (coeff, term) = arena.as_coeff_term(child);
                let sign = if coeff == Ratio::one() {
                    1i8
                } else if coeff == -Ratio::<BigInt>::one() {
                    -1i8
                } else {
                    continue;
                };
                match arena.node(term).clone() {
                    ExprNode::Sin(arg) => trig_terms.push((idx, arg, true, sign)),
                    ExprNode::Cos(arg) => trig_terms.push((idx, arg, false, sign)),
                    _ => {}
                }
            }

            // Find matching pairs (same trig type, different arguments).
            for i in 0..trig_terms.len() {
                for j in (i + 1)..trig_terms.len() {
                    let (idx_i, arg_i, is_sin_i, sign_i) = trig_terms[i];
                    let (idx_j, arg_j, is_sin_j, sign_j) = trig_terms[j];

                    if is_sin_i != is_sin_j {
                        continue;
                    }
                    if arg_i == arg_j {
                        continue;
                    }

                    let half = arena.rational(1, 2);
                    let a_plus_b = arena.add(&[arg_i, arg_j]);
                    let a_minus_b = arena.sub(arg_i, arg_j);
                    let half_sum = arena.mul(&[half, a_plus_b]);
                    let half_diff = arena.mul(&[half, a_minus_b]);
                    let two = arena.int(2);

                    let replacement = if is_sin_i {
                        // sin terms
                        if sign_i == 1 && sign_j == 1 {
                            // sin(a) + sin(b)
                            let s = arena.sin(half_sum);
                            let c = arena.cos(half_diff);
                            arena.mul(&[two, s, c])
                        } else if sign_i == 1 && sign_j == -1 {
                            // sin(a) - sin(b)
                            let c = arena.cos(half_sum);
                            let s = arena.sin(half_diff);
                            arena.mul(&[two, c, s])
                        } else if sign_i == -1 && sign_j == 1 {
                            // -sin(a) + sin(b) = sin(b) - sin(a)
                            let b_minus_a = arena.sub(arg_j, arg_i);
                            let half_diff2 = arena.mul(&[half, b_minus_a]);
                            let half_sum2 = arena.mul(&[half, a_plus_b]);
                            let c = arena.cos(half_sum2);
                            let s = arena.sin(half_diff2);
                            arena.mul(&[two, c, s])
                        } else {
                            // -sin(a) - sin(b)
                            let neg_two = arena.int(-2);
                            let s = arena.sin(half_sum);
                            let c = arena.cos(half_diff);
                            arena.mul(&[neg_two, s, c])
                        }
                    } else {
                        // cos terms
                        if sign_i == 1 && sign_j == 1 {
                            // cos(a) + cos(b)
                            let c1 = arena.cos(half_sum);
                            let c2 = arena.cos(half_diff);
                            arena.mul(&[two, c1, c2])
                        } else if sign_i == 1 && sign_j == -1 {
                            // cos(a) - cos(b)
                            let neg_two = arena.int(-2);
                            let s1 = arena.sin(half_sum);
                            let s2 = arena.sin(half_diff);
                            arena.mul(&[neg_two, s1, s2])
                        } else if sign_i == -1 && sign_j == 1 {
                            // -cos(a) + cos(b) = cos(b) - cos(a)
                            let neg_two = arena.int(-2);
                            let b_minus_a = arena.sub(arg_j, arg_i);
                            let half_diff2 = arena.mul(&[half, b_minus_a]);
                            let half_sum2 = arena.mul(&[half, a_plus_b]);
                            let s1 = arena.sin(half_sum2);
                            let s2 = arena.sin(half_diff2);
                            arena.mul(&[neg_two, s1, s2])
                        } else {
                            // -cos(a) - cos(b)
                            let neg_two = arena.int(-2);
                            let c1 = arena.cos(half_sum);
                            let c2 = arena.cos(half_diff);
                            arena.mul(&[neg_two, c1, c2])
                        }
                    };

                    return Some(replace_pair_in_add(
                        arena,
                        children,
                        idx_i,
                        idx_j,
                        replacement,
                    ));
                }
            }
        }
        None
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// TR10i: Factor compound angle (inverse of expand_trig)
// ═══════════════════════════════════════════════════════════════════════════

/// Information about a product of exactly two trig functions.
struct SinCosPair {
    sin_arg: ExprId,
    cos_arg: ExprId,
    coeff: Q,
}

/// Information about a product of two trig functions of the same type.
struct SameTypePair {
    arg1: ExprId,
    arg2: ExprId,
    is_sin: bool,
    coeff: Q,
}

/// The two trig factors of a term that is *exactly* `coeff · trig(a) · trig(b)`:
/// `Some((sin args, cos args))` with two entries in total, `None` if the
/// product has any other factor.  The addition formulas need the whole
/// term; `sin(c)·cos(a)·cos(b) + sin(a)·sin(b)` is *not* `cos(a − b)`, and
/// ignoring the `sin(c)` used to make `fu` return a wrong value for a
/// rotation-matrix entry (`RᵀR` came out `0.78`, not `1`).
fn exactly_two_trig_factors(arena: &Arena, core: ExprId) -> Option<(Vec<ExprId>, Vec<ExprId>)> {
    let factors: Vec<ExprId> = match arena.node(core) {
        ExprNode::Mul(children) => children.to_vec(),
        _ => return None,
    };
    if factors.len() != 2 {
        return None;
    }
    let mut sin_args = Vec::new();
    let mut cos_args = Vec::new();
    for &f in &factors {
        match arena.node(f) {
            ExprNode::Sin(arg) => sin_args.push(*arg),
            ExprNode::Cos(arg) => cos_args.push(*arg),
            _ => return None,
        }
    }
    Some((sin_args, cos_args))
}

/// Try to extract `coeff * sin(a) * cos(b)` from a term.
fn extract_sin_cos_pair(arena: &mut Arena, term: ExprId) -> Option<SinCosPair> {
    let (coeff, core) = arena.as_coeff_term(term);
    let (sin_args, cos_args) = exactly_two_trig_factors(arena, core)?;
    match (sin_args.as_slice(), cos_args.as_slice()) {
        ([sin_arg], [cos_arg]) => Some(SinCosPair {
            sin_arg: *sin_arg,
            cos_arg: *cos_arg,
            coeff,
        }),
        _ => None,
    }
}

/// Try to extract `coeff * sin(a) * sin(b)` or `coeff * cos(a) * cos(b)`.
fn extract_same_type_pair(arena: &mut Arena, term: ExprId) -> Option<SameTypePair> {
    let (coeff, core) = arena.as_coeff_term(term);
    let (sin_args, cos_args) = exactly_two_trig_factors(arena, core)?;

    if sin_args.len() == 2 {
        return Some(SameTypePair {
            arg1: sin_args[0],
            arg2: sin_args[1],
            is_sin: true,
            coeff,
        });
    }
    if cos_args.len() == 2 {
        return Some(SameTypePair {
            arg1: cos_args[0],
            arg2: cos_args[1],
            is_sin: false,
            coeff,
        });
    }
    None
}

/// TR10i: Factor into compound angle — inverse of the addition formulas.
///
/// - `sin(a)·cos(b) + cos(a)·sin(b) → sin(a+b)`
/// - `sin(a)·cos(b) - cos(a)·sin(b) → sin(a-b)`
/// - `cos(a)·cos(b) - sin(a)·sin(b) → cos(a+b)`
/// - `cos(a)·cos(b) + sin(a)·sin(b) → cos(a-b)`
fn tr10i(arena: &mut Arena, expr: ExprId) -> ExprId {
    walk_transform(arena, expr, |arena, id| {
        let node = arena.node(id).clone();
        if let ExprNode::Add(ref children) = node {
            // Try all pairs
            for i in 0..children.len() {
                for j in (i + 1)..children.len() {
                    // Pattern 1: sin(a)*cos(b) ± cos(a)*sin(b) → sin(a±b)
                    if let Some(result) = try_sin_addition_formula(arena, children[i], children[j])
                    {
                        return Some(replace_pair_in_add(arena, children, i, j, result));
                    }
                    // Pattern 2: cos(a)*cos(b) ∓ sin(a)*sin(b) → cos(a±b)
                    if let Some(result) = try_cos_addition_formula(arena, children[i], children[j])
                    {
                        return Some(replace_pair_in_add(arena, children, i, j, result));
                    }
                }
            }
        }
        None
    })
}

/// Try to match `c*sin(a)*cos(b) + c*cos(a)*sin(b)` → `c*sin(a+b)` or
/// `c*sin(a)*cos(b) - c*cos(a)*sin(b)` → `c*sin(a-b)`.
fn try_sin_addition_formula(arena: &mut Arena, term1: ExprId, term2: ExprId) -> Option<ExprId> {
    let p1 = extract_sin_cos_pair(arena, term1)?;
    let p2 = extract_sin_cos_pair(arena, term2)?;

    // sin(a)*cos(b) + cos(a)*sin(b) → sin(a+b) when c1 == c2
    // p1: sin(a)*cos(b), p2: sin(b)*cos(a) — so p1.sin_arg == p2.cos_arg and p1.cos_arg == p2.sin_arg
    if p1.sin_arg == p2.cos_arg && p1.cos_arg == p2.sin_arg {
        if p1.coeff == p2.coeff {
            // sin(a+b) with coefficient
            let sum = arena.add(&[p1.sin_arg, p1.cos_arg]);
            let sin_sum = arena.sin(sum);
            return Some(arena.make_coeff_term(p1.coeff, sin_sum));
        }
        if p1.coeff == -p2.coeff.clone() {
            // sin(a-b): c*sin(a)*cos(b) - c*cos(b)*sin(a) ... wait this is same args
            // Actually: c*sin(a)*cos(b) + (-c)*sin(b)*cos(a) → c*sin(a-b)
            let diff = arena.sub(p1.sin_arg, p1.cos_arg);
            let sin_diff = arena.sin(diff);
            return Some(arena.make_coeff_term(p1.coeff, sin_diff));
        }
    }

    // Also check the reverse: p1 has (sin=b, cos=a) and p2 has (sin=a, cos=b)
    if p1.cos_arg == p2.sin_arg && p1.sin_arg == p2.cos_arg && p1.coeff == p2.coeff {
        let sum = arena.add(&[p2.sin_arg, p2.cos_arg]);
        let sin_sum = arena.sin(sum);
        return Some(arena.make_coeff_term(p1.coeff, sin_sum));
    }

    None
}

/// Try to match `c*cos(a)*cos(b) - c*sin(a)*sin(b)` → `c*cos(a+b)`.
fn try_cos_addition_formula(arena: &mut Arena, term1: ExprId, term2: ExprId) -> Option<ExprId> {
    let cos_pair;
    let sin_pair;

    // One term should be cos*cos, the other sin*sin
    {
        let p = extract_same_type_pair(arena, term1)?;
        if !p.is_sin {
            cos_pair = p;
            sin_pair = extract_same_type_pair(arena, term2)?;
            if !sin_pair.is_sin {
                return None;
            }
        } else {
            sin_pair = p;
            let p2 = extract_same_type_pair(arena, term2)?;
            if p2.is_sin {
                return None;
            }
            cos_pair = p2;
        }
    }

    // Check argument matching: cos(a)*cos(b) and sin(a)*sin(b) with same a,b
    let args_match = (cos_pair.arg1 == sin_pair.arg1 && cos_pair.arg2 == sin_pair.arg2)
        || (cos_pair.arg1 == sin_pair.arg2 && cos_pair.arg2 == sin_pair.arg1);

    if !args_match {
        return None;
    }

    // cos(a)*cos(b) - sin(a)*sin(b) → cos(a+b): cos_coeff == -sin_coeff
    if cos_pair.coeff == -sin_pair.coeff.clone() {
        let sum = arena.add(&[cos_pair.arg1, cos_pair.arg2]);
        let cos_sum = arena.cos(sum);
        return Some(arena.make_coeff_term(cos_pair.coeff, cos_sum));
    }

    // cos(a)*cos(b) + sin(a)*sin(b) → cos(a-b): cos_coeff == sin_coeff
    if cos_pair.coeff == sin_pair.coeff {
        let diff = arena.sub(cos_pair.arg1, cos_pair.arg2);
        let cos_diff = arena.cos(diff);
        return Some(arena.make_coeff_term(cos_pair.coeff, cos_diff));
    }

    None
}

// ═══════════════════════════════════════════════════════════════════════════
// TR11: Double angle expansion (expand_trig)
// ═══════════════════════════════════════════════════════════════════════════

/// TR11: Double angle expansion — `sin(2x) → 2·sin(x)·cos(x)`, etc.
///
/// Delegates to the existing [`expand_trig`](crate::simplify::trig_expand) module.
fn tr11(arena: &mut Arena, expr: ExprId) -> ExprId {
    crate::simplify::trig_expand::expand_trig(arena, expr)
}

// ═══════════════════════════════════════════════════════════════════════════
// TR13: Reduce tan·cot products
// ═══════════════════════════════════════════════════════════════════════════

/// TR13: Reduce products involving tan and its reciprocal.
///
/// `tan(x)·tan(x)^(-1) → 1`
///
/// Also handles `sin(x)/cos(x) · cos(x)/sin(x) → 1` by detecting
/// matching positive and negative powers of `Tan`.
fn tr13(arena: &mut Arena, expr: ExprId) -> ExprId {
    walk_transform(arena, expr, |arena, id| {
        let node = arena.node(id).clone();
        if let ExprNode::Mul(ref children) = node {
            let mut tan_args: Vec<(usize, ExprId)> = Vec::new();
            let mut tan_inv_args: Vec<(usize, ExprId)> = Vec::new();

            for (idx, &child) in children.iter().enumerate() {
                match arena.node(child).clone() {
                    ExprNode::Tan(arg) => tan_args.push((idx, arg)),
                    ExprNode::Pow(base, exp) => {
                        if is_integer_val(arena, exp, -1)
                            && let ExprNode::Tan(arg) = arena.node(base).clone()
                        {
                            tan_inv_args.push((idx, arg));
                        }
                    }
                    _ => {}
                }
            }

            for &(ti, tan_arg) in &tan_args {
                for &(ii, inv_arg) in &tan_inv_args {
                    if tan_arg == inv_arg {
                        let remaining: Vec<ExprId> = children
                            .iter()
                            .enumerate()
                            .filter(|&(idx, _)| idx != ti && idx != ii)
                            .map(|(_, &c)| c)
                            .collect();
                        if remaining.is_empty() {
                            return Some(arena.one);
                        }
                        if remaining.len() == 1 {
                            return Some(remaining[0]);
                        }
                        return Some(arena.mul(&remaining));
                    }
                }
            }
        }
        None
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// TR14: Half-angle factoring
// ═══════════════════════════════════════════════════════════════════════════

/// TR14: Factor using half-angle identities.
///
/// - `cos(x) - 1 → -2·sin²(x/2)`
/// - `cos(x) + 1 →  2·cos²(x/2)`
pub(crate) fn tr14(arena: &mut Arena, expr: ExprId) -> ExprId {
    walk_transform(arena, expr, |arena, id| {
        let node = arena.node(id).clone();
        if let ExprNode::Add(ref children) = node {
            let mut cos_entries: Vec<(usize, ExprId, Q)> = Vec::new();
            let mut one_entry: Option<(usize, Q)> = None;

            for (idx, &child) in children.iter().enumerate() {
                let (coeff, term) = arena.as_coeff_term(child);
                if let ExprNode::Cos(arg) = arena.node(term).clone() {
                    cos_entries.push((idx, arg, coeff));
                } else if term == arena.one {
                    one_entry = Some((idx, coeff));
                }
            }

            if let Some((one_idx, ref one_coeff)) = one_entry {
                for &(cos_idx, arg, ref cos_coeff) in &cos_entries {
                    // cos(x) - 1: cos_coeff=1, one_coeff=-1
                    if *cos_coeff == Ratio::one() && *one_coeff == -Ratio::<BigInt>::one() {
                        let half = arena.rational(1, 2);
                        let half_x = arena.mul(&[half, arg]);
                        let sin_half = arena.sin(half_x);
                        let two = arena.int(2);
                        let sin_sq = arena.pow(sin_half, two);
                        let neg_two = arena.int(-2);
                        let result = arena.mul(&[neg_two, sin_sq]);
                        return Some(replace_pair_in_add(
                            arena, children, cos_idx, one_idx, result,
                        ));
                    }

                    // cos(x) + 1: cos_coeff=1, one_coeff=1
                    if *cos_coeff == Ratio::one() && *one_coeff == Ratio::one() {
                        let half = arena.rational(1, 2);
                        let half_x = arena.mul(&[half, arg]);
                        let cos_half = arena.cos(half_x);
                        let two = arena.int(2);
                        let cos_sq = arena.pow(cos_half, two);
                        let two_val = arena.int(2);
                        let result = arena.mul(&[two_val, cos_sq]);
                        return Some(replace_pair_in_add(
                            arena, children, cos_idx, one_idx, result,
                        ));
                    }
                }
            }
        }
        None
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// TRmorrie: Morrie's law
// ═══════════════════════════════════════════════════════════════════════════

/// TRmorrie: Morrie's law.
///
/// `cos(x)·cos(2x)·cos(4x)·...·cos(2^(n-1)x) → sin(2^n·x) / (2^n·sin(x))`
///
/// Detects products of cosines whose arguments form a geometric sequence
/// with ratio 2.
fn tr_morrie(arena: &mut Arena, expr: ExprId) -> ExprId {
    walk_transform(arena, expr, |arena, id| {
        let node = arena.node(id).clone();
        if let ExprNode::Mul(ref children) = node {
            let mut cos_factors: Vec<(usize, ExprId)> = Vec::new();

            for (idx, &child) in children.iter().enumerate() {
                if let ExprNode::Cos(arg) = arena.node(child).clone() {
                    cos_factors.push((idx, arg));
                }
            }

            if cos_factors.len() < 2 {
                return None;
            }

            // For each starting cos factor, try to build a Morrie chain
            for start_idx in 0..cos_factors.len() {
                let base_arg = cos_factors[start_idx].1;
                let mut chain = vec![cos_factors[start_idx]];
                let mut current_arg = base_arg;

                // Build chain: base_arg, 2*base_arg, 4*base_arg, ...
                for _attempt in 0..10 {
                    let two = arena.int(2);
                    let next_arg_raw = arena.mul(&[two, current_arg]);
                    let next_arg = crate::transforms::eval::eval(arena, next_arg_raw);

                    let mut found = false;
                    for &(idx, arg) in &cos_factors {
                        if chain.iter().any(|&(ci, _)| ci == idx) {
                            continue;
                        }
                        if arg == next_arg {
                            chain.push((idx, arg));
                            current_arg = arg;
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        break;
                    }
                }

                if chain.len() >= 2 {
                    let n = chain.len() as i64;
                    let two_pow_n = arena.int(1i64 << n);
                    let two_pow_n_x = arena.mul(&[two_pow_n, base_arg]);
                    let sin_top = arena.sin(two_pow_n_x);
                    let sin_bottom = arena.sin(base_arg);
                    let two_pow_n_2 = arena.int(1i64 << n);
                    let denom = arena.mul(&[two_pow_n_2, sin_bottom]);
                    let morrie_result = arena.div(sin_top, denom);

                    let used: Vec<usize> = chain.iter().map(|&(idx, _)| idx).collect();
                    let remaining: Vec<ExprId> = children
                        .iter()
                        .enumerate()
                        .filter(|&(idx, _)| !used.contains(&idx))
                        .map(|(_, &c)| c)
                        .collect();

                    if remaining.is_empty() {
                        return Some(morrie_result);
                    }
                    let mut all = remaining;
                    all.push(morrie_result);
                    return Some(arena.mul(&all));
                }
            }
        }
        None
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// TRpower: Power linearization
// ═══════════════════════════════════════════════════════════════════════════

/// TRpower: Linearize powers of sin/cos.
///
/// Uses the Chebyshev-type expansion to express `sin(x)^n` and `cos(x)^n`
/// as linear combinations of `sin(kx)` and `cos(kx)`.
pub(crate) fn tr_power(arena: &mut Arena, expr: ExprId) -> ExprId {
    walk_transform(arena, expr, |arena, id| {
        let node = arena.node(id).clone();
        if let ExprNode::Pow(base, exp) = node {
            let n_ratio = arena.as_num(exp).cloned()?;
            if !n_ratio.is_integer() {
                return None;
            }
            let n: i64 = n_ratio.to_integer().try_into().ok()?;
            if !(3..=12).contains(&n) {
                return None;
            }

            match arena.node(base).clone() {
                ExprNode::Sin(arg) => Some(linearize_sin_power(arena, arg, n as u32)),
                ExprNode::Cos(arg) => Some(linearize_cos_power(arena, arg, n as u32)),
                _ => None,
            }
        } else {
            None
        }
    })
}

/// Linearize `sin(x)^n` using binomial/Chebyshev expansion.
///
/// - Even n: `sin^n(x) = (1/2^n) * [C(n,n/2) + 2·Σ (-1)^(n/2-k)·C(n,k)·cos((n-2k)x)]`
/// - Odd n:  `sin^n(x) = (1/2^n) * [2·Σ (-1)^((n-1)/2-k)·C(n,k)·sin((n-2k)x)]`
pub(crate) fn linearize_sin_power(arena: &mut Arena, arg: ExprId, n: u32) -> ExprId {
    let denom = arena.big_int(BigInt::one() << n);
    let mut terms: Vec<ExprId> = Vec::new();

    if n.is_multiple_of(2) {
        let half_n = n / 2;
        let central = binomial_coeff(n, half_n);
        terms.push(arena.big_int(central));

        for k in 0..half_n {
            let binom = binomial_coeff(n, k);
            let sign = if (half_n - k).is_multiple_of(2) {
                1i64
            } else {
                -1i64
            };
            let coeff = arena.big_int(2 * sign * binom);
            let mult = arena.int((n - 2 * k) as i64);
            let mult_arg = arena.mul(&[mult, arg]);
            let cos_term = arena.cos(mult_arg);
            terms.push(arena.mul(&[coeff, cos_term]));
        }
    } else {
        let half_n = (n - 1) / 2;
        for k in 0..=half_n {
            let binom = binomial_coeff(n, k);
            let sign = if (half_n - k).is_multiple_of(2) {
                1i64
            } else {
                -1i64
            };
            let coeff = arena.big_int(2 * sign * binom);
            let mult = arena.int((n - 2 * k) as i64);
            let mult_arg = arena.mul(&[mult, arg]);
            let sin_term = arena.sin(mult_arg);
            terms.push(arena.mul(&[coeff, sin_term]));
        }
    }

    let sum = arena.add(&terms);
    arena.div(sum, denom)
}

/// Linearize `cos(x)^n` using binomial/Chebyshev expansion.
///
/// - Even n: `cos^n(x) = (1/2^n) * [C(n,n/2) + 2·Σ C(n,k)·cos((n-2k)x)]`
/// - Odd n:  `cos^n(x) = (1/2^n) * [2·Σ C(n,k)·cos((n-2k)x)]`
pub(crate) fn linearize_cos_power(arena: &mut Arena, arg: ExprId, n: u32) -> ExprId {
    let denom = arena.big_int(BigInt::one() << n);
    let mut terms: Vec<ExprId> = Vec::new();

    if n.is_multiple_of(2) {
        let half_n = n / 2;
        let central = binomial_coeff(n, half_n);
        terms.push(arena.big_int(central));

        for k in 0..half_n {
            let binom = binomial_coeff(n, k);
            let coeff = arena.big_int(2 * binom);
            let mult = arena.int((n - 2 * k) as i64);
            let mult_arg = arena.mul(&[mult, arg]);
            let cos_term = arena.cos(mult_arg);
            terms.push(arena.mul(&[coeff, cos_term]));
        }
    } else {
        let half_n = (n - 1) / 2;
        for k in 0..=half_n {
            let binom = binomial_coeff(n, k);
            let coeff = arena.big_int(2 * binom);
            let mult = arena.int((n - 2 * k) as i64);
            let mult_arg = arena.mul(&[mult, arg]);
            let cos_term = arena.cos(mult_arg);
            terms.push(arena.mul(&[coeff, cos_term]));
        }
    }

    let sum = arena.add(&terms);
    arena.div(sum, denom)
}

/// The binomial coefficient `C(n, k)` (`0` for `k > n`), exact for every
/// `n`: the shared [`combinatorics::binomial`](crate::base::combinatorics::binomial).
pub(crate) fn binomial_coeff(n: u32, k: u32) -> BigInt {
    crate::base::combinatorics::binomial(n, k)
}

// ═══════════════════════════════════════════════════════════════════════════
// Pythagorean substitutions
// ═══════════════════════════════════════════════════════════════════════════

/// Replace every `sin^(2k)(x)` with `(1-cos²(x))^k`, then expand + simplify.
///
/// This is the Pythagorean substitution strategy that complements the
/// standard TR transforms.
fn pyth_sub_sin2(arena: &mut Arena, expr: ExprId) -> ExprId {
    let replaced = walk_transform(arena, expr, |arena, id| {
        let node = arena.node(id).clone();
        if let ExprNode::Pow(base, exp) = node
            && let ExprNode::Sin(inner) = arena.node(base).clone()
            && let Some(n_ratio) = arena.as_num(exp).cloned()
            && n_ratio.is_integer()
            && let Ok(n) = TryInto::<i64>::try_into(n_ratio.to_integer())
            && n >= 2
            && n % 2 == 0
        {
            let k = n / 2;
            let cos_inner = arena.cos(inner);
            let two = arena.int(2);
            let cos_sq = arena.pow(cos_inner, two);
            let one = arena.one;
            let one_minus_cos_sq = arena.sub(one, cos_sq);
            let k_id = arena.int(k);
            return Some(arena.pow(one_minus_cos_sq, k_id));
        }
        None
    });
    let expanded = crate::transforms::expand::expand(arena, replaced);
    tr0(arena, expanded)
}

/// Replace every `cos^(2k)(x)` with `(1-sin²(x))^k`, then expand + simplify.
fn pyth_sub_cos2(arena: &mut Arena, expr: ExprId) -> ExprId {
    let replaced = walk_transform(arena, expr, |arena, id| {
        let node = arena.node(id).clone();
        if let ExprNode::Pow(base, exp) = node
            && let ExprNode::Cos(inner) = arena.node(base).clone()
            && let Some(n_ratio) = arena.as_num(exp).cloned()
            && n_ratio.is_integer()
            && let Ok(n) = TryInto::<i64>::try_into(n_ratio.to_integer())
            && n >= 2
            && n % 2 == 0
        {
            let k = n / 2;
            let sin_inner = arena.sin(inner);
            let two = arena.int(2);
            let sin_sq = arena.pow(sin_inner, two);
            let one = arena.one;
            let one_minus_sin_sq = arena.sub(one, sin_sq);
            let k_id = arena.int(k);
            return Some(arena.pow(one_minus_sin_sq, k_id));
        }
        None
    });
    let expanded = crate::transforms::expand::expand(arena, replaced);
    tr0(arena, expanded)
}

// ═══════════════════════════════════════════════════════════════════════════
// Composite transforms (CTR)
// ═══════════════════════════════════════════════════════════════════════════

/// CTR1: Try `(TR5, TR0)`, `(TR6, TR0)`, or identity — pick best.
fn ctr1(arena: &mut Arena, expr: ExprId) -> ExprId {
    let c1 = {
        let t = tr5(arena, expr);
        tr0(arena, t)
    };
    let c2 = {
        let t = tr6(arena, expr);
        tr0(arena, t)
    };
    pick_best(arena, &[expr, c1, c2])
}

/// CTR2: `TR11`, then try `(TR5, TR0)`, `(TR6, TR0)`, or `TR0` — pick best.
fn ctr2(arena: &mut Arena, expr: ExprId) -> ExprId {
    let expanded = tr11(arena, expr);
    let c1 = {
        let t = tr5(arena, expanded);
        tr0(arena, t)
    };
    let c2 = {
        let t = tr6(arena, expanded);
        tr0(arena, t)
    };
    let c3 = tr0(arena, expanded);
    pick_best(arena, &[expanded, c1, c2, c3])
}

/// CTR3: Try `(TRmorrie, TR8, TR0)`, `(TRmorrie, TR8, TR10i, TR0)`, or identity.
fn ctr3(arena: &mut Arena, expr: ExprId) -> ExprId {
    let c1 = {
        let t1 = tr_morrie(arena, expr);
        let t2 = tr8(arena, t1);
        tr0(arena, t2)
    };
    let c2 = {
        let t1 = tr_morrie(arena, expr);
        let t2 = tr8(arena, t1);
        let t3 = tr10i(arena, t2);
        tr0(arena, t3)
    };
    pick_best(arena, &[expr, c1, c2])
}

/// CTR4: Try `(TR3, TR10i)` or identity — pick best.
fn ctr4(arena: &mut Arena, expr: ExprId) -> ExprId {
    let c1 = {
        let t = tr3(arena, expr);
        tr10i(arena, t)
    };
    pick_best(arena, &[expr, c1])
}

// ═══════════════════════════════════════════════════════════════════════════
// Rule Lists
// ═══════════════════════════════════════════════════════════════════════════

/// RL1: Rule list for expressions with tan/cot.
///
/// `[TR3, TR3, TR13, TR3, TR0]` (simplified from the paper —
/// TR12 is not implemented, so we substitute eval passes).
fn rl1(arena: &mut Arena, expr: ExprId) -> ExprId {
    let mut r = tr3(arena, expr);
    r = tr3(arena, r);
    r = tr13(arena, r);
    r = tr3(arena, r);
    r = tr0(arena, r);
    r
}

/// RL2: Rule list for expressions with sin/cos.
///
/// Tries multiple alternative sequences and picks the best result by measure.
fn rl2(arena: &mut Arena, expr: ExprId) -> ExprId {
    // Alternative 1: [TR3, TR10i, TR3, TR11]
    let alt1 = {
        let mut r = tr3(arena, expr);
        r = tr3(arena, r);
        r = tr10i(arena, r);
        r = tr3(arena, r);
        r = tr3(arena, r);
        r = tr11(arena, r);
        tr0(arena, r)
    };

    // Alternative 2: [TR5, TR7, TR11, TR3]
    let alt2 = {
        let mut r = tr5(arena, expr);
        r = tr7(arena, r);
        r = tr11(arena, r);
        r = tr3(arena, r);
        tr0(arena, r)
    };

    // Alternative 3: [CTR3, CTR1, TR9, CTR2, TR3, TR9, TR9, CTR4]
    let alt3 = {
        let mut r = ctr3(arena, expr);
        r = ctr1(arena, r);
        r = tr9(arena, r);
        r = ctr2(arena, r);
        r = tr3(arena, r);
        r = tr9(arena, r);
        r = tr9(arena, r);
        ctr4(arena, r)
    };

    // Alternative 4: Pythagorean sub (sin² → 1-cos²) + expand + simplify
    let alt_pyth1 = {
        let expanded = tr11(arena, expr);
        pyth_sub_cos2(arena, expanded)
    };

    // Alternative 5: Pythagorean sub (cos² → 1-sin²) + expand + simplify
    let alt_pyth2 = {
        let expanded = tr11(arena, expr);
        pyth_sub_sin2(arena, expanded)
    };

    // Alternative 6: TR5i (inverse power-reduction)
    let alt_tr5i = {
        let r = tr5i(arena, expr);
        tr0(arena, r)
    };

    pick_best(
        arena,
        &[expr, alt1, alt2, alt3, alt_pyth1, alt_pyth2, alt_tr5i],
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// Main entry point
// ═══════════════════════════════════════════════════════════════════════════

/// Apply trig simplification using the algorithm from "Automated and
/// readable simplification of trigonometric expressions" (2006).
///
/// Orchestrates multiple named transforms (TR1–TR14, TRmorrie, TRpower)
/// organized into rule lists, using a greedy tree search to pick the best
/// result measured by `(trig_count, op_count)`.
pub(crate) fn fu(arena: &mut Arena, expr: ExprId) -> ExprId {
    tracing::debug!("fu: starting trig simplification");

    // Early exit: no trig nodes → nothing to do
    if !has_trig(arena, expr) {
        return expr;
    }

    // Step 1: Apply TR1 (remove sec/csc — already normalized in this system)
    let mut result = tr1(arena, expr);

    // Step 2: If expression has tan/cot, try RL1
    if has_tan(arena, result) {
        let rl1_result = rl1(arena, result);
        result = pick_best(arena, &[result, rl1_result]);
        tracing::trace!("fu: RL1 produced measure {:?}", measure(arena, result));
    }

    // Step 3: If still has tan, apply TR2 (convert to sin/cos)
    if has_tan(arena, result) {
        tracing::trace!("fu: applying TR2");
        let converted = tr2(arena, result);
        result = pick_best(arena, &[result, converted]);
    }

    // Step 4: If expression has sin/cos, try RL2 and additional transforms
    if has_sin_or_cos(arena, result) {
        let rl2_result = rl2(arena, result);

        // Also try TRmorrie + TR8
        tracing::trace!("fu: applying TRmorrie + TR8");
        let morrie_then_combine = {
            let t1 = tr_morrie(arena, result);
            let t2 = tr8(arena, t1);
            tr0(arena, t2)
        };

        result = pick_best(arena, &[result, rl2_result, morrie_then_combine]);
        tracing::trace!("fu: RL2 produced measure {:?}", measure(arena, result));
    }

    // Step 5: Try TR2i at the end (convert back to tan if simpler)
    let with_tan = tr2i(arena, result);
    result = pick_best(arena, &[result, with_tan]);
    tracing::trace!("fu: TR2i final measure {:?}", measure(arena, result));

    // Step 6: Try Pythagorean substitutions as additional strategies
    tracing::trace!("fu: applying Pythagorean substitutions");
    let pyth1 = pyth_sub_cos2(arena, result);
    let pyth2 = pyth_sub_sin2(arena, result);
    result = pick_best(arena, &[result, pyth1, pyth2]);

    // Step 7: Try TR5i (inverse power-reduction)
    tracing::trace!("fu: applying TR5i");
    let inv_pr = tr5i(arena, result);
    result = pick_best(arena, &[result, inv_pr]);

    // Step 8: Final normalization
    result = tr0(arena, result);

    // Step 9: Apply TR9 (sum-to-product) — prefer the product form when it
    // does not increase trig count, even if it has more operations.  This
    // matches the paper's intent that product forms are "simpler".
    let tr9_result = tr9(arena, result);
    let tr9_result = tr0(arena, tr9_result);
    if trig_count(arena, tr9_result) <= trig_count(arena, result) && tr9_result != result {
        result = tr9_result;
    }

    // Pick overall best vs the original with basic simplification.
    // Only fall back to the original when it has strictly fewer trig nodes;
    // otherwise keep the pipeline result (which may be a product form from
    // TR9 with the same trig count but more operations).
    let original_simplified = tr0(arena, expr);
    let orig_tc = trig_count(arena, original_simplified);
    let result_tc = trig_count(arena, result);
    if orig_tc < result_tc {
        original_simplified
    } else if orig_tc == result_tc {
        // Same trig count: prefer fewer ops, but if the pipeline produced a
        // structurally different (transformed) result, keep it.
        if result == original_simplified {
            result
        } else if count_ops(arena, original_simplified) < count_ops(arena, result)
            && result == tr0(arena, expr)
        {
            // Pipeline didn't actually transform anything; use original.
            original_simplified
        } else {
            result
        }
    } else {
        result
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(arena: &mut Arena, name: &str) -> ExprId {
        arena.symbol(name)
    }

    fn display(arena: &Arena, id: ExprId) -> String {
        arena.display(id).to_string()
    }

    /// Check numerical equivalence of two expressions at several points.
    fn assert_numerically_equal(arena: &mut Arena, e1: ExprId, e2: ExprId, var: ExprId) {
        // Substitute integer values and compare via evalf
        for val in [1i64, 2, 3] {
            let val_id = arena.int(val);
            let s1 = crate::transforms::subs::subs(arena, e1, var, val_id);
            let s2 = crate::transforms::subs::subs(arena, e2, var, val_id);
            let s1_eval = crate::transforms::eval::eval(arena, s1);
            let s2_eval = crate::transforms::eval::eval(arena, s2);

            let r1 = arena.evalf_expr(s1_eval, 15);
            let r2 = arena.evalf_expr(s2_eval, 15);
            match (r1, r2) {
                (Ok(v1), Ok(v2)) => {
                    let f1: f64 = v1.parse().unwrap_or(f64::NAN);
                    let f2: f64 = v2.parse().unwrap_or(f64::NAN);
                    let diff = (f1 - f2).abs();
                    let scale = f1.abs().max(f2.abs()).max(1.0);
                    assert!(
                        diff < 1e-9 * scale,
                        "Numerical mismatch at var={val}: {f1} vs {f2} (diff={diff})\n  e1={}\n  e2={}",
                        display(arena, e1),
                        display(arena, e2),
                    );
                }
                (Err(_), Err(_)) => {} // both fail at this point, skip
                (Ok(v), Err(e)) | (Err(e), Ok(v)) => {
                    // One succeeds, one fails — might be a singularity; skip
                    let _ = (v, e);
                }
            }
        }
    }

    // ── Measure tests ──────────────────────────────────────────────────

    #[test]
    fn trig_count_basic() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let sin_x = arena.sin(x);
        let cos_x = arena.cos(x);
        let expr = arena.add(&[sin_x, cos_x]);
        assert_eq!(trig_count(&arena, expr), 2);
    }

    #[test]
    fn trig_count_no_trig() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let two = arena.int(2);
        let expr = arena.add(&[x, two]);
        assert_eq!(trig_count(&arena, expr), 0);
    }

    #[test]
    fn measure_prefers_fewer_trig() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let one = arena.one;
        let sin_x = arena.sin(x);
        // one has measure (0, 0), sin(x) has measure (1, 1)
        assert!(measure(&arena, one) < measure(&arena, sin_x));
    }

    // ── TR2 tests ──────────────────────────────────────────────────────

    #[test]
    fn tr2_expands_tan() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let tan_x = arena.tan(x);
        let result = tr2(&mut arena, tan_x);
        let s = display(&arena, result);
        assert!(s.contains("sin"), "TR2 should produce sin: {s}");
        assert!(s.contains("cos"), "TR2 should produce cos: {s}");
        assert!(!s.contains("tan"), "TR2 should remove tan: {s}");
    }

    #[test]
    fn tr2_leaves_non_tan_alone() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let sin_x = arena.sin(x);
        let result = tr2(&mut arena, sin_x);
        assert_eq!(result, sin_x);
    }

    // ── TR2i tests ─────────────────────────────────────────────────────

    #[test]
    fn tr2i_contracts_sin_over_cos() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let sin_x = arena.sin(x);
        let cos_x = arena.cos(x);
        let expr = arena.div(sin_x, cos_x);
        let result = tr2i(&mut arena, expr);
        let s = display(&arena, result);
        assert!(s.contains("tan"), "TR2i should produce tan(x): {s}");
    }

    // ── TR5 tests ──────────────────────────────────────────────────────

    #[test]
    fn tr5_power_reduces_sin2() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let sin_x = arena.sin(x);
        let two = arena.int(2);
        let sin2 = arena.pow(sin_x, two);
        let original = sin2;
        let result = tr5(&mut arena, sin2);
        let s = display(&arena, result);
        // Should contain cos(2*x) in some form
        assert!(s.contains("cos"), "TR5 should produce cos(2x) form: {s}");
        assert_numerically_equal(&mut arena, original, result, x);
    }

    // ── TR6 tests ──────────────────────────────────────────────────────

    #[test]
    fn tr6_power_reduces_cos2() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let cos_x = arena.cos(x);
        let two = arena.int(2);
        let cos2 = arena.pow(cos_x, two);
        let original = cos2;
        let result = tr6(&mut arena, cos2);
        let s = display(&arena, result);
        assert!(s.contains("cos"), "TR6 should produce cos(2x) form: {s}");
        assert_numerically_equal(&mut arena, original, result, x);
    }

    // ── TR9 tests ──────────────────────────────────────────────────────

    #[test]
    fn tr9_sum_to_product_sin() {
        let mut arena = Arena::new();
        let a = sym(&mut arena, "a");
        let b = sym(&mut arena, "b");
        let sin_a = arena.sin(a);
        let sin_b = arena.sin(b);
        let expr = arena.add(&[sin_a, sin_b]);
        let result = tr9(&mut arena, expr);
        let s = display(&arena, result);
        // Should produce 2*sin(...)*cos(...) form
        assert!(
            s.contains("sin") && s.contains("cos"),
            "TR9 should produce product form: {s}"
        );
    }

    #[test]
    fn tr9_sum_to_product_cos() {
        let mut arena = Arena::new();
        let a = sym(&mut arena, "a");
        let b = sym(&mut arena, "b");
        let cos_a = arena.cos(a);
        let cos_b = arena.cos(b);
        let expr = arena.add(&[cos_a, cos_b]);
        let result = tr9(&mut arena, expr);
        let s = display(&arena, result);
        assert!(
            s.contains("cos"),
            "TR9 should produce product of cos form: {s}"
        );
    }

    // ── TR10i tests ────────────────────────────────────────────────────

    #[test]
    fn tr10i_sin_addition_formula() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let y = sym(&mut arena, "y");
        let sin_x = arena.sin(x);
        let cos_y = arena.cos(y);
        let cos_x = arena.cos(x);
        let sin_y = arena.sin(y);
        // sin(x)*cos(y) + cos(x)*sin(y) → sin(x+y)
        let term1 = arena.mul(&[sin_x, cos_y]);
        let term2 = arena.mul(&[cos_x, sin_y]);
        let expr = arena.add(&[term1, term2]);
        let result = tr10i(&mut arena, expr);
        let s = display(&arena, result);
        assert!(s.contains("sin"), "TR10i should produce sin(x+y): {s}");
        // The result should have fewer trig nodes
        assert!(
            trig_count(&arena, result) <= trig_count(&arena, expr),
            "TR10i should reduce trig count"
        );
    }

    // ── TR13 tests ─────────────────────────────────────────────────────

    #[test]
    fn tr13_tan_times_tan_inv() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let tan_x = arena.tan(x);
        let neg_one = arena.neg_one;
        let cot_x = arena.pow(tan_x, neg_one);
        let expr = arena.mul(&[tan_x, cot_x]);
        // Canon might already simplify this, but let's check TR13 too
        let result = tr13(&mut arena, expr);
        let ops = count_ops(&arena, result);
        // Should be simple (ideally 1 or very few ops)
        assert!(
            ops <= 2,
            "tan(x)*cot(x) should simplify to ~1: {}",
            display(&arena, result)
        );
    }

    // ── TRmorrie tests ─────────────────────────────────────────────────

    #[test]
    fn tr_morrie_three_cosines() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let two = arena.int(2);
        let four = arena.int(4);
        let cos_x = arena.cos(x);
        let two_x = arena.mul(&[two, x]);
        let cos_2x = arena.cos(two_x);
        let four_x = arena.mul(&[four, x]);
        let cos_4x = arena.cos(four_x);
        let expr = arena.mul(&[cos_x, cos_2x, cos_4x]);
        let original = expr;
        let result = tr_morrie(&mut arena, expr);
        let s = display(&arena, result);
        assert!(
            s.contains("sin"),
            "TRmorrie should produce sin(8x)/(8sin(x)): {s}"
        );
        assert_numerically_equal(&mut arena, original, result, x);
    }

    // ── TRpower tests ──────────────────────────────────────────────────

    #[test]
    fn tr_power_sin_cubed() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let sin_x = arena.sin(x);
        let three = arena.int(3);
        let sin3 = arena.pow(sin_x, three);
        let original = sin3;
        let result = tr_power(&mut arena, sin3);
        // Should be linearized (multiple sin terms)
        assert_ne!(result, sin3, "TRpower should transform sin³(x)");
        assert_numerically_equal(&mut arena, original, result, x);
    }

    // ── TR5i tests ─────────────────────────────────────────────────────

    #[test]
    fn tr5i_recognizes_half_minus_cos2x_half() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let two = arena.int(2);
        let two_x = arena.mul(&[two, x]);
        let cos_2x = arena.cos(two_x);
        let half = arena.rational(1, 2);
        let neg_half = arena.rational(-1, 2);
        let term1 = half; // 1/2
        let term2 = arena.mul(&[neg_half, cos_2x]); // -cos(2x)/2
        let expr = arena.add(&[term1, term2]);
        let original = expr;
        let result = tr5i(&mut arena, expr);
        let s = display(&arena, result);
        assert!(
            s.contains("sin") && s.contains("^2"),
            "TR5i should produce sin²(x): {s}"
        );
        assert_numerically_equal(&mut arena, original, result, x);
    }

    // ── Pythagorean substitution tests ─────────────────────────────────

    #[test]
    fn pyth_sub_sin2_replaces_sin_squared() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let sin_x = arena.sin(x);
        let two = arena.int(2);
        let sin2 = arena.pow(sin_x, two);
        let cos_x = arena.cos(x);
        let cos2 = arena.pow(cos_x, two);
        // sin²(x) + cos²(x) → should become 1
        let expr = arena.add(&[sin2, cos2]);
        let result = pyth_sub_sin2(&mut arena, expr);
        assert_eq!(
            display(&arena, result),
            "1",
            "sin²(x)+cos²(x) via Pythagorean sub should → 1"
        );
    }

    #[test]
    fn pyth_sub_cos2_replaces_cos_squared() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let sin_x = arena.sin(x);
        let two = arena.int(2);
        let sin2 = arena.pow(sin_x, two);
        let cos_x = arena.cos(x);
        let cos2 = arena.pow(cos_x, two);
        let expr = arena.add(&[sin2, cos2]);
        let result = pyth_sub_cos2(&mut arena, expr);
        assert_eq!(
            display(&arena, result),
            "1",
            "sin²(x)+cos²(x) via Pythagorean sub should → 1"
        );
    }

    // ── Main fu() tests ────────────────────────────────────────────────

    #[test]
    fn fu_pythagorean() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let sin_x = arena.sin(x);
        let cos_x = arena.cos(x);
        let two = arena.int(2);
        let sin2 = arena.pow(sin_x, two);
        let cos2 = arena.pow(cos_x, two);
        let expr = arena.add(&[sin2, cos2]);
        let result = fu(&mut arena, expr);
        assert_eq!(display(&arena, result), "1");
    }

    #[test]
    fn fu_sin_cos_to_tan() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let sin_x = arena.sin(x);
        let cos_x = arena.cos(x);
        let expr = arena.div(sin_x, cos_x);
        let result = fu(&mut arena, expr);
        let s = display(&arena, result);
        assert!(
            s.contains("tan"),
            "fu should simplify sin(x)/cos(x) to tan(x): {s}"
        );
        assert_numerically_equal(&mut arena, expr, result, x);
    }

    #[test]
    fn fu_double_angle() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let two = arena.int(2);
        let sin_x = arena.sin(x);
        let cos_x = arena.cos(x);
        let expr = arena.mul(&[two, sin_x, cos_x]);
        let original = expr;
        let result = fu(&mut arena, expr);
        let result_ops = count_ops(&arena, result);
        let expr_ops = count_ops(&arena, original);
        assert!(
            result_ops <= expr_ops,
            "fu should simplify 2sin(x)cos(x): got {} ops vs {} original, result={}",
            result_ops,
            expr_ops,
            display(&arena, result)
        );
        assert_numerically_equal(&mut arena, original, result, x);
    }

    #[test]
    fn fu_power_reduce() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let sin_x = arena.sin(x);
        let two = arena.int(2);
        let sin2 = arena.pow(sin_x, two);
        let original = sin2;
        let result = fu(&mut arena, sin2);
        // Should either be (1-cos(2x))/2 or remain sin²(x) — both valid
        // Just verify numerical equivalence
        assert_numerically_equal(&mut arena, original, result, x);
    }

    #[test]
    fn fu_preserves_simple() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let sin_x = arena.sin(x);
        let result = fu(&mut arena, sin_x);
        assert_eq!(
            display(&arena, result),
            "sin(x)",
            "fu should not transform simple sin(x)"
        );
    }

    #[test]
    fn fu_no_trig_passthrough() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let two = arena.int(2);
        let expr = arena.add(&[x, two]);
        let result = fu(&mut arena, expr);
        assert_eq!(result, expr, "fu should pass through non-trig expressions");
    }

    #[test]
    fn fu_half_minus_cos2x() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let two = arena.int(2);
        let two_x = arena.mul(&[two, x]);
        let cos_2x = arena.cos(two_x);
        let half = arena.rational(1, 2);
        let neg_half = arena.rational(-1, 2);
        let term2 = arena.mul(&[neg_half, cos_2x]);
        let expr = arena.add(&[half, term2]);
        let original = expr;
        let result = fu(&mut arena, expr);
        // Should be sin²(x) or numerically equivalent
        assert_numerically_equal(&mut arena, original, result, x);
        // Check measure improved or stayed same
        assert!(
            measure(&arena, result) <= measure(&arena, original),
            "fu should not make 1/2-cos(2x)/2 worse: result={}, original measure={:?}",
            display(&arena, result),
            measure(&arena, original)
        );
    }

    /// `sin(c)·cos(a)·cos(b) + sin(a)·sin(b)` must not collapse to `cos(a − b)`:
    /// TR10i used to pick the two trig factors out of a longer product and
    /// drop the rest.  Found by `Quaternion::from_rotation_matrix` on a
    /// numeric Euler rotation (`RᵀR` evaluated to `0.78`).
    #[test]
    fn tr10i_requires_exactly_two_factors() {
        let mut arena = Arena::new();
        let a = arena.rational(2, 5);
        let b = arena.rational(29, 10);
        let c = arena.rational(-6, 5);
        let sin_c = arena.sin(c);
        let cos_a = arena.cos(a);
        let cos_b = arena.cos(b);
        let sin_a = arena.sin(a);
        let sin_b = arena.sin(b);
        let t1 = arena.mul(&[sin_c, cos_a, cos_b]);
        let t2 = arena.mul(&[sin_a, sin_b]);
        let expr = arena.add(&[t1, t2]);
        let result = fu(&mut arena, expr);
        let before: f64 = arena.evalf_expr(expr, 15).unwrap().parse().unwrap();
        let after: f64 = arena.evalf_expr(result, 15).unwrap().parse().unwrap();
        assert!(
            (before - after).abs() < 1e-12,
            "fu changed the value: {before} -> {after} ({})",
            display(&arena, result)
        );
    }

    #[test]
    fn fu_morrie() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let two = arena.int(2);
        let four = arena.int(4);
        let cos_x = arena.cos(x);
        let two_x = arena.mul(&[two, x]);
        let cos_2x = arena.cos(two_x);
        let four_x = arena.mul(&[four, x]);
        let cos_4x = arena.cos(four_x);
        let expr = arena.mul(&[cos_x, cos_2x, cos_4x]);
        let original = expr;
        let result = fu(&mut arena, expr);
        let s = display(&arena, result);
        // Should contain sin (from Morrie's law: sin(8x)/(8sin(x)))
        assert!(s.contains("sin"), "fu Morrie should produce sin form: {s}");
        assert_numerically_equal(&mut arena, original, result, x);
    }

    #[test]
    fn fu_tan_cot_reduce() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let sin_x = arena.sin(x);
        let cos_x = arena.cos(x);
        // Build tan(x) * cot(x) = sin(x)/cos(x) * cos(x)/sin(x)
        let tan_x = arena.div(sin_x, cos_x);
        let cot_x = arena.div(cos_x, sin_x);
        let expr = arena.mul(&[tan_x, cot_x]);
        let result = fu(&mut arena, expr);
        let s = display(&arena, result);
        // Canonicalization or fu should reduce this to 1
        assert_eq!(s, "1", "tan(x)*cot(x) should simplify to 1: {s}");
    }

    #[test]
    fn fu_binomial_coeff_values() {
        let b = |n, k| binomial_coeff(n, k).to_string();
        assert_eq!(b(0, 0), "1");
        assert_eq!(b(4, 0), "1");
        assert_eq!(b(4, 1), "4");
        assert_eq!(b(4, 2), "6");
        assert_eq!(b(4, 3), "4");
        assert_eq!(b(4, 4), "1");
        assert_eq!(b(6, 3), "20");
        assert_eq!(b(3, 5), "0");
        // Beyond `i64`: the old implementation overflowed here.
        assert_eq!(b(70, 35), "112186277816662845432");
    }
}

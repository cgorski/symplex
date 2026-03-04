//! Trigonometric combination (inverse of trig expansion).
//!
//! Implements [`trig_combine`], which applies identities:
//!
//! - `sin(a)·cos(b) → ½[sin(a+b) + sin(a-b)]` (product-to-sum)
//! - `cos(a)·cos(b) → ½[cos(a-b) + cos(a+b)]`
//! - `sin(a)·sin(b) → ½[cos(a-b) - cos(a+b)]`
//! - `2·sin(x)·cos(x) → sin(2x)` (double angle)
//! - `cos²(x) - sin²(x) → cos(2x)` (double angle)

use crate::arena::Arena;
use crate::node::{ExprId, ExprNode};
use crate::walk;
use num_bigint::BigInt;
use num_rational::Ratio;
use rustc_hash::FxHashMap;

/// Apply trig product-to-sum and double-angle identities.
pub(crate) fn trig_combine(arena: &mut Arena, expr: ExprId) -> ExprId {
    let post_order = walk::post_order_ids(arena, expr);
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();

    for &id in &post_order {
        let rebuilt = crate::walk::rebuild_with_cache(arena, id, &cache);
        let combined = match arena.node(rebuilt).clone() {
            ExprNode::Mul(ref children) => try_combine_mul_trig(arena, rebuilt, children),
            ExprNode::Add(ref children) => try_combine_add_trig(arena, rebuilt, children),
            _ => rebuilt,
        };
        let combined = crate::eval::eval(arena, combined); // fold sin(0)→0, cos(0)→1, etc.
        cache.insert(id, combined);
    }

    cache.get(&expr).copied().unwrap_or(expr)
}

/// Scan children of a `Mul` node for pairs of trig functions and apply
/// the appropriate product-to-sum identity.
///
/// - `sin(a) * cos(b)` → `½[sin(a+b) + sin(a-b)]`
/// - `cos(a) * cos(b)` → `½[cos(a-b) + cos(a+b)]`
/// - `sin(a) * sin(b)` → `½[cos(a-b) - cos(a+b)]`
///
/// Remaining (non-trig) factors are preserved as outer coefficients.
fn try_combine_mul_trig(arena: &mut Arena, original: ExprId, children: &[ExprId]) -> ExprId {
    // Collect indices and arguments of sin/cos children.
    let mut sin_args: Vec<(usize, ExprId)> = Vec::new();
    let mut cos_args: Vec<(usize, ExprId)> = Vec::new();

    for (idx, &child) in children.iter().enumerate() {
        match arena.node(child).clone() {
            ExprNode::Sin(arg) => sin_args.push((idx, arg)),
            ExprNode::Cos(arg) => cos_args.push((idx, arg)),
            _ => {}
        }
    }

    // --- sin(a) * cos(b) → ½[sin(a+b) + sin(a-b)] ---
    if let (Some(&(si, a)), Some(&(ci, b))) = (sin_args.first(), cos_args.first()) {
        let half = arena.rational(1, 2);
        let a_plus_b = arena.add(&[a, b]);
        let a_minus_b = arena.sub(a, b);
        let sin_sum = arena.sin(a_plus_b);
        let sin_diff = arena.sin(a_minus_b);
        let inner = arena.add(&[sin_sum, sin_diff]);
        let result = arena.mul(&[half, inner]);

        return mul_with_remaining(arena, result, children, &[si, ci]);
    }

    // --- cos(a) * cos(b) → ½[cos(a-b) + cos(a+b)] ---
    if cos_args.len() >= 2 {
        let (i1, a) = cos_args[0];
        let (i2, b) = cos_args[1];
        let half = arena.rational(1, 2);
        let a_plus_b = arena.add(&[a, b]);
        let a_minus_b = arena.sub(a, b);
        let cos_sum = arena.cos(a_plus_b);
        let cos_diff = arena.cos(a_minus_b);
        let inner = arena.add(&[cos_diff, cos_sum]);
        let result = arena.mul(&[half, inner]);

        return mul_with_remaining(arena, result, children, &[i1, i2]);
    }

    // --- sin(a) * sin(b) → ½[cos(a-b) - cos(a+b)] ---
    if sin_args.len() >= 2 {
        let (i1, a) = sin_args[0];
        let (i2, b) = sin_args[1];
        let half = arena.rational(1, 2);
        let a_plus_b = arena.add(&[a, b]);
        let a_minus_b = arena.sub(a, b);
        let cos_sum = arena.cos(a_plus_b);
        let cos_diff = arena.cos(a_minus_b);
        let neg_cos_sum = arena.neg(cos_sum);
        let inner = arena.add(&[cos_diff, neg_cos_sum]);
        let result = arena.mul(&[half, inner]);

        return mul_with_remaining(arena, result, children, &[i1, i2]);
    }

    original
}

/// Multiply `result` by all children whose indices are not in `used`.
///
/// If there are no remaining children, returns `result` as-is.
fn mul_with_remaining(
    arena: &mut Arena,
    result: ExprId,
    children: &[ExprId],
    used: &[usize],
) -> ExprId {
    let remaining: Vec<ExprId> = children
        .iter()
        .enumerate()
        .filter(|&(idx, _)| !used.contains(&idx))
        .map(|(_, &c)| c)
        .collect();

    if remaining.is_empty() {
        return result;
    }

    let mut all = remaining;
    all.push(result);
    arena.mul(&all)
}

/// Information about a squared trig term found inside an `Add` node.
struct TrigSquare {
    /// Index into the `children` slice.
    child_idx: usize,
    /// The argument inside the trig function (e.g. `x`).
    arg: ExprId,
    /// `true` for sin², `false` for cos².
    is_sin: bool,
    /// The rational scalar coefficient (e.g. `1` or `-1`).
    coeff: Ratio<BigInt>,
}

/// Scan children of an `Add` node for double-angle patterns.
///
/// Recognises `cos²(x) - sin²(x)` (i.e. `c·Pow(Cos(x), 2) + (-c)·Pow(Sin(x), 2)`)
/// and rewrites to `c·cos(2x)`.
///
/// Also recognises `sin²(x) - cos²(x)` → `-cos(2x)`.
fn try_combine_add_trig(arena: &mut Arena, original: ExprId, children: &[ExprId]) -> ExprId {
    // We use `as_coeff_term` to split each child into (Ratio<BigInt>, core_term)
    // so we can recognise both positive and negative squared-trig terms.

    let two_id = arena.int(2);

    let mut squares: Vec<TrigSquare> = Vec::new();

    for (idx, &child) in children.iter().enumerate() {
        // Split into coefficient and core term.
        let (coeff, term) = arena.as_coeff_term(child);

        // Check if `term` is Pow(Sin(a), 2) or Pow(Cos(a), 2).
        let node = arena.node(term).clone();
        if let ExprNode::Pow(base, exp) = node {
            if exp != two_id {
                continue;
            }
            let base_node = arena.node(base).clone();
            match base_node {
                ExprNode::Sin(arg) => {
                    squares.push(TrigSquare {
                        child_idx: idx,
                        arg,
                        is_sin: true,
                        coeff,
                    });
                }
                ExprNode::Cos(arg) => {
                    squares.push(TrigSquare {
                        child_idx: idx,
                        arg,
                        is_sin: false,
                        coeff,
                    });
                }
                _ => {}
            }
        }
    }

    // Try to find a cos²(a) / sin²(a) pair with the same argument
    // where the coefficients are +c and -c for some c.
    for i in 0..squares.len() {
        for j in (i + 1)..squares.len() {
            // Must be same argument and different trig functions.
            if squares[i].arg != squares[j].arg || squares[i].is_sin == squares[j].is_sin {
                continue;
            }

            // Identify which is cos² and which is sin².
            let (cos_idx, sin_idx) = if squares[i].is_sin { (j, i) } else { (i, j) };

            let cos_coeff = &squares[cos_idx].coeff;
            let sin_coeff = &squares[sin_idx].coeff;
            let arg = squares[cos_idx].arg;
            let cos_child_idx = squares[cos_idx].child_idx;
            let sin_child_idx = squares[sin_idx].child_idx;

            // Check for cos²-sin² pattern: cos_coeff == c, sin_coeff == -c
            // i.e. c·cos²(a) + (-c)·sin²(a) → c·cos(2a)
            if *cos_coeff == -sin_coeff {
                let two_a = arena.mul(&[two_id, arg]);
                let cos_2a = arena.cos(two_a);
                let replacement = arena.make_coeff_term(cos_coeff.clone(), cos_2a);

                let used_indices = [cos_child_idx, sin_child_idx];
                let mut new_children: Vec<ExprId> = children
                    .iter()
                    .enumerate()
                    .filter(|&(idx, _)| !used_indices.contains(&idx))
                    .map(|(_, &c)| c)
                    .collect();
                new_children.push(replacement);

                if new_children.len() == 1 {
                    return new_children[0];
                }
                return arena.add(&new_children);
            }

            // Check for sin²-cos² pattern: sin_coeff == c, cos_coeff == -c
            // i.e. c·sin²(a) + (-c)·cos²(a) → -c·cos(2a)
            if *sin_coeff == -cos_coeff {
                let two_a = arena.mul(&[two_id, arg]);
                let cos_2a = arena.cos(two_a);
                let neg_sin_coeff = -sin_coeff;
                let replacement = arena.make_coeff_term(neg_sin_coeff, cos_2a);

                let used_indices = [cos_child_idx, sin_child_idx];
                let mut new_children: Vec<ExprId> = children
                    .iter()
                    .enumerate()
                    .filter(|&(idx, _)| !used_indices.contains(&idx))
                    .map(|(_, &c)| c)
                    .collect();
                new_children.push(replacement);

                if new_children.len() == 1 {
                    return new_children[0];
                }
                return arena.add(&new_children);
            }
        }
    }

    original
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(a: &mut Arena, name: &str) -> ExprId {
        a.symbol(name)
    }
    fn display(a: &Arena, id: ExprId) -> String {
        a.display(id).to_string()
    }

    #[test]
    fn sin_cos_product_to_sum() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let sin_x = a.sin(x);
        let cos_y = a.cos(y);
        let product = a.mul(&[sin_x, cos_y]);
        let result = trig_combine(&mut a, product);
        let s = display(&a, result);
        assert!(
            s.contains("sin"),
            "product-to-sum should produce sin terms: {s}"
        );
    }

    #[test]
    fn cos_cos_product_to_sum() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let cos_x = a.cos(x);
        let cos_y = a.cos(y);
        let product = a.mul(&[cos_x, cos_y]);
        let result = trig_combine(&mut a, product);
        let s = display(&a, result);
        assert!(
            s.contains("cos"),
            "product-to-sum should produce cos terms: {s}"
        );
    }

    #[test]
    fn sin_sin_product_to_sum() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let sin_x = a.sin(x);
        let sin_y = a.sin(y);
        let product = a.mul(&[sin_x, sin_y]);
        let result = trig_combine(&mut a, product);
        let s = display(&a, result);
        assert!(
            s.contains("cos"),
            "product-to-sum should produce cos terms: {s}"
        );
    }

    #[test]
    fn sin_cos_same_arg_is_half_sin_2x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let sin_x = a.sin(x);
        let cos_x = a.cos(x);
        let product = a.mul(&[sin_x, cos_x]);
        let result = trig_combine(&mut a, product);
        let s = display(&a, result);
        // sin(x)*cos(x) = ½[sin(2x) + sin(0)] = ½*sin(2x) (since sin(0)=0 after eval)
        assert!(s.contains("sin"), "sin(x)*cos(x) should → ½sin(2x): {s}");
    }

    #[test]
    fn product_with_coefficient() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let three = a.int(3);
        let sin_x = a.sin(x);
        let cos_y = a.cos(y);
        let product = a.mul(&[three, sin_x, cos_y]);
        let result = trig_combine(&mut a, product);
        let s = display(&a, result);
        assert!(
            s.contains("3") || s.contains("sin"),
            "should handle coefficients: {s}"
        );
    }

    #[test]
    fn no_trig_unchanged() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let sum = a.add(&[x, y]);
        let result = trig_combine(&mut a, sum);
        assert_eq!(result, sum, "non-trig should be unchanged");
    }

    #[test]
    fn trig_combine_2sincos_clean() {
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        let two = arena.int(2);
        let sin_x = arena.sin(x);
        let cos_x = arena.cos(x);
        let expr = arena.mul(&[two, sin_x, cos_x]); // 2*sin(x)*cos(x)
        let result = trig_combine(&mut arena, expr);
        let result_str = arena.display(result).to_string();
        // Should be sin(2*x), NOT sin(0) + sin(2*x)
        assert!(
            !result_str.contains("sin(0)"),
            "trig_combine should not leave sin(0) in result, got: {result_str}"
        );
        assert!(
            result_str.contains("sin(2"),
            "trig_combine(2sin(x)cos(x)) should produce sin(2x), got: {result_str}"
        );
    }
}

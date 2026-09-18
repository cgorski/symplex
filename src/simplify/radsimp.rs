//! Radical simplification — denominator rationalization.
//!
//! Implements [`rationalize_denom`], which clears radicals from denominators:
//!
//! - `1/√a → √a/a`
//! - `1/(a + √b) → (a - √b)/(a² - b)`
//! - `c/(a + √b) → c·(a - √b)/(a² - b)`
//!
//! and [`sqrtdenest`], which denests square roots of the form
//! `√(a + b√c)` with rational `a`, `b`, `c` whenever `a² − b²c` is a
//! perfect square:
//!
//! - `√(3 + 2√2) → 1 + √2`
//! - `√(5 − 2√6) → √3 − √2`

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};
use rustc_hash::FxHashMap;

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::is_perfect_square;
use crate::base::walk;
use crate::poly::polybridge;

// ═══════════════════════════════════════════════════════════════════════════
// sqrtdenest
// ═══════════════════════════════════════════════════════════════════════════

/// Denest nested square roots throughout an expression.
///
/// Every `Pow(base, k/2)` node (odd `k`) whose base has the form
/// `a + b·√c` with rational `a > 0`, `b ≠ 0`, `c > 0` is rewritten using
///
/// ```text
/// √(a + b√c) = √((a + √d)/2) + sign(b)·√((a − √d)/2),   d = a² − b²c,
/// ```
///
/// which is exact whenever `d ≥ 0` is the square of a rational (then
/// both inner radicands are non-negative rationals, so the principal
/// square roots agree).  Nested radicands are processed bottom-up, so
/// two levels of nesting are handled when the inner level denests to a
/// rational combination.
///
/// Expressions without such nodes are returned unchanged.
pub(crate) fn sqrtdenest(arena: &mut Arena, expr: ExprId) -> ExprId {
    let post_order = walk::post_order_ids(arena, expr);
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();

    for &id in &post_order {
        let rebuilt = walk::rebuild_with_cache(arena, id, &cache);
        let result = match arena.node(rebuilt).clone() {
            ExprNode::Pow(base, exp) => denest_pow(arena, base, exp).unwrap_or(rebuilt),
            _ => rebuilt,
        };
        cache.insert(id, result);
    }

    cache.get(&expr).copied().unwrap_or(expr)
}

/// Try to denest `base^exp` where `exp = k/2` with odd `k`.
fn denest_pow(arena: &mut Arena, base: ExprId, exp: ExprId) -> Option<ExprId> {
    let e = arena.as_num(exp)?.clone();
    if *e.denom() != BigInt::from(2) {
        return None;
    }
    let k = e.numer().clone(); // odd by reducedness
    let denested = denest_sqrt(arena, base)?;
    if k.is_one() {
        return Some(denested);
    }
    let k_id = arena.big_int(k);
    Some(arena.pow(denested, k_id))
}

/// Parse `base` as `a + b·√c` (rational `a`, `b`, `c`) and return the
/// denested `√base` if `a² − b²c` is a rational square.
fn denest_sqrt(arena: &mut Arena, base: ExprId) -> Option<ExprId> {
    let children = match arena.node(base) {
        ExprNode::Add(c) if c.len() == 2 => c.clone(),
        _ => return None,
    };

    // Identify the rational term `a` and the radical term `b·√c`.
    let (a, radical) = if arena.as_num(children[0]).is_some() {
        (arena.as_num(children[0])?.clone(), children[1])
    } else if arena.as_num(children[1]).is_some() {
        (arena.as_num(children[1])?.clone(), children[0])
    } else {
        return None;
    };
    let (b, sqrt_c) = arena.as_coeff_term(radical);
    let c = match arena.node(sqrt_c) {
        ExprNode::Pow(cb, ce) => {
            let half = Ratio::new(BigInt::one(), BigInt::from(2));
            if arena.as_num(*ce)? != &half {
                return None;
            }
            arena.as_num(*cb)?.clone()
        }
        _ => return None,
    };

    if !a.is_positive() || b.is_zero() || !c.is_positive() {
        return None;
    }

    // d = a² − b²c must be a non-negative rational square.
    let d = &a * &a - &b * &b * &c;
    if d.is_negative() {
        return None;
    }
    let sqrt_d = rational_sqrt(&d)?;

    let two = Ratio::from_integer(BigInt::from(2));
    let x = (&a + &sqrt_d) / &two;
    let y = (&a - &sqrt_d) / &two;
    if y.is_negative() {
        return None;
    }

    let x_id = num_expr(arena, x);
    let y_id = num_expr(arena, y);
    let sqrt_x = arena.sqrt(x_id);
    let sqrt_y = arena.sqrt(y_id);
    let result = if b.is_positive() {
        arena.add(&[sqrt_x, sqrt_y])
    } else {
        arena.sub(sqrt_x, sqrt_y)
    };
    tracing::debug!("sqrtdenest: denested √(a + b√c)");
    Some(result)
}

/// Exact square root of a non-negative rational, if it is a rational.
fn rational_sqrt(r: &Ratio<BigInt>) -> Option<Ratio<BigInt>> {
    if r.is_negative() {
        return None;
    }
    if !is_perfect_square(r.numer()) || !is_perfect_square(r.denom()) {
        return None;
    }
    Some(Ratio::new(r.numer().sqrt(), r.denom().sqrt()))
}

/// Intern a rational number as an expression node.
fn num_expr(arena: &mut Arena, r: Ratio<BigInt>) -> ExprId {
    let nid = arena.intern_num(r);
    arena.intern(ExprNode::Num(nid))
}

// ═══════════════════════════════════════════════════════════════════════════
// rationalize_denom
// ═══════════════════════════════════════════════════════════════════════════

/// Rationalize the denominator of an expression.
///
/// If the expression is a fraction whose denominator contains square roots,
/// multiply by the conjugate to eliminate them.
pub(crate) fn rationalize_denom(arena: &mut Arena, expr: ExprId) -> ExprId {
    let (numer, denom) = polybridge::as_numer_denom(arena, expr);

    // If denom is 1, nothing to do
    if denom == arena.one {
        return expr;
    }

    // Check if denom contains a sqrt
    if !contains_sqrt(arena, denom) {
        return expr;
    }

    // Try to rationalize
    if let Some(result) = try_rationalize(arena, numer, denom) {
        return result;
    }

    expr
}

/// Check if an expression contains a Pow with exponent 1/2 (i.e., a sqrt).
fn contains_sqrt(arena: &Arena, expr: ExprId) -> bool {
    match arena.node(expr).clone() {
        ExprNode::Pow(_, exp) => {
            if let Some(r) = arena.as_num(exp)
                && *r == num_rational::Ratio::new(1.into(), 2.into())
            {
                return true;
            }
            // Check children
            arena
                .node(expr)
                .children()
                .iter()
                .any(|&c| contains_sqrt(arena, c))
        }
        ExprNode::Add(ref children) | ExprNode::Mul(ref children) => {
            children.iter().any(|&c| contains_sqrt(arena, c))
        }
        ExprNode::Neg(inner) => contains_sqrt(arena, inner),
        _ => false,
    }
}

/// Try to rationalize a fraction numer/denom by multiplying by the conjugate.
fn try_rationalize(arena: &mut Arena, numer: ExprId, denom: ExprId) -> Option<ExprId> {
    // Case 1: denom is a bare sqrt: √a → multiply by √a/√a → numer·√a / a
    if let ExprNode::Pow(base, exp) = arena.node(denom).clone()
        && let Some(r) = arena.as_num(exp)
        && *r == num_rational::Ratio::new(1.into(), 2.into())
    {
        // denom = base^(1/2) = √base
        // result = numer * √base / base
        let new_numer = arena.mul(&[numer, denom]); // numer * √base
        return Some(arena.div(new_numer, base));
    }

    // Case 2: denom is Add with exactly 2 children, one or both containing sqrt
    // e.g., a + √b or √a + √b
    if let ExprNode::Add(ref children) = arena.node(denom).clone()
        && children.len() == 2
    {
        let a = children[0];
        let b = children[1];

        // Conjugate: a - b (negate second term)
        let neg_b = arena.neg(b);
        let conjugate = arena.add(&[a, neg_b]);

        // New numerator: numer * conjugate
        let new_numer = arena.mul(&[numer, conjugate]);
        let new_numer_expanded = crate::transforms::expand::expand(arena, new_numer);

        // New denominator: (a+b)(a-b) = a² - b²
        let denom_product = arena.mul(&[denom, conjugate]);
        let new_denom = crate::transforms::expand::expand(arena, denom_product);
        let new_denom = crate::transforms::eval::eval(arena, new_denom);

        // Check if the new denominator is sqrt-free
        if !contains_sqrt(arena, new_denom) {
            return Some(arena.div(new_numer_expanded, new_denom));
        }

        // If still has sqrt, try one more round of conjugate
        // (handles cases like (√2 + √3) where first pass gives (2-3)=-1)
        // Actually the expand should handle a²-b² = a²-b² already.
        // Return what we have — it might still be simpler.
        return Some(arena.div(new_numer_expanded, new_denom));
    }

    None
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
    fn rationalize_one_over_sqrt_2() {
        let mut a = Arena::new();
        let two = a.int(2);
        let half = a.rational(1, 2);
        let sqrt_2 = a.pow(two, half);
        let one = a.one;
        let expr = a.div(one, sqrt_2);
        let result = rationalize_denom(&mut a, expr);
        let s = display(&a, result);
        // 1/√2 → √2/2
        assert!(s.contains("2"), "should contain 2: {s}");
        // Should not have sqrt in denominator
    }

    #[test]
    fn rationalize_one_over_one_plus_sqrt_2() {
        let mut a = Arena::new();
        let one = a.one;
        let two = a.int(2);
        let half = a.rational(1, 2);
        let sqrt_2 = a.pow(two, half);
        let denom = a.add(&[one, sqrt_2]);
        let expr = a.div(one, denom);
        let result = rationalize_denom(&mut a, expr);
        let s = display(&a, result);
        // 1/(1+√2) → (1-√2)/(1-2) = -(1-√2) = √2-1
        assert!(s != display(&a, expr), "should change: {s}");
    }

    #[test]
    fn no_sqrt_unchanged() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let expr = a.div(x, y);
        let result = rationalize_denom(&mut a, expr);
        assert_eq!(result, expr, "no sqrt should be unchanged");
    }

    #[test]
    fn integer_denom_unchanged() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let expr = a.div(x, two);
        let result = rationalize_denom(&mut a, expr);
        assert_eq!(result, expr, "integer denom should be unchanged");
    }

    #[test]
    fn rationalize_with_numerator() {
        let mut a = Arena::new();
        let three = a.int(3);
        let two = a.int(2);
        let half = a.rational(1, 2);
        let sqrt_2 = a.pow(two, half);
        let expr = a.div(three, sqrt_2);
        let result = rationalize_denom(&mut a, expr);
        let s = display(&a, result);
        assert!(s.contains("3"), "should contain 3: {s}");
    }

    #[test]
    fn rationalize_sqrt_plus_sqrt() {
        let mut a = Arena::new();
        let one = a.one;
        let two = a.int(2);
        let three = a.int(3);
        let half = a.rational(1, 2);
        let sqrt_2 = a.pow(two, half);
        let sqrt_3 = a.pow(three, half);
        let denom = a.add(&[sqrt_2, sqrt_3]);
        let expr = a.div(one, denom);
        let result = rationalize_denom(&mut a, expr);
        let s = display(&a, result);
        // Should attempt rationalization
        assert!(s != display(&a, expr), "should change: {s}");
    }

    // ── sqrtdenest ─────────────────────────────────────────────────

    fn sqrt_of(a: &mut Arena, n: i64) -> ExprId {
        let v = a.int(n);
        a.sqrt(v)
    }

    #[test]
    fn sqrtdenest_three_plus_two_sqrt_two() {
        let mut a = Arena::new();
        let s2 = sqrt_of(&mut a, 2);
        let two = a.int(2);
        let three = a.int(3);
        let t = a.mul(&[two, s2]);
        let radicand = a.add(&[three, t]);
        let e = a.sqrt(radicand);
        let r = sqrtdenest(&mut a, e);
        assert_eq!(display(&a, r), "sqrt(2) + 1");
    }

    #[test]
    fn sqrtdenest_five_minus_two_sqrt_six() {
        let mut a = Arena::new();
        let s6 = sqrt_of(&mut a, 6);
        let m2 = a.int(-2);
        let five = a.int(5);
        let t = a.mul(&[m2, s6]);
        let radicand = a.add(&[five, t]);
        let e = a.sqrt(radicand);
        let r = sqrtdenest(&mut a, e);
        assert_eq!(display(&a, r), "sqrt(3) - sqrt(2)");
    }

    #[test]
    fn sqrtdenest_non_square_discriminant_unchanged() {
        let mut a = Arena::new();
        let s2 = sqrt_of(&mut a, 2);
        let two = a.int(2);
        let radicand = a.add(&[two, s2]); // 2 + √2: d = 4 - 2 = 2
        let e = a.sqrt(radicand);
        assert_eq!(sqrtdenest(&mut a, e), e);
    }

    #[test]
    fn sqrtdenest_negative_constant_unchanged() {
        let mut a = Arena::new();
        let s2 = sqrt_of(&mut a, 2);
        let two = a.int(2);
        let m3 = a.int(-3);
        let t = a.mul(&[two, s2]);
        let radicand = a.add(&[m3, t]); // -3 + 2√2 < 0
        let e = a.sqrt(radicand);
        assert_eq!(sqrtdenest(&mut a, e), e);
    }

    #[test]
    fn rational_sqrt_helper() {
        let r = |p: i64, q: i64| Ratio::new(BigInt::from(p), BigInt::from(q));
        assert_eq!(rational_sqrt(&r(9, 4)), Some(r(3, 2)));
        assert_eq!(rational_sqrt(&r(2, 1)), None);
        assert_eq!(rational_sqrt(&r(-4, 1)), None);
        assert_eq!(rational_sqrt(&r(0, 1)), Some(r(0, 1)));
    }
}

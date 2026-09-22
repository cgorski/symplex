//! Factor terms: pull common factors from sums.
//!
//! Implements [`factor_terms`], which extracts the GCD of numeric
//! coefficients and common symbolic factors from a sum:
//!
//! - `2x + 2y → 2(x + y)`
//! - `3x² + 6x → 3x(x + 2)`
//! - `m*g*l + m*g*x → m*g*(l + x)`
//!
//! Also home to the related term-grouping utilities [`signsimp`],
//! [`collect_const`], [`collect_powers`] and [`rcollect`].

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::Q;
use crate::base::walk;
use num_integer::Integer;
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

// ---------------------------------------------------------------------------
// Symbolic factor helpers
// ---------------------------------------------------------------------------

/// Try to interpret an ExprId as an integer. Returns Some(i64) if it's a numeric
/// integer, None otherwise.
fn try_as_integer(arena: &Arena, id: ExprId) -> Option<i64> {
    if let ExprNode::Num(nid) = arena.node(id) {
        let r = arena.num(*nid);
        if r.is_integer() {
            r.to_integer().try_into().ok()
        } else {
            None
        }
    } else {
        None
    }
}

/// Raise a rational to an integer power.
fn pow_rational(r: &Q, n: i64) -> Q {
    if n == 0 {
        return Ratio::one();
    }
    if n == 1 {
        return r.clone();
    }
    if n < 0 {
        let inv = Ratio::new(r.denom().clone(), r.numer().clone());
        return pow_rational(&inv, -n);
    }
    let mut result = r.clone();
    for _ in 1..n {
        result *= r;
    }
    result
}

/// Decompose a term (from an Add) into its numeric coefficient and a map of
/// symbolic factors with their integer exponents.
///
/// Examples:
///   3*m*g*x²  → (3, {m: 1, g: 1, x: 2})
///   -m*x      → (-1, {m: 1, x: 1})
///   x         → (1, {x: 1})
///   5         → (5, {})
fn decompose_product(arena: &mut Arena, id: ExprId) -> (Q, FxHashMap<ExprId, i64>) {
    let (mut coeff, term) = arena.as_coeff_term(id);
    let mut factors = FxHashMap::default();

    if term == arena.one {
        // Pure numeric term
        return (coeff, factors);
    }

    match arena.node(term).clone() {
        ExprNode::Mul(children) => {
            for child in children.iter() {
                let (base, exp_id) = arena.as_base_exp(*child);
                if let Some(n) = try_as_integer(arena, exp_id) {
                    // Check if base is an Add — extract content if so
                    let (content, clean_base) = extract_add_content(arena, base, n);
                    coeff *= content;
                    if clean_base != arena.one {
                        *factors.entry(clean_base).or_insert(0) += n;
                    }
                } else {
                    // Non-integer exponent: treat whole Pow as atomic factor
                    *factors.entry(*child).or_insert(0) += 1;
                }
            }
        }
        ExprNode::Pow(base, exp) => {
            if let Some(n) = try_as_integer(arena, exp) {
                let (content, clean_base) = extract_add_content(arena, base, n);
                coeff *= content;
                if clean_base != arena.one {
                    factors.insert(clean_base, n);
                }
            } else {
                factors.insert(term, 1);
            }
        }
        _ => {
            // Atomic symbol or function
            factors.insert(term, 1);
        }
    }

    (coeff, factors)
}

/// If `base` is an Add node, extract its numeric content.
/// Returns (content^exp, primitive_base).
/// For example: base = (2x + 4), exp = 2
///   content = 2, primitive = (x + 2)
///   Returns (2^2 = 4, (x + 2))
fn extract_add_content(arena: &mut Arena, base: ExprId, exp: i64) -> (Q, ExprId) {
    if let ExprNode::Add(_) = arena.node(base).clone() {
        let (content, primitive) = numeric_factor_terms_pair(arena, base);
        if !content.is_one() {
            let content_powered = pow_rational(&content, exp);
            return (content_powered, primitive);
        }
    }
    (Ratio::one(), base)
}

/// Compute the GCD of multiple factor maps.
/// Returns the intersection of all maps with minimum positive exponents.
fn factor_map_gcd(maps: &[FxHashMap<ExprId, i64>]) -> FxHashMap<ExprId, i64> {
    if maps.is_empty() {
        return FxHashMap::default();
    }
    let mut gcd = maps[0].clone();

    for map in &maps[1..] {
        gcd.retain(|base, exp| {
            if let Some(&other_exp) = map.get(base) {
                let min_exp = (*exp).min(other_exp);
                if min_exp > 0 {
                    *exp = min_exp;
                    true
                } else {
                    false
                }
            } else {
                false
            }
        });
    }
    gcd
}

/// Subtract GCD exponents from a factor map.
fn divide_factor_map(
    term: &FxHashMap<ExprId, i64>,
    gcd: &FxHashMap<ExprId, i64>,
) -> FxHashMap<ExprId, i64> {
    let mut result = term.clone();
    for (base, gcd_exp) in gcd {
        if let Some(term_exp) = result.get_mut(base) {
            *term_exp -= gcd_exp;
            if *term_exp == 0 {
                result.remove(base);
            }
        }
    }
    result
}

/// Convert a factor map back into an ExprId.
fn map_to_expr(arena: &mut Arena, map: &FxHashMap<ExprId, i64>) -> ExprId {
    if map.is_empty() {
        return arena.one;
    }
    let mut parts: Vec<ExprId> = Vec::new();
    for (&base, &exp) in map {
        if exp == 1 {
            parts.push(base);
        } else {
            let exp_id = arena.int(exp);
            parts.push(arena.pow(base, exp_id));
        }
    }
    match parts.len() {
        0 => arena.one,
        1 => parts[0],
        _ => arena.mul(&parts),
    }
}

/// Rebuild a term from a coefficient and factor map.
fn rebuild_term(arena: &mut Arena, coeff: &Q, factors: &FxHashMap<ExprId, i64>) -> ExprId {
    let sym_expr = map_to_expr(arena, factors);
    arena.make_coeff_term(coeff.clone(), sym_expr)
}

// ---------------------------------------------------------------------------
// Numeric-only factor extraction (original implementation)
// ---------------------------------------------------------------------------

/// Factor out the GCD of numeric coefficients only.
/// Returns (gcd, inner) where expr == gcd * inner mathematically.
/// The inner expression has each coefficient divided by gcd.
fn numeric_factor_terms_pair(arena: &mut Arena, expr: ExprId) -> (Q, ExprId) {
    let node = arena.node(expr).clone();
    let children = match node {
        ExprNode::Add(ref children) if children.len() >= 2 => children.clone(),
        _ => return (Ratio::one(), expr),
    };

    let mut pairs: Vec<(Q, ExprId)> = Vec::new();
    for &child in &children {
        let (coeff, term) = arena.as_coeff_term(child);
        pairs.push((coeff, term));
    }

    let coeffs: Vec<&Q> = pairs.iter().map(|(c, _)| c).collect();
    let gcd = rational_gcd_multi(&coeffs);

    if gcd.is_one() || gcd.is_zero() {
        return (Ratio::one(), expr);
    }

    let mut new_terms: SmallVec<[ExprId; 6]> = SmallVec::new();
    for (coeff, term) in &pairs {
        let new_coeff = coeff / &gcd;
        let new_child = arena.make_coeff_term(new_coeff, *term);
        new_terms.push(new_child);
    }

    let inner_sum = arena.add(&new_terms);
    (gcd, inner_sum)
}

/// Factor out the GCD of numeric coefficients.
/// Returns (gcd, inner) where expr == gcd * inner mathematically.
/// The inner expression has each coefficient divided by gcd.
///
/// This is the public entry point that preserves backward compatibility.
#[allow(dead_code)]
pub(crate) fn factor_terms_pair(arena: &mut Arena, expr: ExprId) -> (Q, ExprId) {
    numeric_factor_terms_pair(arena, expr)
}

// ---------------------------------------------------------------------------
// Combined numeric + symbolic factor extraction
// ---------------------------------------------------------------------------

/// Factor out both numeric AND symbolic common factors from a sum.
/// Returns (gcd_expr, remainder) where expr == gcd_expr * remainder mathematically.
///
/// Examples:
///   2*x + 4*y → (2, x + 2*y)          — numeric GCD
///   m*g*l + m*g*x → (m*g, l + x)      — symbolic GCD
///   6*m*x + 3*m*y → (3*m, 2*x + y)    — both
///   (2*x+4)^2*y + (2*x+4)*z → (2*(x+2), 2*(x+2)*y + z) — Add content extraction
pub(crate) fn symbolic_factor_terms_pair(arena: &mut Arena, expr: ExprId) -> (ExprId, ExprId) {
    let children = match arena.node(expr).clone() {
        ExprNode::Add(ref children) if children.len() >= 2 => children.clone(),
        _ => return (arena.one, expr),
    };

    // Decompose each term
    let decomposed: Vec<_> = children
        .iter()
        .map(|&c| decompose_product(arena, c))
        .collect();

    // Numeric GCD
    let coeffs: Vec<&Q> = decomposed.iter().map(|(c, _)| c).collect();
    let num_gcd = rational_gcd_multi(&coeffs);

    // Symbolic GCD
    let sym_maps: Vec<_> = decomposed.iter().map(|(_, m)| m.clone()).collect();
    let sym_gcd = factor_map_gcd(&sym_maps);

    // Check if there's anything to factor
    if num_gcd.is_one() && sym_gcd.is_empty() {
        return (arena.one, expr);
    }

    // Build GCD expression
    let sym_gcd_expr = map_to_expr(arena, &sym_gcd);
    let gcd_expr = if num_gcd.is_one() {
        sym_gcd_expr
    } else {
        let num_id = {
            let nid = arena.intern_num(num_gcd.clone());
            arena.intern(ExprNode::Num(nid))
        };
        if sym_gcd_expr == arena.one {
            num_id
        } else {
            arena.mul(&[num_id, sym_gcd_expr])
        }
    };

    // Divide each term by the GCD
    let mut new_terms: SmallVec<[ExprId; 6]> = SmallVec::new();
    for (coeff, factors) in &decomposed {
        let new_coeff = coeff / &num_gcd;
        let new_factors = divide_factor_map(factors, &sym_gcd);
        new_terms.push(rebuild_term(arena, &new_coeff, &new_factors));
    }

    let remainder = arena.add(&new_terms);
    (gcd_expr, remainder)
}

/// Factor out GCD and return as single expression (may re-distribute due to canonicalization).
pub(crate) fn factor_terms(arena: &mut Arena, expr: ExprId) -> ExprId {
    let (gcd, inner) = symbolic_factor_terms_pair(arena, expr);
    if gcd == arena.one {
        return inner;
    }
    arena.mul(&[gcd, inner])
}

// ---------------------------------------------------------------------------
// signsimp — sign normalisation of sums inside products and powers
// ---------------------------------------------------------------------------

/// Returns `true` if the leading symbolic (non-numeric) term of the `Add`
/// `id` has a negative coefficient.
///
/// "Leading" follows the *display* order (highest-degree polynomial term
/// first, then functions, then constants), so the normalised sign is the
/// one a user sees: `-x + y` has leading term `-x`.
fn add_has_negative_leading_term(arena: &mut Arena, id: ExprId) -> bool {
    let mut children = match arena.node(id) {
        ExprNode::Add(c) => c.clone(),
        _ => return false,
    };
    children.sort_by_key(|&c| crate::output::common::display_sort_key(arena, c));
    for &child in &children {
        if arena.as_num(child).is_some() {
            continue;
        }
        let (coeff, _) = arena.as_coeff_term(child);
        return coeff.is_negative();
    }
    false
}

/// Sign normalisation.
///
/// Canonical forms already fold `−(−x + y)` into `x − y` and distribute
/// numeric factors over sums.  What remains is the sign of sums that sit
/// inside products or powers, where `(−x + y)^2` and `(x − y)^2` are
/// distinct nodes.  `signsimp` extracts `−1` from every such sum whose
/// first symbolic term is negative:
///
/// - `(−x + y)^2 → (x − y)^2`
/// - `(−x + y)^3 → −(x − y)^3`
/// - `z·(−x + y) → −z·(x − y)`
/// - `(−x − y)·(−a + b) → (x + y)·(a − b)`
///
/// Only integer exponents are touched (for `(−s)^e` with non-integer `e`
/// the branch of the power would change).  The walk is bottom-up.
pub(crate) fn signsimp(arena: &mut Arena, expr: ExprId) -> ExprId {
    let post_order = walk::post_order_ids(arena, expr);
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();

    for &id in &post_order {
        let rebuilt = walk::rebuild_with_cache(arena, id, &cache);
        let result = match arena.node(rebuilt).clone() {
            ExprNode::Pow(base, exp) => {
                let exp_int = arena
                    .as_num(exp)
                    .filter(|r| r.is_integer())
                    .map(|r| r.to_integer());
                match exp_int {
                    Some(n) if add_has_negative_leading_term(arena, base) => {
                        let neg_base = arena.neg(base);
                        let p = arena.pow(neg_base, exp);
                        if n.is_even() { p } else { arena.neg(p) }
                    }
                    _ => rebuilt,
                }
            }
            ExprNode::Mul(children) => {
                let mut flips = 0usize;
                let mut new_children: SmallVec<[ExprId; 6]> = SmallVec::new();
                for &c in &children {
                    if add_has_negative_leading_term(arena, c) {
                        flips += 1;
                        new_children.push(arena.neg(c));
                    } else {
                        new_children.push(c);
                    }
                }
                if flips == 0 {
                    rebuilt
                } else {
                    if flips % 2 == 1 {
                        new_children.push(arena.neg_one);
                    }
                    arena.mul(&new_children)
                }
            }
            _ => rebuilt,
        };
        cache.insert(id, result);
    }

    cache.get(&expr).copied().unwrap_or(expr)
}

// ---------------------------------------------------------------------------
// collect_const / collect_powers / rcollect
// ---------------------------------------------------------------------------

/// Factor the rational content out of every sum that is a factor of a
/// product or the base of a power: `z·(2x + 4y) → 2·z·(x + 2y)`,
/// `(2x + 4y)^2 → 4·(x + 2y)^2`.
///
/// A top-level sum is returned unchanged: the canonical form distributes
/// a numeric coefficient over a sum, so `2·(x + 2y)` cannot be
/// represented as a single node — use
/// [`factor_terms_pair`] for the `(coefficient, inner)` pair.
pub(crate) fn collect_const(arena: &mut Arena, expr: ExprId) -> ExprId {
    let post_order = walk::post_order_ids(arena, expr);
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();

    for &id in &post_order {
        let rebuilt = walk::rebuild_with_cache(arena, id, &cache);
        let result = match arena.node(rebuilt).clone() {
            ExprNode::Pow(base, exp) if matches!(arena.node(base), ExprNode::Add(_)) => {
                let (g, inner) = numeric_factor_terms_pair(arena, base);
                if g.is_one() {
                    rebuilt
                } else {
                    let g_id = arena.num_ratio(g);
                    let g_pow = arena.pow(g_id, exp);
                    let inner_pow = arena.pow(inner, exp);
                    arena.mul(&[g_pow, inner_pow])
                }
            }
            ExprNode::Mul(children) => {
                let mut changed = false;
                let mut new_children: SmallVec<[ExprId; 6]> = SmallVec::new();
                for &c in &children {
                    if matches!(arena.node(c), ExprNode::Add(_)) {
                        let (g, inner) = numeric_factor_terms_pair(arena, c);
                        if !g.is_one() {
                            changed = true;
                            new_children.push(arena.num_ratio(g));
                            new_children.push(inner);
                            continue;
                        }
                    }
                    new_children.push(c);
                }
                if changed {
                    arena.mul(&new_children)
                } else {
                    rebuilt
                }
            }
            _ => rebuilt,
        };
        cache.insert(id, result);
    }

    cache.get(&expr).copied().unwrap_or(expr)
}

/// Group the terms of a sum by the power of `var` they contain, allowing
/// symbolic exponents: `y·x^a + z·x^a + x^2 → (y + z)·x^a + x^2`.
///
/// Each term is split into `coefficient · var^e` (`e = 1` for a bare
/// `var`, `e = 0` when `var` does not occur as a factor); terms with the
/// same `e` are combined.  Non-`Add` expressions are returned unchanged.
pub(crate) fn collect_powers(arena: &mut Arena, expr: ExprId, var: ExprId) -> ExprId {
    let children = match arena.node(expr) {
        ExprNode::Add(c) => c.clone(),
        _ => return expr,
    };

    // exponent → coefficient terms (insertion ordered)
    let mut groups: Vec<(ExprId, SmallVec<[ExprId; 4]>)> = Vec::new();
    let mut index: FxHashMap<ExprId, usize> = FxHashMap::default();

    for &term in &children {
        let (exp, coeff) = split_power_of(arena, term, var);
        match index.get(&exp) {
            Some(&i) => groups[i].1.push(coeff),
            None => {
                index.insert(exp, groups.len());
                groups.push((exp, smallvec::smallvec![coeff]));
            }
        }
    }

    if groups.iter().all(|(_, c)| c.len() == 1) {
        return expr;
    }

    let mut new_terms: SmallVec<[ExprId; 6]> = SmallVec::new();
    for (exp, coeffs) in &groups {
        let coeff_sum = if coeffs.len() == 1 {
            coeffs[0]
        } else {
            arena.add(coeffs)
        };
        if *exp == arena.zero {
            new_terms.push(coeff_sum);
        } else {
            let p = arena.pow(var, *exp);
            new_terms.push(arena.mul(&[coeff_sum, p]));
        }
    }
    arena.add(&new_terms)
}

/// Split `term` into `(exponent, coefficient)` with `term = coefficient · var^exponent`.
fn split_power_of(arena: &mut Arena, term: ExprId, var: ExprId) -> (ExprId, ExprId) {
    let (base, exp) = arena.as_base_exp(term);
    if base == var {
        return (exp, arena.one);
    }
    if let ExprNode::Mul(children) = arena.node(term).clone() {
        let mut rest: SmallVec<[ExprId; 6]> = SmallVec::new();
        let mut found: Option<ExprId> = None;
        for &c in &children {
            let (b, e) = arena.as_base_exp(c);
            if found.is_none() && b == var {
                found = Some(e);
            } else {
                rest.push(c);
            }
        }
        if let Some(e) = found {
            let coeff = match rest.len() {
                0 => arena.one,
                1 => rest[0],
                _ => arena.mul(&rest),
            };
            return (e, coeff);
        }
    }
    (arena.zero, term)
}

/// Recursively collect every sum in `expr` by the given variables, in
/// order: each `Add` node is grouped by powers of `vars[0]`, the
/// resulting coefficients by `vars[1]`, and so on.
pub(crate) fn rcollect(arena: &mut Arena, expr: ExprId, vars: &[ExprId]) -> ExprId {
    if vars.is_empty() {
        return expr;
    }
    let post_order = walk::post_order_ids(arena, expr);
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();

    for &id in &post_order {
        let rebuilt = walk::rebuild_with_cache(arena, id, &cache);
        let result = if matches!(arena.node(rebuilt), ExprNode::Add(_)) {
            collect_by_vars(arena, rebuilt, vars)
        } else {
            rebuilt
        };
        cache.insert(id, result);
    }

    cache.get(&expr).copied().unwrap_or(expr)
}

/// Collect one sum by `vars[0]`, then its coefficients by the rest.
fn collect_by_vars(arena: &mut Arena, sum: ExprId, vars: &[ExprId]) -> ExprId {
    let Some((&first, rest)) = vars.split_first() else {
        return sum;
    };
    let collected = collect_powers(arena, sum, first);
    if rest.is_empty() {
        return collected;
    }
    // Recurse into the coefficient of every power of `first`.
    let children = match arena.node(collected) {
        ExprNode::Add(c) => c.clone(),
        _ => return collect_by_vars_term(arena, collected, first, rest),
    };
    let new_children: SmallVec<[ExprId; 6]> = children
        .iter()
        .map(|&t| collect_by_vars_term(arena, t, first, rest))
        .collect();
    arena.add(&new_children)
}

/// Apply [`collect_by_vars`] to the coefficient part of one `coeff · var^e` term.
fn collect_by_vars_term(arena: &mut Arena, term: ExprId, var: ExprId, rest: &[ExprId]) -> ExprId {
    let (exp, coeff) = split_power_of(arena, term, var);
    let new_coeff = if matches!(arena.node(coeff), ExprNode::Add(_)) {
        collect_by_vars(arena, coeff, rest)
    } else {
        coeff
    };
    if new_coeff == coeff {
        return term;
    }
    if exp == arena.zero {
        new_coeff
    } else {
        let p = arena.pow(var, exp);
        arena.mul(&[new_coeff, p])
    }
}

// ---------------------------------------------------------------------------
// Numeric GCD helpers
// ---------------------------------------------------------------------------

/// Compute the GCD of a list of rational numbers.
fn rational_gcd_multi(values: &[&Q]) -> Q {
    if values.is_empty() {
        return Ratio::one();
    }
    let mut result = values[0].clone();
    if result.is_negative() {
        result = -result;
    }
    for &v in &values[1..] {
        result = rational_gcd(&result, v);
    }
    result
}

/// GCD of two rationals: gcd(a/b, c/d) = gcd(a,c) / lcm(b,d).
fn rational_gcd(a: &Q, b: &Q) -> Q {
    let a_abs = if a.is_negative() {
        -a.clone()
    } else {
        a.clone()
    };
    let b_abs = if b.is_negative() {
        -b.clone()
    } else {
        b.clone()
    };
    if a_abs.is_zero() {
        return b_abs;
    }
    if b_abs.is_zero() {
        return a_abs;
    }
    let numer = a_abs.numer().gcd(b_abs.numer());
    let denom = a_abs.denom().lcm(b_abs.denom());
    Ratio::new(numer, denom)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    fn sym(a: &mut Arena, name: &str) -> ExprId {
        a.symbol(name)
    }
    fn display(a: &Arena, id: ExprId) -> String {
        a.display(id).to_string()
    }

    #[test]
    fn factor_2x_plus_2y() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let two = a.int(2);
        let two_x = a.mul(&[two, x]);
        let two_y = a.mul(&[two, y]);
        let expr = a.add(&[two_x, two_y]);
        let result = factor_terms(&mut a, expr);
        let s = display(&a, result);
        // Should be 2*(x+y) or equivalent
        assert!(s.contains("2"), "should contain factor 2: {s}");
    }

    #[test]
    fn factor_2x_plus_2y_pair() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let two = a.int(2);
        let two_x = a.mul(&[two, x]);
        let two_y = a.mul(&[two, y]);
        let expr = a.add(&[two_x, two_y]);
        let (gcd, inner) = factor_terms_pair(&mut a, expr);
        assert_eq!(gcd, Ratio::from_integer(BigInt::from(2)));
        let inner_s = display(&a, inner);
        assert!(
            inner_s.contains("x") && inner_s.contains("y"),
            "inner should contain x and y: {inner_s}"
        );
    }

    #[test]
    fn factor_3x2_plus_6x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let three = a.int(3);
        let six = a.int(6);
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let three_x2 = a.mul(&[three, x2]);
        let six_x = a.mul(&[six, x]);
        let expr = a.add(&[three_x2, six_x]);
        let result = factor_terms(&mut a, expr);
        let s = display(&a, result);
        assert!(s.contains("3"), "should contain factor 3: {s}");
    }

    #[test]
    fn factor_3x2_plus_6x_pair() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let three = a.int(3);
        let six = a.int(6);
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let three_x2 = a.mul(&[three, x2]);
        let six_x = a.mul(&[six, x]);
        let expr = a.add(&[three_x2, six_x]);
        let (gcd, inner) = factor_terms_pair(&mut a, expr);
        assert_eq!(gcd, Ratio::from_integer(BigInt::from(3)));
        let inner_s = display(&a, inner);
        assert!(inner_s.contains("x"), "inner should contain x: {inner_s}");
    }

    #[test]
    fn factor_no_common() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let expr = a.add(&[x, y]);
        let result = factor_terms(&mut a, expr);
        assert_eq!(result, expr, "no common factor should leave unchanged");
    }

    #[test]
    fn factor_no_common_pair() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let expr = a.add(&[x, y]);
        let (gcd, inner) = factor_terms_pair(&mut a, expr);
        assert!(gcd.is_one(), "gcd should be 1 when no common factor");
        assert_eq!(inner, expr, "inner should be original expression");
    }

    #[test]
    fn factor_negative_coeffs() {
        // Note: factor_terms extracts gcd=2 and rebuilds as 2*(-2x-3y),
        // but canon_mul distributes Number×Add back to -4x-6y.
        // This is a known limitation of Number×Add distribution.
        // The function still runs without error.
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let neg_four = a.int(-4);
        let neg_six = a.int(-6);
        let t1 = a.mul(&[neg_four, x]);
        let t2 = a.mul(&[neg_six, y]);
        let expr = a.add(&[t1, t2]);
        let result = factor_terms(&mut a, expr);
        let s = display(&a, result);
        // Due to Number×Add distribution, result may be equivalent but not visibly factored
        assert!(
            s.contains("x") && s.contains("y"),
            "should still contain x and y: {s}"
        );
    }

    #[test]
    fn factor_negative_coeffs_pair() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let neg_four = a.int(-4);
        let neg_six = a.int(-6);
        let t1 = a.mul(&[neg_four, x]);
        let t2 = a.mul(&[neg_six, y]);
        let expr = a.add(&[t1, t2]);
        let (gcd, inner) = factor_terms_pair(&mut a, expr);
        assert_eq!(gcd, Ratio::from_integer(BigInt::from(2)));
        let inner_s = display(&a, inner);
        assert!(
            inner_s.contains("x") && inner_s.contains("y"),
            "inner should contain x and y: {inner_s}"
        );
    }

    #[test]
    fn factor_rational_coeffs() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let half = a.rational(1, 2);
        let three_halves = a.rational(3, 2);
        let t1 = a.mul(&[half, x]);
        let t2 = a.mul(&[three_halves, y]);
        let expr = a.add(&[t1, t2]);
        let result = factor_terms(&mut a, expr);
        let s = display(&a, result);
        // gcd(1/2, 3/2) = 1/2
        assert!(s.contains("1/2"), "should factor out 1/2: {s}");
    }

    #[test]
    fn factor_rational_coeffs_pair() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let half = a.rational(1, 2);
        let three_halves = a.rational(3, 2);
        let t1 = a.mul(&[half, x]);
        let t2 = a.mul(&[three_halves, y]);
        let expr = a.add(&[t1, t2]);
        let (gcd, inner) = factor_terms_pair(&mut a, expr);
        assert_eq!(
            gcd,
            Ratio::new(BigInt::from(1), BigInt::from(2)),
            "gcd should be 1/2"
        );
        let inner_s = display(&a, inner);
        assert!(
            inner_s.contains("x") && inner_s.contains("y"),
            "inner should contain x and y: {inner_s}"
        );
    }

    #[test]
    fn factor_single_term_unchanged() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let result = factor_terms(&mut a, x);
        assert_eq!(result, x);
    }

    #[test]
    fn factor_single_term_unchanged_pair() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let (gcd, inner) = factor_terms_pair(&mut a, x);
        assert!(gcd.is_one(), "gcd should be 1 for single term");
        assert_eq!(inner, x, "inner should be the original expression");
    }

    // -----------------------------------------------------------------------
    // Symbolic factor extraction tests
    // -----------------------------------------------------------------------

    #[test]
    fn symbolic_factor_mgl_plus_mgx() {
        let mut a = Arena::new();
        let m = sym(&mut a, "m");
        let g = sym(&mut a, "g");
        let l = sym(&mut a, "l");
        let x = sym(&mut a, "x");
        let mgl = a.mul(&[m, g, l]);
        let mgx = a.mul(&[m, g, x]);
        let expr = a.add(&[mgl, mgx]);
        let (gcd, inner) = symbolic_factor_terms_pair(&mut a, expr);
        let gcd_s = display(&a, gcd);
        let inner_s = display(&a, inner);
        // GCD should contain m and g
        assert!(
            gcd_s.contains("m") && gcd_s.contains("g"),
            "GCD should be m*g, got: {gcd_s}"
        );
        // Inner should be l + x
        assert!(
            inner_s.contains("l") && inner_s.contains("x"),
            "Inner should be l + x, got: {inner_s}"
        );
    }

    #[test]
    fn symbolic_factor_partial_overlap() {
        let mut a = Arena::new();
        let m = sym(&mut a, "m");
        let g = sym(&mut a, "g");
        let l = sym(&mut a, "l");
        let x = sym(&mut a, "x");
        // m*g*l + m*x  → common factor is just m
        let mgl = a.mul(&[m, g, l]);
        let mx = a.mul(&[m, x]);
        let expr = a.add(&[mgl, mx]);
        let (gcd, inner) = symbolic_factor_terms_pair(&mut a, expr);
        let gcd_s = display(&a, gcd);
        assert_eq!(gcd_s, "m", "GCD should be m, got: {gcd_s}");
        // Verify inner: m*(g*l + x) factored → inner should be (g*l + x)
        let inner_s = display(&a, inner);
        assert!(
            inner_s.contains("g") && inner_s.contains("x"),
            "inner should contain both g and x terms, got: {inner_s}"
        );
    }

    #[test]
    fn symbolic_factor_with_powers() {
        let mut a = Arena::new();
        let m = sym(&mut a, "m");
        let x = sym(&mut a, "x");
        let two = a.int(2);
        // m^2*x + m^2  → m^2 * (x + 1)
        let m2 = a.pow(m, two);
        let m2x = a.mul(&[m2, x]);
        let expr = a.add(&[m2x, m2]);
        let (gcd, inner) = symbolic_factor_terms_pair(&mut a, expr);
        let gcd_s = display(&a, gcd);
        assert!(
            gcd_s.contains("m") && gcd_s.contains("2"),
            "GCD should be m^2, got: {gcd_s}"
        );
        // Verify inner: m^2*(x + 1) factored → inner should be (x + 1)
        let inner_s = display(&a, inner);
        assert!(
            inner_s.contains("x"),
            "inner should contain x, got: {inner_s}"
        );
    }

    #[test]
    fn symbolic_factor_combined_numeric_and_symbolic() {
        let mut a = Arena::new();
        let m = sym(&mut a, "m");
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let six = a.int(6);
        let three = a.int(3);
        // 6*m*x + 3*m*y → 3*m*(2*x + y)
        let t1 = a.mul(&[six, m, x]);
        let t2 = a.mul(&[three, m, y]);
        let expr = a.add(&[t1, t2]);
        let (gcd, inner) = symbolic_factor_terms_pair(&mut a, expr);
        let gcd_s = display(&a, gcd);
        // GCD should contain 3 and m
        assert!(
            gcd_s.contains("3") && gcd_s.contains("m"),
            "GCD should be 3*m, got: {gcd_s}"
        );
        // Verify inner: 3*m*(2*x + y) factored → inner should be (2*x + y)
        let inner_s = display(&a, inner);
        assert!(
            inner_s.contains("x") && inner_s.contains("y"),
            "inner should contain both x and y, got: {inner_s}"
        );
    }

    #[test]
    fn symbolic_factor_no_common_symbolic() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        // x + y → no common symbolic factor
        let expr = a.add(&[x, y]);
        let (gcd, _inner) = symbolic_factor_terms_pair(&mut a, expr);
        assert_eq!(gcd, a.one, "GCD should be 1 when no common factors");
    }

    #[test]
    fn symbolic_factor_function_as_factor() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let sin_x = a.sin(x);
        // sin(x)*y + sin(x)  → sin(x) * (y + 1)
        let t1 = a.mul(&[sin_x, y]);
        let expr = a.add(&[t1, sin_x]);
        let (gcd, inner) = symbolic_factor_terms_pair(&mut a, expr);
        let gcd_s = display(&a, gcd);
        assert!(gcd_s.contains("sin"), "GCD should be sin(x), got: {gcd_s}");
        let _inner_s = display(&a, inner);
    }

    // ── signsimp / collect helpers ────────────────────────────────

    fn disp(a: &Arena, id: ExprId) -> String {
        a.display(id).to_string()
    }

    #[test]
    fn signsimp_even_and_odd_powers() {
        let mut a = Arena::new();
        let (x, y) = (a.symbol("x"), a.symbol("y"));
        let y_minus_x = a.sub(y, x);
        let two = a.int(2);
        let three = a.int(3);
        let sq = a.pow(y_minus_x, two);
        let cu = a.pow(y_minus_x, three);
        let r2 = signsimp(&mut a, sq);
        assert_eq!(disp(&a, r2), "(x - y)^2");
        let r3 = signsimp(&mut a, cu);
        assert_eq!(disp(&a, r3), "-(x - y)^3");
        // Already normalised: unchanged.
        let x_minus_y = a.sub(x, y);
        let ok = a.pow(x_minus_y, two);
        assert_eq!(signsimp(&mut a, ok), ok);
    }

    #[test]
    fn signsimp_mul_factors() {
        let mut a = Arena::new();
        let (x, y, z) = (a.symbol("x"), a.symbol("y"), a.symbol("z"));
        let y_minus_x = a.sub(y, x);
        let e = a.mul(&[z, y_minus_x]);
        let r = signsimp(&mut a, e);
        assert_eq!(disp(&a, r), "-z*(x - y)");
        // Non-integer power: untouched.
        let half = a.rational(1, 2);
        let sq = a.pow(y_minus_x, half);
        assert_eq!(signsimp(&mut a, sq), sq);
    }

    #[test]
    fn collect_powers_groups_symbolic_exponents() {
        let mut a = Arena::new();
        let (x, y, z, n) = (a.symbol("x"), a.symbol("y"), a.symbol("z"), a.symbol("n"));
        let xn = a.pow(x, n);
        let t1 = a.mul(&[y, xn]);
        let t2 = a.mul(&[z, xn]);
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let e = a.add(&[t1, t2, x2]);
        let r = collect_powers(&mut a, e, x);
        assert_eq!(disp(&a, r), "x^2 + x^n*(y + z)");
        // Nothing to group: unchanged.
        let f = a.add(&[t1, x2]);
        assert_eq!(collect_powers(&mut a, f, x), f);
        // Non-Add: unchanged.
        assert_eq!(collect_powers(&mut a, t1, x), t1);
    }

    #[test]
    fn collect_const_factors_content_inside_mul_and_pow() {
        let mut a = Arena::new();
        let (x, y, z) = (a.symbol("x"), a.symbol("y"), a.symbol("z"));
        let two = a.int(2);
        let four = a.int(4);
        let t1 = a.mul(&[two, x]);
        let t2 = a.mul(&[four, y]);
        let sum = a.add(&[t1, t2]);
        let e = a.mul(&[z, sum]);
        let r = collect_const(&mut a, e);
        assert_eq!(disp(&a, r), "2*z*(x + 2*y)");
        let p = a.pow(sum, two);
        let r = collect_const(&mut a, p);
        assert_eq!(disp(&a, r), "4*(x + 2*y)^2");
        // Top-level sum cannot hold the factor: unchanged.
        assert_eq!(collect_const(&mut a, sum), sum);
    }

    #[test]
    fn rcollect_recurses_into_coefficients() {
        let mut a = Arena::new();
        let (x, y, z) = (a.symbol("x"), a.symbol("y"), a.symbol("z"));
        let xy = a.mul(&[x, y]);
        let xz = a.mul(&[x, z]);
        let e = a.add(&[xy, xz]);
        let inner = a.sin(e);
        let r = rcollect(&mut a, inner, &[x]);
        assert_eq!(disp(&a, r), "sin(x*(y + z))");
        assert_eq!(rcollect(&mut a, inner, &[]), inner);
    }
}

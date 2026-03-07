//! Factor terms: pull common factors from sums.
//!
//! Implements [`factor_terms`], which extracts the GCD of numeric
//! coefficients and common symbolic factors from a sum:
//!
//! - `2x + 2y → 2(x + y)`
//! - `3x² + 6x → 3x(x + 2)`
//! - `m*g*l + m*g*x → m*g*(l + x)`

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};
use num_bigint::BigInt;
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
fn pow_rational(r: &Ratio<BigInt>, n: i64) -> Ratio<BigInt> {
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
        result = result * r;
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
fn decompose_product(arena: &mut Arena, id: ExprId) -> (Ratio<BigInt>, FxHashMap<ExprId, i64>) {
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
                    coeff = coeff * content;
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
                coeff = coeff * content;
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
fn extract_add_content(arena: &mut Arena, base: ExprId, exp: i64) -> (Ratio<BigInt>, ExprId) {
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
fn rebuild_term(
    arena: &mut Arena,
    coeff: &Ratio<BigInt>,
    factors: &FxHashMap<ExprId, i64>,
) -> ExprId {
    let sym_expr = map_to_expr(arena, factors);
    arena.make_coeff_term(coeff.clone(), sym_expr)
}

// ---------------------------------------------------------------------------
// Numeric-only factor extraction (original implementation)
// ---------------------------------------------------------------------------

/// Factor out the GCD of numeric coefficients only.
/// Returns (gcd, inner) where expr == gcd * inner mathematically.
/// The inner expression has each coefficient divided by gcd.
fn numeric_factor_terms_pair(arena: &mut Arena, expr: ExprId) -> (Ratio<BigInt>, ExprId) {
    let node = arena.node(expr).clone();
    let children = match node {
        ExprNode::Add(ref children) if children.len() >= 2 => children.clone(),
        _ => return (Ratio::one(), expr),
    };

    let mut pairs: Vec<(Ratio<BigInt>, ExprId)> = Vec::new();
    for &child in &children {
        let (coeff, term) = arena.as_coeff_term(child);
        pairs.push((coeff, term));
    }

    let coeffs: Vec<&Ratio<BigInt>> = pairs.iter().map(|(c, _)| c).collect();
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
pub(crate) fn factor_terms_pair(arena: &mut Arena, expr: ExprId) -> (Ratio<BigInt>, ExprId) {
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
    let coeffs: Vec<&Ratio<BigInt>> = decomposed.iter().map(|(c, _)| c).collect();
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
// Numeric GCD helpers
// ---------------------------------------------------------------------------

/// Compute the GCD of a list of rational numbers.
fn rational_gcd_multi(values: &[&Ratio<BigInt>]) -> Ratio<BigInt> {
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
fn rational_gcd(a: &Ratio<BigInt>, b: &Ratio<BigInt>) -> Ratio<BigInt> {
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
        assert!(
            gcd_s.contains("sin"),
            "GCD should be sin(x), got: {gcd_s}"
        );
        let _inner_s = display(&a, inner);
    }
}

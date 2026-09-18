//! Bridge between symbolic expressions ([`ExprId`]) and polynomials ([`Poly`]).
//!
//! This module provides:
//!
//! - [`expr_to_poly`]: convert a symbolic expression to a [`Poly`] in a
//!   given variable (returns `None` if the expression is not polynomial
//!   in that variable).
//! - [`poly_to_expr`]: convert a [`Poly`] back to a symbolic expression.
//! - [`cancel`]: cancel common polynomial factors in a rational
//!   expression (numerator / denominator).
//!
//! # Design
//!
//! Expression → Poly conversion walks the expression tree iteratively
//! (using [`walk::post_order_ids`]) and builds a `Poly` bottom-up.
//! At each node it checks whether the sub-expression is polynomial in
//! the given variable; if not, the conversion fails with `None`.
//!
//! Poly → expression conversion builds the expression from the
//! coefficient list using Horner's method for efficiency.
//!
//! `cancel` decomposes an expression into numerator and denominator
//! (by collecting negative-exponent factors), converts both to `Poly`,
//! divides out their GCD, and rebuilds the expression.

use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};
use crate::base::walk;
use crate::poly::Poly;
use crate::poly::multipoly::{GrevLex, MultiPoly};

// ═══════════════════════════════════════════════════════════════════════════
// Expression → Poly
// ═══════════════════════════════════════════════════════════════════════════

/// Try to convert an expression into a univariate polynomial in `var`.
///
/// Returns `None` if the expression contains terms that are not
/// polynomial in `var` (e.g., `sin(x)`, `x^(1/2)`, `x^y`).
///
/// Constant sub-expressions (not containing `var`) are treated as
/// degree-0 coefficients.
pub(crate) fn expr_to_poly(arena: &Arena, expr: ExprId, var: ExprId) -> Option<Poly> {
    // Fast path: if the expression doesn't contain var at all, it's
    // a constant polynomial.
    if !contains_id(arena, expr, var) {
        let coeff = expr_to_rational(arena, expr)?;
        return Some(Poly::constant(coeff));
    }

    // The variable itself.
    if expr == var {
        return Some(Poly::x());
    }

    // Walk the tree bottom-up.
    let post_order = walk::post_order_ids(arena, expr);
    let mut cache: FxHashMap<ExprId, Poly> = FxHashMap::default();

    for &id in &post_order {
        let poly = convert_node(arena, id, var, &cache)?;
        cache.insert(id, poly);
    }

    cache.remove(&expr)
}

/// Convert a single node to a Poly, given its children's Poly values.
fn convert_node(
    arena: &Arena,
    id: ExprId,
    var: ExprId,
    cache: &FxHashMap<ExprId, Poly>,
) -> Option<Poly> {
    // The variable itself.
    if id == var {
        return Some(Poly::x());
    }

    let node = arena.node(id);

    match node {
        // Numeric literal → constant polynomial.
        ExprNode::Num(nid) => {
            let r = arena.num(*nid).clone();
            Some(Poly::constant(r))
        }

        // Symbol that is NOT the variable → try as rational constant.
        ExprNode::Symbol(_) => {
            if !contains_id(arena, id, var) {
                // It's a different symbol — not polynomial unless we
                // treat it as a parameter.  For univariate polynomials,
                // other symbols make conversion fail.
                // Exception: if it doesn't depend on var, treat as a
                // "constant" but we can't represent it as Ratio<BigInt>.
                None
            } else {
                // This IS the variable (caught above), shouldn't reach here.
                Some(Poly::x())
            }
        }

        // Constants.
        ExprNode::Pi
        | ExprNode::E
        | ExprNode::ImaginaryUnit
        | ExprNode::EulerGamma
        | ExprNode::Catalan
        | ExprNode::GoldenRatio
        | ExprNode::PhysicalConstant(_, _) => None,
        ExprNode::Infinity | ExprNode::NegInfinity | ExprNode::ComplexInfinity | ExprNode::NaN => {
            None
        }

        // Add: sum of child polynomials.
        ExprNode::Add(children) => {
            let mut result = Poly::zero();
            for &child in children.iter() {
                let child_poly = cache.get(&child)?;
                result = &result + child_poly;
            }
            Some(result)
        }

        // Mul: product of child polynomials.
        ExprNode::Mul(children) => {
            let mut result = Poly::from_int(1);
            for &child in children.iter() {
                let child_poly = cache.get(&child)?;
                result = &result * child_poly;
            }
            Some(result)
        }

        // Pow: base^exp where exp must be a non-negative integer.
        ExprNode::Pow(base, exp) => {
            let base_poly = cache.get(base)?;

            // The exponent must be a non-negative integer constant
            // (not depending on var).
            if contains_id(arena, *exp, var) {
                return None; // x^x is not polynomial
            }

            let exp_val = expr_to_rational(arena, *exp)?;
            if !exp_val.is_integer() {
                return None; // x^(1/2) is not polynomial
            }

            let n: i64 = exp_val.to_integer().try_into().ok()?;
            if n < 0 {
                return None; // x^(-1) is not polynomial
            }

            let mut result = Poly::from_int(1);
            for _ in 0..n {
                result = &result * base_poly;
            }
            Some(result)
        }

        // Neg: negate the child polynomial.
        ExprNode::Neg(inner) => {
            let inner_poly = cache.get(inner)?;
            Some(-inner_poly)
        }

        // Functions containing var → not polynomial.
        ExprNode::Sin(_)
        | ExprNode::Cos(_)
        | ExprNode::Tan(_)
        | ExprNode::Asin(_)
        | ExprNode::Acos(_)
        | ExprNode::Atan(_)
        | ExprNode::Atan2(_, _)
        | ExprNode::Sinh(_)
        | ExprNode::Cosh(_)
        | ExprNode::Tanh(_)
        | ExprNode::Asinh(_)
        | ExprNode::Acosh(_)
        | ExprNode::Atanh(_)
        | ExprNode::Exp(_)
        | ExprNode::Ln(_)
        | ExprNode::Abs(_)
        | ExprNode::Sign(_)
        | ExprNode::Heaviside(_)
        | ExprNode::DiracDelta(_)
        | ExprNode::Floor(_)
        | ExprNode::Ceiling(_) => {
            // If the function argument doesn't contain var, the whole
            // thing is a constant — but we can't represent transcendentals
            // as Ratio<BigInt>, so we fail.
            None
        }

        ExprNode::Apply(_, _)
        | ExprNode::Derivative(_, _)
        | ExprNode::Integral(_, _)
        | ExprNode::DefiniteIntegral(_, _, _, _) => None,

        // Min/Max/Sum/Product are not polynomial.
        ExprNode::Min(_)
        | ExprNode::Max(_)
        | ExprNode::Sum(_, _, _, _)
        | ExprNode::Product_(_, _, _, _) => None,

        // Combinatorial nodes are not polynomial.
        ExprNode::Factorial(_) | ExprNode::Binomial(_, _) => None,

        // Special functions are not polynomial.
        ExprNode::Gamma(_)
        | ExprNode::LogGamma(_)
        | ExprNode::Digamma(_)
        | ExprNode::Erf(_)
        | ExprNode::Erfc(_)
        | ExprNode::LambertW(_)
        | ExprNode::Beta(_, _)
        | ExprNode::Re(_)
        | ExprNode::Im(_)
        | ExprNode::Conjugate(_)
        | ExprNode::Arg(_)
        | ExprNode::Si(_)
        | ExprNode::Ci(_)
        | ExprNode::Ei(_)
        | ExprNode::Li(_)
        | ExprNode::Zeta(_)
        | ExprNode::Polygamma(_, _)
        | ExprNode::KroneckerDelta(_, _) => None,

        // Boolean, relational, logical, and piecewise nodes are not polynomial.
        ExprNode::BoolTrue
        | ExprNode::BoolFalse
        | ExprNode::Gt(_, _)
        | ExprNode::Ge(_, _)
        | ExprNode::Eq_(_, _)
        | ExprNode::Ne(_, _)
        | ExprNode::And(_)
        | ExprNode::Or(_)
        | ExprNode::Not(_)
        | ExprNode::Piecewise(_) => None,

        // Set-valued nodes are not polynomial.
        ExprNode::EmptySet
        | ExprNode::UniversalSet
        | ExprNode::Interval(_, _, _)
        | ExprNode::FiniteSet(_)
        | ExprNode::SetUnion(_)
        | ExprNode::SetIntersection(_)
        | ExprNode::SetComplement(_, _) => None,

        // Formal/unevaluated nodes are not polynomial.
        ExprNode::Limit(_, _, _)
        | ExprNode::Series(_, _, _, _)
        | ExprNode::LaplaceTransform(_, _, _)
        | ExprNode::InverseLaplaceTransform(_, _, _)
        | ExprNode::Residue(_, _, _)
        | ExprNode::RootOf(_, _)
        | ExprNode::DSolve(_, _, _)
        | ExprNode::RootSum(_, _, _)
        | ExprNode::ConditionSet(_, _) => None,
    }
}

/// Try to extract a rational number from an expression that is a
/// pure numeric constant (no symbols, no transcendentals).
fn expr_to_rational(arena: &Arena, id: ExprId) -> Option<Ratio<BigInt>> {
    match arena.node(id) {
        ExprNode::Num(nid) => Some(arena.num(*nid).clone()),
        // For Add/Mul/Pow/Neg of pure numbers, the canonical form
        // should already be a single Num node.  But just in case:
        _ => None,
    }
}

/// Check if `needle` appears anywhere in the expression tree rooted
/// at `haystack`.  Uses an explicit stack (no recursion).
fn contains_id(arena: &Arena, haystack: ExprId, needle: ExprId) -> bool {
    if haystack == needle {
        return true;
    }
    let mut visited: FxHashMap<ExprId, ()> = FxHashMap::default();
    let mut stack: Vec<ExprId> = vec![haystack];
    while let Some(id) = stack.pop() {
        if id == needle {
            return true;
        }
        if visited.contains_key(&id) {
            continue;
        }
        visited.insert(id, ());
        let children = arena.node(id).children();
        stack.extend_from_slice(&children);
    }
    false
}

// ═══════════════════════════════════════════════════════════════════════════
// Poly → Expression
// ═══════════════════════════════════════════════════════════════════════════

/// Convert a polynomial back to a symbolic expression in `var`.
///
/// Uses Horner-style construction for efficiency:
/// `a₀ + a₁x + a₂x² = a₀ + x*(a₁ + x*a₂)`
///
/// However, for clarity and canonical form, we build
/// `a₀ + a₁*x + a₂*x^2 + …` and let the arena's canonicalization
/// handle the rest.
pub(crate) fn poly_to_expr(arena: &mut Arena, poly: &Poly, var: ExprId) -> ExprId {
    if poly.is_zero() {
        return arena.zero;
    }

    let coeffs = poly.coeffs();
    let mut terms: SmallVec<[ExprId; 8]> = SmallVec::new();

    for (i, c) in coeffs.iter().enumerate() {
        if c.is_zero() {
            continue;
        }

        let coeff_id = {
            let nid = arena.intern_num(c.clone());
            arena.intern(ExprNode::Num(nid))
        };

        if i == 0 {
            // Constant term.
            terms.push(coeff_id);
        } else {
            // c * var^i
            let power = if i == 1 {
                var
            } else {
                let exp = arena.int(i as i64);
                arena.pow(var, exp)
            };

            if c.is_one() {
                terms.push(power);
            } else {
                terms.push(arena.mul(&[coeff_id, power]));
            }
        }
    }

    match terms.len() {
        0 => arena.zero,
        1 => terms[0],
        _ => arena.add(&terms),
    }
}

/// Convert a [`RationalFn`](crate::poly::ratfn::RationalFn) (a rational function `p(var)/q(var)`) to an
/// arena expression.
///
/// Uses [`poly_to_expr`] for both the numerator and denominator polynomials.
/// If the denominator is the constant 1, only the numerator expression is
/// returned (no division node).
pub(crate) fn ratfn_to_expr(
    arena: &mut Arena,
    rf: &crate::poly::ratfn::RationalFn,
    var: ExprId,
) -> ExprId {
    let n = poly_to_expr(arena, rf.numer(), var);
    if rf.denom().is_constant() {
        let d_val = rf.denom().coeff(0);
        if d_val.is_one() {
            return n;
        }
    }
    let d = poly_to_expr(arena, rf.denom(), var);
    arena.div(n, d)
}

// ═══════════════════════════════════════════════════════════════════════════
// Expression ↔ MultiPoly
// ═══════════════════════════════════════════════════════════════════════════

/// Try to convert an expression into a multivariate polynomial over ℚ in
/// the given variables (`vars[i]` becomes variable index `i`).
///
/// Returns `None` if the expression is not polynomial in those variables
/// (transcendental functions, fractional or negative powers, or symbols
/// not listed in `vars`).  Uses an explicit post-order walk.
pub(crate) fn expr_to_multipoly(
    arena: &Arena,
    expr: ExprId,
    vars: &[ExprId],
) -> Option<MultiPoly<GrevLex>> {
    let nv = vars.len();
    let var_index: FxHashMap<ExprId, usize> =
        vars.iter().enumerate().map(|(i, &v)| (v, i)).collect();

    let post_order = walk::post_order_ids(arena, expr);
    let mut cache: FxHashMap<ExprId, MultiPoly<GrevLex>> = FxHashMap::default();

    for &id in &post_order {
        if let Some(&i) = var_index.get(&id) {
            cache.insert(id, MultiPoly::var(nv, i));
            continue;
        }
        let poly = match arena.node(id) {
            ExprNode::Num(nid) => MultiPoly::constant(nv, arena.num(*nid).clone()),
            ExprNode::Add(children) => {
                let mut acc = MultiPoly::zero(nv);
                for c in children.iter() {
                    acc = acc.add(cache.get(c)?);
                }
                acc
            }
            ExprNode::Mul(children) => {
                let mut acc = MultiPoly::from_int(nv, 1);
                for c in children.iter() {
                    acc = acc.mul(cache.get(c)?);
                }
                acc
            }
            ExprNode::Pow(base, exp) => {
                let base_poly = cache.get(base)?;
                let exp_val = expr_to_rational(arena, *exp)?;
                if !exp_val.is_integer() {
                    return None;
                }
                let n: u32 = exp_val.to_integer().try_into().ok()?;
                let mut acc = MultiPoly::from_int(nv, 1);
                for _ in 0..n {
                    acc = acc.mul(base_poly);
                }
                acc
            }
            ExprNode::Neg(inner) => cache.get(inner)?.neg(),
            _ => return None,
        };
        cache.insert(id, poly);
    }

    cache.remove(&expr)
}

/// Convert a multivariate polynomial back to an expression, mapping
/// variable index `i` to `vars[i]`.
pub(crate) fn multipoly_to_expr(
    arena: &mut Arena,
    poly: &MultiPoly<GrevLex>,
    vars: &[ExprId],
) -> ExprId {
    let mut terms: Vec<ExprId> = Vec::with_capacity(poly.num_terms());
    for (exp, c) in poly.terms() {
        let mut factors: SmallVec<[ExprId; 6]> = SmallVec::new();
        if !c.is_one() {
            let nid = arena.intern_num(c.clone());
            factors.push(arena.intern(ExprNode::Num(nid)));
        }
        for (i, &e) in exp.iter().enumerate() {
            if e == 0 {
                continue;
            }
            if e == 1 {
                factors.push(vars[i]);
            } else {
                let e_id = arena.int(e as i64);
                factors.push(arena.pow(vars[i], e_id));
            }
        }
        let term = match factors.len() {
            0 => arena.one,
            1 => factors[0],
            _ => arena.mul(&factors),
        };
        terms.push(term);
    }
    match terms.len() {
        0 => arena.zero,
        1 => terms[0],
        _ => arena.add(&terms),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Expression → sparse polynomial with symbolic coefficients
// ═══════════════════════════════════════════════════════════════════════════

/// View `expr` as a polynomial in the generators `gens` whose coefficients
/// are arbitrary generator-free expressions.
///
/// The expression is expanded, each resulting product term is split into
/// an exponent vector over `gens` and a generator-free coefficient, and
/// terms with equal exponent vectors are summed and evaluated.  Zero
/// coefficients are dropped, so the zero polynomial yields an empty list.
///
/// Returns `None` if a generator occurs in a non-polynomial position
/// (inside a function, under a non-integer or negative power, in an
/// exponent, …).  The result is in ascending lexicographic order of the
/// exponent vectors.
pub(crate) fn symbolic_multipoly_terms(
    arena: &mut Arena,
    expr: ExprId,
    gens: &[ExprId],
) -> Option<Vec<(Vec<u32>, ExprId)>> {
    let expanded = crate::transforms::expand::expand(arena, expr);
    let terms: Vec<ExprId> = match arena.node(expanded) {
        ExprNode::Add(children) => children.to_vec(),
        _ => vec![expanded],
    };
    let mut buckets: std::collections::BTreeMap<Vec<u32>, Vec<ExprId>> =
        std::collections::BTreeMap::new();
    for term in terms {
        let (exps, coeff) = term_exponents_coeff(arena, term, gens)?;
        buckets.entry(exps).or_default().push(coeff);
    }
    let mut out = Vec::with_capacity(buckets.len());
    for (exps, bucket) in buckets {
        let c = if bucket.len() == 1 {
            bucket[0]
        } else {
            arena.add(&bucket)
        };
        let c = crate::transforms::eval::eval(arena, c);
        if !arena.is_zero_structural(c) {
            out.push((exps, c));
        }
    }
    Some(out)
}

/// Does `expr` contain any of `gens`?
fn contains_any(arena: &Arena, expr: ExprId, gens: &[ExprId]) -> bool {
    if gens.contains(&expr) {
        return true;
    }
    let mut visited: rustc_hash::FxHashSet<ExprId> = rustc_hash::FxHashSet::default();
    let mut stack: Vec<ExprId> = vec![expr];
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        if gens.contains(&id) {
            return true;
        }
        stack.extend(arena.node(id).children());
    }
    false
}

/// Split one product term into `(exponent vector over gens, coefficient)`.
fn term_exponents_coeff(
    arena: &mut Arena,
    term: ExprId,
    gens: &[ExprId],
) -> Option<(Vec<u32>, ExprId)> {
    let mut exps = vec![0u32; gens.len()];
    if !contains_any(arena, term, gens) {
        return Some((exps, term));
    }
    let mut consts: SmallVec<[ExprId; 4]> = SmallVec::new();
    match arena.node(term).clone() {
        ExprNode::Mul(children) => {
            for &child in &children {
                accumulate_factor(arena, child, gens, &mut exps, &mut consts)?;
            }
        }
        _ => accumulate_factor(arena, term, gens, &mut exps, &mut consts)?,
    }
    let c = match consts.len() {
        0 => arena.one,
        1 => consts[0],
        _ => arena.mul(&consts),
    };
    Some((exps, c))
}

/// Fold a single factor of a product term into the exponent vector
/// (`gen`, `gen^n`) or the coefficient list (generator-free), failing on
/// anything else.
fn accumulate_factor(
    arena: &Arena,
    factor: ExprId,
    gens: &[ExprId],
    exps: &mut [u32],
    consts: &mut SmallVec<[ExprId; 4]>,
) -> Option<()> {
    if let Some(i) = gens.iter().position(|&g| g == factor) {
        exps[i] = exps[i].checked_add(1)?;
        return Some(());
    }
    if !contains_any(arena, factor, gens) {
        consts.push(factor);
        return Some(());
    }
    match arena.node(factor) {
        ExprNode::Pow(base, exp) => {
            let i = gens.iter().position(|g| g == base)?;
            let n = arena.as_num(*exp)?;
            if !n.is_integer() || n.is_negative() {
                return None;
            }
            let d: u32 = n.to_integer().try_into().ok()?;
            exps[i] = exps[i].checked_add(d)?;
            Some(())
        }
        ExprNode::Neg(inner) => {
            consts.push(arena.neg_one);
            accumulate_factor(arena, *inner, gens, exps, consts)
        }
        _ => None,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Numerator / Denominator decomposition
// ═══════════════════════════════════════════════════════════════════════════

/// Decompose an expression into (numerator, denominator) where both
/// are free of negative-exponent factors.
///
/// For example:
/// - `x` → `(x, 1)`
/// - `1/x` = `x^(-1)` → `(1, x)`
/// - `(x+1)/(x-1)` = `(x+1)*(x-1)^(-1)` → `(x+1, x-1)`
/// - `x^2 * y^(-3)` → `(x^2, y^3)`
pub(crate) fn as_numer_denom(arena: &mut Arena, expr: ExprId) -> (ExprId, ExprId) {
    let node = arena.node(expr).clone();

    match node {
        // Pow(base, exp) where exp is a negative integer → denominator factor.
        ExprNode::Pow(base, exp) => {
            if let Some(r) = arena.as_num(exp) {
                let r = r.clone();
                if r.is_negative() {
                    // base^(-n) → numerator=1, denominator=base^n
                    let pos_exp = {
                        let neg_r = -r;
                        let nid = arena.intern_num(neg_r);
                        arena.intern(ExprNode::Num(nid))
                    };
                    let denom = arena.pow(base, pos_exp);
                    return (arena.one, denom);
                }
            }
            (expr, arena.one)
        }

        // Mul(factors): separate into numer_factors and denom_factors.
        ExprNode::Mul(ref children) => {
            let children = children.clone();
            let mut numer_factors: SmallVec<[ExprId; 6]> = SmallVec::new();
            let mut denom_factors: SmallVec<[ExprId; 6]> = SmallVec::new();

            for &child in &children {
                let (n, d) = as_numer_denom(arena, child);
                if n != arena.one {
                    numer_factors.push(n);
                }
                if d != arena.one {
                    denom_factors.push(d);
                }
            }

            let numer = if numer_factors.is_empty() {
                arena.one
            } else if numer_factors.len() == 1 {
                numer_factors[0]
            } else {
                arena.mul(&numer_factors)
            };

            let denom = if denom_factors.is_empty() {
                arena.one
            } else if denom_factors.len() == 1 {
                denom_factors[0]
            } else {
                arena.mul(&denom_factors)
            };

            (numer, denom)
        }

        // Everything else: numerator is the expression, denominator is 1.
        _ => (expr, arena.one),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// cancel()
// ═══════════════════════════════════════════════════════════════════════════

/// Cancel common polynomial factors in a rational expression.
///
/// Decomposes the expression into numerator / denominator, converts
/// both to polynomials in `var`, divides out the GCD, and rebuilds
/// the expression.
///
/// Returns the original expression unchanged if:
/// - The expression is not a rational function in `var`
/// - The numerator and denominator have no common factor
///
/// # Example
///
/// `(x² - 1) / (x - 1)` → `x + 1`
pub(crate) fn cancel(arena: &mut Arena, expr: ExprId, var: ExprId) -> ExprId {
    let (numer, denom) = as_numer_denom(arena, expr);

    // If denominator is 1, nothing to cancel.
    if denom == arena.one {
        return expr;
    }

    // Try to convert both to polynomials.
    let numer_poly = match expr_to_poly(arena, numer, var) {
        Some(p) => p,
        None => return expr,
    };
    let denom_poly = match expr_to_poly(arena, denom, var) {
        Some(p) => p,
        None => return expr,
    };

    // Compute polynomial GCD and divide out common factors.
    let gcd = Poly::gcd(&numer_poly, &denom_poly);
    let (mut new_numer, mut new_denom, mut changed) = if gcd.is_constant() && gcd.coeff(0).is_one()
    {
        // GCD is trivial (constant 1); no polynomial factor to divide out.
        (numer_poly, denom_poly, false)
    } else {
        (numer_poly.div(&gcd), denom_poly.div(&gcd), true)
    };

    // Also cancel constant content factors between numerator and denominator.
    let n_content = new_numer.content();
    let d_content = new_denom.content();
    if !n_content.is_zero() && !d_content.is_zero() {
        let numer_gcd = n_content.numer().gcd(d_content.numer());
        let denom_lcm = n_content.denom().lcm(d_content.denom());
        let content_gcd = Ratio::new(numer_gcd, denom_lcm);
        if !content_gcd.is_one() {
            let inv = Ratio::one() / content_gcd;
            new_numer = new_numer.scale(&inv);
            new_denom = new_denom.scale(&inv);
            changed = true;
        }
    }

    // If nothing was cancelled, return the original expression unchanged.
    if !changed {
        return expr;
    }

    // Convert back to expressions.
    let new_numer_expr = poly_to_expr(arena, &new_numer, var);

    if new_denom.is_zero() {
        // Shouldn't happen (GCD divides the denominator), but guard.
        return expr;
    }

    if new_denom.degree() == Some(0) && new_denom.coeff(0).is_one() {
        // Denominator cancelled to 1.
        return new_numer_expr;
    }

    // Rebuild as numer * denom^(-1).
    let new_denom_expr = poly_to_expr(arena, &new_denom, var);
    let neg_one = arena.neg_one;
    let denom_inv = arena.pow(new_denom_expr, neg_one);
    arena.mul(&[new_numer_expr, denom_inv])
}

/// Group an expression by powers of `var`.
///
/// Converts the expression to a univariate polynomial in `var`,
/// then rebuilds it term-by-term. This naturally groups coefficients
/// by power of `var`.
///
/// Returns the expression unchanged if it is not polynomial in `var`.
pub(crate) fn collect(arena: &mut Arena, expr: ExprId, var: ExprId) -> ExprId {
    let poly = match expr_to_poly(arena, expr, var) {
        Some(p) => p,
        None => return expr,
    };
    poly_to_expr(arena, &poly, var)
}

/// Return the degree of `expr` as a polynomial in `var`.
///
/// Returns `None` if the expression is not polynomial in `var`
/// or if it is the zero polynomial.
pub(crate) fn poly_degree(arena: &Arena, expr: ExprId, var: ExprId) -> Option<usize> {
    let poly = expr_to_poly(arena, expr, var)?;
    poly.degree()
}

/// Return the coefficients of `expr` as a polynomial in `var`,
/// in ascending degree order: `[a_0, a_1, a_2, ...]` where
/// `expr = a_0 + a_1*var + a_2*var^2 + ...`.
///
/// Returns `None` if the expression is not polynomial in `var`.
/// Returns an empty vec for the zero polynomial.
pub(crate) fn poly_coefficients(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
) -> Option<Vec<ExprId>> {
    let poly = expr_to_poly(arena, expr, var)?;
    let rational_coeffs = poly.coeffs();
    let mut result = Vec::with_capacity(rational_coeffs.len());
    for c in rational_coeffs {
        let nid = arena.intern_num(c.clone());
        result.push(arena.intern(crate::base::node::ExprNode::Num(nid)));
    }
    Some(result)
}

/// Combine fractions over a common denominator.
///
/// For an Add node, decomposes each term into numerator/denominator
/// via `as_numer_denom`, computes a common denominator (product of
/// all unique denominators, simplified via polynomial GCD), scales
/// each numerator, and rebuilds as `sum_of_numerators / common_denom`.
///
/// Returns the expression unchanged if it is not an Add, or if all
/// terms already have denominator 1.
pub(crate) fn together(arena: &mut Arena, expr: ExprId) -> ExprId {
    let node = arena.node(expr).clone();

    let children = match node {
        ExprNode::Add(ref ch) => ch.clone(),
        _ => return expr,
    };

    // Decompose each term into (numerator, denominator).
    let mut parts: Vec<(ExprId, ExprId)> = Vec::with_capacity(children.len());
    let mut all_denom_one = true;
    for &child in &children {
        let (n, d) = as_numer_denom(arena, child);
        if d != arena.one {
            all_denom_one = false;
        }
        parts.push((n, d));
    }

    // If every term has denominator 1, nothing to do.
    if all_denom_one {
        return expr;
    }

    // Collect distinct denominators (deduplicated by ExprId).
    let mut unique_denoms: Vec<ExprId> = Vec::new();
    for &(_, d) in &parts {
        if d != arena.one && !unique_denoms.contains(&d) {
            unique_denoms.push(d);
        }
    }

    // Try polynomial LCM for a simpler common denominator; fall back to
    // the product of distinct denominators when poly conversion fails.
    if let Some(result) = try_together_poly_lcm(arena, &parts, &unique_denoms) {
        return result;
    }
    together_product_fallback(arena, &parts, &unique_denoms)
}

/// Attempt to combine fractions using polynomial LCM of the denominators.
///
/// Returns `Some(combined_expr)` on success, `None` if polynomial conversion
/// fails for any denominator (e.g. multiple variables, transcendental denoms).
fn try_together_poly_lcm(
    arena: &mut Arena,
    parts: &[(ExprId, ExprId)],
    unique_denoms: &[ExprId],
) -> Option<ExprId> {
    if unique_denoms.is_empty() {
        return None;
    }

    // Find free variables across all denominators.
    let mut all_syms: Vec<ExprId> = Vec::new();
    for &d in unique_denoms {
        all_syms.extend(walk::free_symbols(arena, d));
    }
    all_syms.sort_by_key(|id| id.0);
    all_syms.dedup();

    // Need exactly one variable for univariate polynomial operations.
    if all_syms.len() != 1 {
        return None;
    }
    let var = all_syms[0];

    // Convert every distinct denominator to a Poly.
    let mut denom_poly_map: Vec<(ExprId, Poly)> = Vec::new();
    for &d in unique_denoms {
        let p = expr_to_poly(arena, d, var)?;
        denom_poly_map.push((d, p));
    }

    // Compute LCM of all denominator polynomials incrementally.
    let mut lcm = denom_poly_map[0].1.clone();
    for (_, p) in &denom_poly_map[1..] {
        let g = Poly::gcd(&lcm, p);
        if g.is_zero() {
            return None;
        }
        // lcm(a, b) = (a / gcd(a, b)) * b
        let a_over_g = lcm.div(&g);
        lcm = &a_over_g * p;
    }

    let common_denom_expr = poly_to_expr(arena, &lcm, var);

    // Build scaled numerators: numer_i * (lcm / denom_i).
    let one_poly = Poly::constant(Ratio::one());
    let mut scaled_numers: SmallVec<[ExprId; 6]> = SmallVec::new();
    for &(n, d) in parts {
        let d_poly = if d == arena.one {
            &one_poly
        } else {
            match denom_poly_map.iter().find(|(id, _)| *id == d) {
                Some((_, p)) => p,
                None => return None,
            }
        };
        let scale_poly = lcm.div(d_poly);
        if scale_poly.degree() == Some(0) && scale_poly.coeff(0).is_one() {
            // Scale factor is 1 — numerator unchanged.
            scaled_numers.push(n);
        } else {
            let scale_expr = poly_to_expr(arena, &scale_poly, var);
            let scaled = arena.mul(&[n, scale_expr]);
            scaled_numers.push(scaled);
        }
    }

    let numer_sum = arena.add(&scaled_numers);
    Some(arena.div(numer_sum, common_denom_expr))
}

/// Fallback: use the product of distinct denominators as the common denominator.
fn together_product_fallback(
    arena: &mut Arena,
    parts: &[(ExprId, ExprId)],
    unique_denoms: &[ExprId],
) -> ExprId {
    let common_denom = if unique_denoms.len() == 1 {
        unique_denoms[0]
    } else {
        arena.mul(unique_denoms)
    };

    let mut scaled_numers: SmallVec<[ExprId; 6]> = SmallVec::new();
    for &(n, d) in parts {
        if d == arena.one {
            let scaled = arena.mul(&[n, common_denom]);
            scaled_numers.push(scaled);
        } else {
            let mut other_denoms: Vec<ExprId> = Vec::new();
            for &ud in unique_denoms {
                if ud != d {
                    other_denoms.push(ud);
                }
            }
            if other_denoms.is_empty() {
                scaled_numers.push(n);
            } else {
                let scale = if other_denoms.len() == 1 {
                    other_denoms[0]
                } else {
                    arena.mul(&other_denoms)
                };
                let scaled = arena.mul(&[n, scale]);
                scaled_numers.push(scaled);
            }
        }
    }

    let numer_sum = arena.add(&scaled_numers);
    arena.div(numer_sum, common_denom)
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

    // ── expr_to_poly ────────────────────────────────────────────────

    #[test]
    fn expr_to_poly_constant() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let five = a.int(5);
        let p = expr_to_poly(&a, five, x).unwrap();
        assert_eq!(format!("{p}"), "5");
    }

    #[test]
    fn expr_to_poly_variable() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let p = expr_to_poly(&a, x, x).unwrap();
        assert_eq!(format!("{p}"), "θ");
    }

    #[test]
    fn expr_to_poly_linear() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let three = a.int(3);
        // 2*x + 3
        let two_x = a.mul(&[two, x]);
        let expr = a.add(&[two_x, three]);
        let p = expr_to_poly(&a, expr, x).unwrap();
        assert_eq!(p.degree(), Some(1));
        assert_eq!(format!("{p}"), "2*θ + 3");
    }

    #[test]
    fn expr_to_poly_quadratic() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        // x^2 + 2*x + 1
        let x_sq = a.pow(x, two);
        let two_x = a.mul(&[two, x]);
        let one = a.one;
        let expr = a.add(&[x_sq, two_x, one]);
        let p = expr_to_poly(&a, expr, x).unwrap();
        assert_eq!(p.degree(), Some(2));
        // Evaluate at x=3: should be 16.
        let val = p.eval(&Ratio::from_integer(BigInt::from(3)));
        assert_eq!(val, Ratio::from_integer(BigInt::from(16)));
    }

    #[test]
    fn expr_to_poly_fails_for_sin() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.sin(x);
        assert!(expr_to_poly(&a, expr, x).is_none());
    }

    #[test]
    fn expr_to_poly_fails_for_fractional_power() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let half = a.rational(1, 2);
        let expr = a.pow(x, half);
        assert!(expr_to_poly(&a, expr, x).is_none());
    }

    #[test]
    fn expr_to_poly_fails_for_negative_power() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let neg_one = a.int(-1);
        let expr = a.pow(x, neg_one);
        assert!(expr_to_poly(&a, expr, x).is_none());
    }

    #[test]
    fn expr_to_poly_product() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.one;
        // (x + 1) * (x - 1) — expanded by canon_mul's Number*Add distribution,
        // but the overall product might stay as Mul if factors aren't numeric.
        // Let's build x^2 - 1 directly.
        let two = a.int(2);
        let x_sq = a.pow(x, two);
        let expr = a.sub(x_sq, one);
        let p = expr_to_poly(&a, expr, x).unwrap();
        assert_eq!(p.degree(), Some(2));
        // Check roots.
        let val_1 = p.eval(&Ratio::from_integer(BigInt::from(1)));
        let val_neg1 = p.eval(&Ratio::from_integer(BigInt::from(-1)));
        assert!(val_1.is_zero(), "x²-1 at x=1 should be 0");
        assert!(val_neg1.is_zero(), "x²-1 at x=-1 should be 0");
    }

    // ── poly_to_expr ────────────────────────────────────────────────

    #[test]
    fn poly_to_expr_constant() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let p = Poly::from_int(7);
        let expr = poly_to_expr(&mut a, &p, x);
        assert_eq!(display(&a, expr), "7");
    }

    #[test]
    fn poly_to_expr_linear() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let p = Poly::from_coeffs(vec![
            Ratio::from_integer(BigInt::from(3)),
            Ratio::from_integer(BigInt::from(2)),
        ]);
        let expr = poly_to_expr(&mut a, &p, x);
        assert_eq!(display(&a, expr), "2*x + 3");
    }

    #[test]
    fn poly_to_expr_zero() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let p = Poly::zero();
        let expr = poly_to_expr(&mut a, &p, x);
        assert_eq!(expr, a.zero);
    }

    #[test]
    fn poly_to_expr_quadratic() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let p = Poly::from_coeffs(vec![
            Ratio::from_integer(BigInt::from(1)),
            Ratio::from_integer(BigInt::from(2)),
            Ratio::from_integer(BigInt::from(1)),
        ]);
        let expr = poly_to_expr(&mut a, &p, x);
        assert_eq!(display(&a, expr), "x^2 + 2*x + 1");
    }

    #[test]
    fn roundtrip_expr_poly_expr() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let three = a.int(3);
        // x^2 + 3*x + 2
        let x_sq = a.pow(x, two);
        let three_x = a.mul(&[three, x]);
        let orig = a.add(&[x_sq, three_x, two]);

        let poly = expr_to_poly(&a, orig, x).unwrap();
        let rebuilt = poly_to_expr(&mut a, &poly, x);

        // Should produce the same canonical expression.
        assert_eq!(orig, rebuilt, "roundtrip should preserve expression");
    }

    // ── as_numer_denom ──────────────────────────────────────────────

    #[test]
    fn numer_denom_plain_symbol() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let (n, d) = as_numer_denom(&mut a, x);
        assert_eq!(n, x);
        assert_eq!(d, a.one);
    }

    #[test]
    fn numer_denom_inverse() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let neg_one = a.int(-1);
        let expr = a.pow(x, neg_one); // x^(-1)
        let (n, d) = as_numer_denom(&mut a, expr);
        assert_eq!(n, a.one, "numerator of 1/x should be 1");
        assert_eq!(d, x, "denominator of 1/x should be x");
    }

    #[test]
    fn numer_denom_fraction() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        // x / y = x * y^(-1)
        let expr = a.div(x, y);
        let (n, d) = as_numer_denom(&mut a, expr);
        assert_eq!(display(&a, n), "x");
        assert_eq!(display(&a, d), "y");
    }

    // ── cancel ──────────────────────────────────────────────────────

    #[test]
    fn cancel_x_squared_minus_1_over_x_minus_1() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let one = a.one;
        // (x^2 - 1) / (x - 1)
        let x_sq = a.pow(x, two);
        let numer = a.sub(x_sq, one);
        let denom = a.sub(x, one);
        let expr = a.div(numer, denom);

        let result = cancel(&mut a, expr, x);
        let s = display(&a, result);
        // Should simplify to x + 1.
        assert_eq!(s, "x + 1", "cancel should give x + 1, got: {s}");
    }

    #[test]
    fn cancel_no_common_factor() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let one = a.one;
        // (x^2 + 1) / (x + 1) — no common factor.
        let x_sq = a.pow(x, two);
        let numer = a.add(&[x_sq, one]);
        let denom = a.add(&[x, one]);
        let expr = a.div(numer, denom);

        let result = cancel(&mut a, expr, x);
        // Should be unchanged.
        assert_eq!(result, expr, "no common factor → unchanged");
    }

    #[test]
    fn cancel_already_simple() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x + 1 — no denominator.
        let expr = a.add(&[x, a.one]);
        let result = cancel(&mut a, expr, x);
        assert_eq!(result, expr, "no denominator → unchanged");
    }

    #[test]
    fn cancel_quadratic_common_factor() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let three = a.int(3);
        let six = a.int(6);

        // numer = x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
        // denom = x^2 - 3x + 2 = (x-1)(x-2)
        // cancel → x - 3
        let x2 = a.pow(x, two);
        let x3 = a.pow(x, three);
        let six_x2 = a.mul(&[six, x2]);
        let eleven = a.int(11);
        let eleven_x = a.mul(&[eleven, x]);
        let neg_six_x2 = a.neg(six_x2);
        let neg_six = a.neg(six);
        let numer = a.add(&[x3, neg_six_x2, eleven_x, neg_six]);

        let three_x = a.mul(&[three, x]);
        let neg_three_x = a.neg(three_x);
        let denom = a.add(&[x2, neg_three_x, two]);

        let expr = a.div(numer, denom);
        let result = cancel(&mut a, expr, x);
        let s = display(&a, result);

        // The result should be x - 3 = -3 + x.
        assert!(
            s.contains('x') && s.contains('3'),
            "cancel should give x - 3, got: {s}"
        );
        // Verify numerically: at x=10, (10-1)(10-2)(10-3)/((10-1)(10-2)) = 7.
        let ten = a.int(10);
        let val = crate::transforms::subs::subs(&mut a, result, x, ten);
        assert_eq!(display(&a, val), "7", "at x=10, x-3 = 7");
    }

    #[test]
    fn cancel_with_coefficients() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        // (2*x^2 - 2) / (2*x - 2) = (2*(x^2 - 1)) / (2*(x - 1))
        //   = (2*(x-1)*(x+1)) / (2*(x-1)) = x + 1
        let x_sq = a.pow(x, two);
        let two_x_sq = a.mul(&[two, x_sq]);
        let numer = a.sub(two_x_sq, two);
        let two_x = a.mul(&[two, x]);
        let denom = a.sub(two_x, two);
        let expr = a.div(numer, denom);

        let result = cancel(&mut a, expr, x);
        let s = display(&a, result);
        assert_eq!(s, "x + 1", "cancel should give x + 1, got: {s}");
    }

    #[test]
    fn cancel_non_polynomial_returns_unchanged() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // sin(x) / x — not polynomial in numerator.
        let sin_x = a.sin(x);
        let expr = a.div(sin_x, x);
        let result = cancel(&mut a, expr, x);
        assert_eq!(result, expr, "non-polynomial should be unchanged");
    }

    #[test]
    fn cancel_constant_factor() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let two_x = a.mul(&[two, x]);
        let two_x_plus_2 = a.add(&[two_x, two]); // 2x + 2
        let frac = a.div(two_x_plus_2, two); // (2x+2)/2
        let result = cancel(&mut a, frac, x);
        let expected = a.add(&[x, a.one]); // x + 1
        assert_eq!(result, expected, "(2x+2)/2 should cancel to x+1");
    }

    #[test]
    fn together_uses_lcm_not_product() {
        // together() uses polynomial LCM so the denominator is minimal.
        // a/(x-1) + b/(x-1)^2 should give denom (x-1)^2, not (x-1)^3.
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let sym_a = sym(&mut a, "a");
        let sym_b = sym(&mut a, "b");
        let one = a.one;
        let two = a.int(2);
        let x_minus_1 = a.sub(x, one);
        let x_minus_1_sq = a.pow(x_minus_1, two);
        let frac1 = a.div(sym_a, x_minus_1); // a/(x-1)
        let frac2 = a.div(sym_b, x_minus_1_sq); // b/(x-1)^2
        let sum = a.add(&[frac1, frac2]);
        let result = together(&mut a, sum);
        let s = display(&a, result);
        // The denominator should be (x-1)^2, so no ^3 should appear.
        assert!(
            !s.contains("^3"),
            "together should use LCM not product. Got: {s}"
        );
    }

    #[test]
    fn together_simplifies_common_factors() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let sym_a = sym(&mut a, "a");
        let sym_b = sym(&mut a, "b");
        let one = a.one;
        let x_minus_1 = a.sub(x, one);
        let two = a.int(2);
        let x_minus_1_sq = a.pow(x_minus_1, two);
        let frac1 = a.div(sym_a, x_minus_1);
        let frac2 = a.div(sym_b, x_minus_1_sq);
        let sum = a.add(&[frac1, frac2]);
        let result = together(&mut a, sum);
        // After together + cancel, the denominator should be (x-1)^2, not (x-1)^3
        let result_str = display(&a, result);
        // The result should not have (x-1)^3 in it
        assert!(
            !result_str.contains("^3"),
            "together should use LCM not product. Got: {result_str}"
        );
    }
}

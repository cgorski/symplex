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
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};
use crate::base::walk;
use crate::poly::Poly;
use crate::poly::multipoly::{GrevLex, MultiPoly};

// ═══════════════════════════════════════════════════════════════════════════
// Expression → Poly
// ═══════════════════════════════════════════════════════════════════════════

/// The largest degree [`expr_to_poly`] builds a power to.  A dense
/// polynomial of higher degree is never what a caller can use, and
/// building one is not: `1/(x^1000000000 + 1)` asked the integrator's
/// rational routes for a billion coefficients.
pub(crate) const MAX_EXPR_POLY_DEGREE: usize = 10_000;

/// Try to convert an expression into a univariate polynomial in `var`.
///
/// Returns `None` if the expression contains terms that are not
/// polynomial in `var` (e.g., `sin(x)`, `x^(1/2)`, `x^y`), or a power of
/// degree above [`MAX_EXPR_POLY_DEGREE`].
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
            let n = usize::try_from(n).ok()?;
            let base_degree = base_poly.degree().unwrap_or(0);
            if n > MAX_EXPR_POLY_DEGREE || base_degree.checked_mul(n)? > MAX_EXPR_POLY_DEGREE {
                return None;
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
        | ExprNode::RootOf(_, _, _)
        | ExprNode::DSolve(_, _, _)
        | ExprNode::RootSum(_, _, _)
        | ExprNode::ConditionSet(_, _)
        | ExprNode::Subs(_, _, _) => None,
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

/// Why `expr` is not a polynomial in `gens`, for error messages: the first
/// generator found inside a function, under a negative or non-integer
/// power, or in an exponent.  `None` if no such position exists (the
/// expression *is* polynomial in `gens`).
pub(crate) fn non_polynomial_reason(
    arena: &Arena,
    expr: ExprId,
    gens: &[ExprId],
) -> Option<String> {
    let mut visited: rustc_hash::FxHashSet<ExprId> = rustc_hash::FxHashSet::default();
    let mut stack: Vec<ExprId> = vec![expr];
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        let node = arena.node(id).clone();
        match node {
            ExprNode::Add(_)
            | ExprNode::Mul(_)
            | ExprNode::Neg(_)
            | ExprNode::Symbol(_)
            | ExprNode::Num(_) => {
                stack.extend(node.children());
            }
            ExprNode::Pow(base, exp) => {
                let shown = |g: ExprId| arena.display(g).to_string();
                if let Some(&g) = gens.iter().find(|&&g| contains_any(arena, exp, &[g])) {
                    return Some(format!(
                        "generator `{}` occurs in the exponent of `{}`",
                        shown(g),
                        arena.display(id)
                    ));
                }
                if let Some(&g) = gens.iter().find(|&&g| contains_any(arena, base, &[g])) {
                    let ok = arena
                        .as_num(exp)
                        .is_some_and(|n| n.is_integer() && !n.is_negative());
                    if !ok {
                        let kind = match arena.as_num(exp) {
                            Some(n) if n.is_integer() => "a negative power (a rational function)",
                            Some(_) => "a fractional power (an algebraic function)",
                            None => "a symbolic power",
                        };
                        return Some(format!(
                            "generator `{}` occurs under {kind} in `{}`",
                            shown(g),
                            arena.display(id)
                        ));
                    }
                }
                stack.push(base);
            }
            other => {
                if let Some(&g) = gens.iter().find(|&&g| contains_any(arena, id, &[g])) {
                    return Some(format!(
                        "generator `{}` occurs inside the function `{}`",
                        arena.display(g),
                        arena.display(id)
                    ));
                }
                stack.extend(other.children());
            }
        }
    }
    None
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
    arena: &mut Arena,
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
    match arena.node(factor).clone() {
        ExprNode::Pow(base, exp) => {
            let n = arena.as_num(exp)?.clone();
            if !n.is_integer() || n.is_negative() {
                return None;
            }
            let d: u32 = n.to_integer().try_into().ok()?;
            if let Some(i) = gens.iter().position(|&g| g == base) {
                exps[i] = exps[i].checked_add(d)?;
                return Some(());
            }
            // `(g₁·g₂·c)^d`: a power of a product distributes over its
            // factors (canonicalisation only does this for small exponents).
            if let ExprNode::Mul(children) = arena.node(base).clone() {
                let mut inner_exps = vec![0u32; gens.len()];
                let mut inner_consts: SmallVec<[ExprId; 4]> = SmallVec::new();
                for &c in &children {
                    accumulate_factor(arena, c, gens, &mut inner_exps, &mut inner_consts)?;
                }
                for (e, ie) in exps.iter_mut().zip(&inner_exps) {
                    *e = e.checked_add(ie.checked_mul(d)?)?;
                }
                if !inner_consts.is_empty() {
                    let c = if inner_consts.len() == 1 {
                        inner_consts[0]
                    } else {
                        arena.mul(&inner_consts)
                    };
                    consts.push(arena.pow(c, exp));
                }
                return Some(());
            }
            None
        }
        ExprNode::Neg(inner) => {
            consts.push(arena.neg_one);
            accumulate_factor(arena, inner, gens, exps, consts)
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
    // An identically vanishing denominator is not a rational function:
    // `gcd(0, 0) = 0` would be the divisor below.
    let denom_poly = match expr_to_poly(arena, denom, var) {
        Some(p) if !p.is_zero() => p,
        _ => return expr,
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
    let (n, d) = match try_together_poly_lcm(arena, &parts, &unique_denoms) {
        Some(nd) => nd,
        None => together_product_fallback(arena, &parts, &unique_denoms),
    };
    arena.div(n, d)
}

// ═══════════════════════════════════════════════════════════════════════════
// Deep fraction decomposition (public `as_numer_denom` / `together`)
// ═══════════════════════════════════════════════════════════════════════════

/// Deep `(numerator, denominator)` decomposition with the semantics of
/// SymPy's `as_numer_denom`:
///
/// * a rational literal `p/q` is `(p, q)`;
/// * a sum is combined over a common denominator (polynomial LCM when the
///   denominators are univariate, product of the distinct denominators
///   otherwise), recursively, so nested fractions are flattened;
/// * a product multiplies numerators and denominators (a rational
///   coefficient `p/q` contributes `p` above and `q` below);
/// * an integer power `f^k` is `(n^k, d^k)`, swapped for negative `k`;
/// * everything else (symbols, constants, function applications, powers
///   with a non-integer exponent) is opaque: `(expr, 1)`.
///
/// No common factors are cancelled; use `ratsimp` for that.  The walk is
/// iterative (post-order with a cache).
pub(crate) fn fraction_parts(arena: &mut Arena, expr: ExprId) -> (ExprId, ExprId) {
    fraction_parts_tracked(arena, expr, &mut Vec::new())
}

/// [`fraction_parts`], also collecting in `cancelled` the denominator of
/// every quotient met on the way whose numerator cancelled to a structural
/// 0 (`(x/(x + 1) + 1/(x + 1) − 1)/s` is `(0, (x + 1)·s)`), paired with its
/// node, and the base of a negative non-integer power multiplied by such a
/// 0 (`0·s^(−1/2)`).  Such a quotient leaves no trace in the combined
/// fraction (`0` over anything is `0`), though it is `0/0` wherever its
/// denominator is 0.
fn fraction_parts_tracked(
    arena: &mut Arena,
    expr: ExprId,
    cancelled: &mut Vec<(ExprId, ExprId)>,
) -> (ExprId, ExprId) {
    let one = arena.one;
    let order = walk::post_order_ids(arena, expr);
    let mut cache: FxHashMap<ExprId, (ExprId, ExprId)> = FxHashMap::default();
    let lookup = |cache: &FxHashMap<ExprId, (ExprId, ExprId)>, id: ExprId| {
        cache.get(&id).copied().unwrap_or((id, one))
    };

    for &id in &order {
        let node = arena.node(id).clone();
        let parts = match node {
            ExprNode::Num(nid) => {
                let r = arena.num(nid).clone();
                if r.is_integer() {
                    (id, one)
                } else {
                    let n = arena.big_int(r.numer().clone());
                    let d = arena.big_int(r.denom().clone());
                    (n, d)
                }
            }
            ExprNode::Add(children) => {
                let parts: Vec<(ExprId, ExprId)> =
                    children.iter().map(|&c| lookup(&cache, c)).collect();
                combine_fraction_sum(arena, &parts)
            }
            ExprNode::Mul(children) => {
                let mut numers: SmallVec<[ExprId; 6]> = SmallVec::new();
                let mut denoms: SmallVec<[ExprId; 6]> = SmallVec::new();
                for &c in children.iter() {
                    let (n, d) = lookup(&cache, c);
                    if n != one {
                        numers.push(n);
                    }
                    if d != one {
                        denoms.push(d);
                    }
                }
                (
                    product_or_one(arena, &numers),
                    product_or_one(arena, &denoms),
                )
            }
            ExprNode::Pow(base, exp) => match arena.as_num(exp).cloned() {
                Some(k) if k.is_integer() && !k.is_zero() => {
                    let (bn, bd) = lookup(&cache, base);
                    if bn == base && bd == one {
                        // Opaque base with a plain integer power: keep as is
                        // (negative exponents are the denominator).
                        if k.is_negative() {
                            let m = arena.big_int(-k.to_integer());
                            (one, arena.pow(base, m))
                        } else {
                            (id, one)
                        }
                    } else {
                        let m = arena.big_int(k.to_integer().abs());
                        let bn_m = arena.pow(bn, m);
                        let bd_m = arena.pow(bd, m);
                        if k.is_negative() {
                            (bd_m, bn_m)
                        } else {
                            (bn_m, bd_m)
                        }
                    }
                }
                _ => (id, one),
            },
            ExprNode::Neg(inner) => {
                let (n, d) = lookup(&cache, inner);
                (arena.neg(n), d)
            }
            _ => (id, one),
        };
        if arena.is_zero_structural(parts.0) {
            if parts.1 != one {
                cancelled.push((id, parts.1));
            }
            // A negative non-integer power is opaque here (a factor of the
            // numerator): `0·s^(−1/2)` drops it as well.
            if let ExprNode::Mul(children) = arena.node(id) {
                for &c in children.iter() {
                    if let ExprNode::Pow(b, e) = *arena.node(c)
                        && arena
                            .as_num(e)
                            .is_some_and(|q| q.is_negative() && !q.is_integer())
                    {
                        cancelled.push((id, b));
                    }
                }
            }
        }
        cache.insert(id, parts);
    }
    lookup(&cache, expr)
}

/// The nodes of the rational skeleton of `expr`: those reached from it
/// through sums, products, negations and integer powers (not function
/// arguments, nor the bases of other powers).  An explicit stack.
fn rational_skeleton(arena: &Arena, expr: ExprId) -> FxHashSet<ExprId> {
    let mut seen: FxHashSet<ExprId> = FxHashSet::default();
    let mut stack = vec![expr];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Add(children) | ExprNode::Mul(children) => stack.extend(children.iter()),
            ExprNode::Neg(child) => stack.push(*child),
            ExprNode::Pow(base, e) if arena.as_num(*e).is_some_and(|q| q.is_integer()) => {
                stack.push(*base);
            }
            _ => {}
        }
    }
    seen
}

/// Is some factor of `d` identically 0: as a polynomial in its generators
/// ([`vanishes_identically`]), or, when it holds a function or a radical,
/// by an identity of its functions
/// ([`Arena::vanishes_by_identity`]: `tan x·cos x − sin x`, `cosh²x −
/// sinh²x − 1`).
fn has_vanishing_factor(arena: &mut Arena, d: ExprId) -> bool {
    let factors: Vec<ExprId> = match arena.node(d) {
        ExprNode::Mul(children) => children.to_vec(),
        _ => vec![d],
    };
    factors.into_iter().any(|f| {
        let base = match *arena.node(f) {
            ExprNode::Pow(b, e) if arena.as_num(e).is_some_and(|q| q.is_integer()) => b,
            _ => f,
        };
        if arena.node(base).is_atom() {
            return arena.is_zero_structural(base);
        }
        if vanishes_identically(arena, base) {
            return true;
        }
        let transcendental = walk::post_order_ids(arena, base).into_iter().any(|id| {
            !matches!(
                arena.node(id),
                ExprNode::Num(_)
                    | ExprNode::Symbol(_)
                    | ExprNode::Add(_)
                    | ExprNode::Mul(_)
                    | ExprNode::Neg(_)
            ) && !matches!(arena.node(id), ExprNode::Pow(_, e) if arena.as_num(*e).is_some_and(|q| q.is_integer()))
        });
        transcendental && arena.vanishes_by_identity(base)
    })
}

/// Deep [`together`]: `fraction_parts` rebuilt as a single quotient.
///
/// Note that a purely numeric common denominator cannot survive
/// canonicalisation (`(3x + 2)/6` is stored as `1/2*x + 1/3`), so for such
/// inputs the result prints like the input; `fraction_parts` still reports
/// `(3x + 2, 6)`.
///
/// A denominator that cancels to `0` (`−x/(x + 1) + x·(−x/(x + 1) + 1)`)
/// gives `zoo` for a numerator that does not vanish and `nan` for one that
/// vanishes identically as a polynomial in its generators (`x·(x + 1) −
/// x² − x`), as `ratsimp` does: `0/0` is undefined for every value of the
/// variables.  Up to 0.31 such a quotient was `zoo` here (the numerator
/// was not multiplied out) and `nan` from `ratsimp` (SymPy 1.14 gives
/// `nan` from `together` but `0` from `ratsimp` and `cancel`, which return
/// a zero numerator before they look at the denominator).
///
/// So does a denominator that is 0 only once multiplied out
/// ([`vanishes_identically`]): up to 0.31 `(x·(x + 1) − x² − x)/(x·(x + 2)
/// − x² − 2x)` was returned unchanged and `x + 1/(x·(x + 1) − x² − x)`
/// became `(x·(x·(x + 1) − x² − x) + 1)/(x·(x + 1) − x² − x)`, while
/// `ratsimp` gave `nan` and `zoo`.
///
/// So does a quotient whose numerator cancels to 0 over a denominator that
/// vanishes identically, by an identity of its functions or as a
/// polynomial: `0/0` anywhere in the rational skeleton is `nan` for every
/// value of the variables, while the combined fraction keeps no trace of
/// it.  Up to 0.37 `together((x/(x + 1) + 1/(x + 1) − 1)/(tan x·cos x −
/// sin x))` was `0`, and `together((cos x + 2)·(x² + 1) − (x/(x − 1) +
/// x·(−x/(x − 1) + 1))/(cos 2x − cos²x + sin²x))` was `(x² + 1)·(cos x +
/// 2)`, as `ratsimp`, `expand` and `simplify` give `nan`.  Only the
/// denominators of cancelled quotients are tested (rare), one certified
/// evaluation each when they hold a function.  Likewise a numerator with a
/// factor that vanishes by an identity over a denominator that is 0 is
/// `0/0`: `together(((x/(x + 1) + 1/(x + 1) − 1)/(tan x·cos x − sin
/// x))⁻¹)` was `zoo`.
pub(crate) fn together_deep(arena: &mut Arena, expr: ExprId) -> ExprId {
    let mut cancelled = Vec::new();
    let (n, d) = fraction_parts_tracked(arena, expr, &mut cancelled);
    if !cancelled.is_empty() {
        let skeleton = rational_skeleton(arena, expr);
        let mut tested: FxHashSet<ExprId> = FxHashSet::default();
        for (node, den) in cancelled {
            if skeleton.contains(&node) && tested.insert(den) && has_vanishing_factor(arena, den) {
                return arena.nan;
            }
        }
    }
    if d == arena.one {
        return n;
    }
    if arena.is_zero_structural(d) {
        if !arena.is_zero_structural(n)
            && (expands_to_zero(arena, n) || has_vanishing_factor(arena, n))
        {
            return arena.nan;
        }
    } else if vanishes_identically(arena, d) {
        if expands_to_zero(arena, n) || has_vanishing_factor(arena, n) {
            return arena.nan;
        }
        let zero = arena.zero;
        return arena.div(n, zero);
    }
    arena.div(n, d)
}

/// Is `expr` structurally 0 once multiplied out (`expand`, then `eval`)?
fn expands_to_zero(arena: &mut Arena, expr: ExprId) -> bool {
    let expanded = crate::transforms::expand::expand(arena, expr);
    let expanded = crate::transforms::eval::eval(arena, expanded);
    arena.is_zero_structural(expanded)
}

/// Does `expr` vanish identically as a rational function of its
/// generators — the maximal subexpressions that are not numbers, sums,
/// products, negations or integer powers (the free symbols, `sin x`, `√x`,
/// `π`), taken as independent indeterminates as `ratsimp` takes them?
/// `x·(x + 1) − x² − x` does, `sin²x + cos²x − 1` does not.
///
/// Decided by [`may_vanish_identically`] (a residue at a pseudo-random
/// point, linear in the size of `expr`) and, only when that residue is 0,
/// confirmed by multiplying out: `true` only for an `expr` that is
/// structurally 0 after `expand` and `eval`.
pub(crate) fn vanishes_identically(arena: &mut Arena, expr: ExprId) -> bool {
    if arena.is_zero_structural(expr) {
        return true;
    }
    may_vanish_identically(arena, expr) && expands_to_zero(arena, expr)
}

/// The Mersenne prime `2⁶¹ − 1`, the modulus of [`may_vanish_identically`].
const RESIDUE_PRIME: u64 = (1 << 61) - 1;

/// The exponent scale `D` of [`residues_at_random_point`]: a generator `g`
/// takes the value `ρ_g^D` for a pseudo-random unit `ρ_g`, so that its
/// rational powers `g^(a/b)` take `ρ_g^(a·D/b)`, and `exp(c·m)` takes
/// `ρ_m^(c·D)` (see [`residue_exponent`]).  `720720 = lcm(1, …, 16)`; the
/// values of the generators range over the `D`-th powers, a subgroup of
/// `(2⁶¹ − 2)/90090 ≈ 2.6·10¹³` units.
const RESIDUE_ROOT_SCALE: u64 = 720_720;

/// Salts of the two kinds of keys of [`residues_at_random_point`]: the
/// `ExprId` of a generator, and the monomial `m` of a term `c·m` of an
/// exponent (so that `exp(x)` and `x` are unrelated).
const RESIDUE_GENERATOR_SALT: u64 = 0x5851_F42D_4C95_7F2D;
const RESIDUE_EXPONENT_SALT: u64 = 0x1405_7B7E_F767_814F;

/// Salt of the angle units `e^{iu}` behind `sin u`, `cos u`, `tan u` (see
/// [`generator_residue`]): unrelated to the units of the exponentials.
const RESIDUE_ANGLE_SALT: u64 = 0x2F3B_9D07_A6C1_E485;

/// `a·b mod p` for the Mersenne prime `p = 2⁶¹ − 1`: `2⁶¹ ≡ 1`, so the
/// high bits fold onto the low ones (twice, for any `u64` operands), and
/// one subtraction finishes — the value of `(a·b) % p` without a 128-bit
/// division.
fn residue_mul(a: u64, b: u64) -> u64 {
    let p = RESIDUE_PRIME;
    let x = u128::from(a) * u128::from(b);
    let folded = (x & u128::from(p)) + (x >> 61);
    let mut r = ((folded & u128::from(p)) + (folded >> 61)) as u64;
    while r >= p {
        r -= p;
    }
    r
}

fn residue_pow(mut b: u64, mut e: u64) -> u64 {
    let mut r = 1u64;
    while e > 0 {
        if e & 1 == 1 {
            r = residue_mul(r, b);
        }
        b = residue_mul(b, b);
        e >>= 1;
    }
    r
}

/// splitmix64.
fn residue_hash(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A pseudo-random unit modulo the prime for the key `key`.
fn residue_unit(key: u64) -> u64 {
    residue_hash(key) % (RESIDUE_PRIME - 1) + 1
}

/// `q·D` ([`RESIDUE_ROOT_SCALE`]) as an exponent modulo the group order
/// `N = 2⁶¹ − 2`: for `q = n/d` and `g = gcd(d, D)`, `n·(D/g)·(d/g)⁻¹ mod
/// N`, when `d/g` is invertible modulo `N`.  The map is additive on those
/// rationals (the ring map from the integers localised at the primes of
/// `d/g`), so `ρ^(q₁·D)·ρ^(q₂·D) = ρ^((q₁ + q₂)·D)` and `(ρ^(D/d))^d = ρ^D`:
/// `exp(−46/11)·exp(−13/7) = exp(−465/77)` holds among the residues.
/// `None` otherwise (a denominator with a factor `16`, `31`, …).
fn residue_exponent(q: &crate::base::numeric::Q) -> Option<u64> {
    use num_traits::ToPrimitive;
    let order = RESIDUE_PRIME - 1;
    // An integer `n` (the common coefficient): `n·D mod N`, the value of
    // the general formula with `d = 1`, without big integers.
    if q.is_integer()
        && let Some(n) = q.numer().to_i128()
    {
        let n = n.rem_euclid(i128::from(order)) as u128;
        return Some(((n * u128::from(RESIDUE_ROOT_SCALE)) % u128::from(order)) as u64);
    }
    let order_big = BigInt::from(order);
    let scale = BigInt::from(RESIDUE_ROOT_SCALE);
    let g = q.denom().gcd(&scale);
    let rest = (q.denom() / &g).mod_floor(&order_big).to_u64()?;
    let inv = inverse_modulo(rest, order)?;
    let k = (q.numer() * (scale / g)).mod_floor(&order_big).to_u64()?;
    Some(((u128::from(k) * u128::from(inv)) % u128::from(order)) as u64)
}

/// `a⁻¹ mod m` by the extended Euclidean algorithm, `None` when `gcd(a, m)
/// ≠ 1`.
fn inverse_modulo(a: u64, m: u64) -> Option<u64> {
    let (mut r0, mut r1) = (i128::from(m), i128::from(a));
    let (mut t0, mut t1) = (0i128, 1i128);
    while r1 != 0 {
        let q = r0 / r1;
        (r0, r1) = (r1, r0 - q * r1);
        (t0, t1) = (t1, t0 - q * t1);
    }
    (r0 == 1).then(|| t0.rem_euclid(i128::from(m)) as u64)
}

/// `u^(q·D)` ([`residue_exponent`]) for a unit `u`.
fn residue_scaled_power(u: u64, q: &crate::base::numeric::Q) -> Option<u64> {
    residue_exponent(q).map(|k| residue_pow(u, k))
}

/// An element `a + b·i` of `F_p[i] = F_{p²}`, `p = 2⁶¹ − 1`: `−1` is not a
/// square modulo `p ≡ 3 (mod 4)`, so `i² = −1` defines the field with `p²`
/// elements, in which every rational has a square root.  The imaginary
/// unit and the square roots of integers take their true values there.
type Residue = (u64, u64);

fn residue2_add(x: Residue, y: Residue) -> Residue {
    ((x.0 + y.0) % RESIDUE_PRIME, (x.1 + y.1) % RESIDUE_PRIME)
}

fn residue2_neg(x: Residue) -> Residue {
    (
        (RESIDUE_PRIME - x.0) % RESIDUE_PRIME,
        (RESIDUE_PRIME - x.1) % RESIDUE_PRIME,
    )
}

fn residue2_mul(x: Residue, y: Residue) -> Residue {
    let p = RESIDUE_PRIME;
    let re = (residue_mul(x.0, y.0) + p - residue_mul(x.1, y.1)) % p;
    let im = (residue_mul(x.0, y.1) + residue_mul(x.1, y.0)) % p;
    (re, im)
}

/// `x⁻¹ = conj(x)/|x|²`; `None` for 0 (`a² + b² ≠ 0` otherwise, as `−1` is
/// not a square).
fn residue2_inv(x: Residue) -> Option<Residue> {
    let p = RESIDUE_PRIME;
    let norm = (residue_mul(x.0, x.0) + residue_mul(x.1, x.1)) % p;
    if norm == 0 {
        return None;
    }
    let n = residue_pow(norm, p - 2);
    Some((residue_mul(x.0, n), residue_mul((p - x.1) % p, n)))
}

fn residue2_pow(mut b: Residue, mut e: u128) -> Residue {
    let mut r: Residue = (1, 0);
    while e > 0 {
        if e & 1 == 1 {
            r = residue2_mul(r, b);
        }
        b = residue2_mul(b, b);
        e >>= 1;
    }
    r
}

/// A square root of the residue `s` of a positive integer in `F_{p²}`:
/// `s^((p+1)/4)` when `s` is a square modulo `p`, else `i·(−s)^((p+1)/4)`.
fn residue_sqrt(s: u64) -> Residue {
    let p = RESIDUE_PRIME;
    let t = residue_pow(s, (p + 1) / 4);
    if residue_mul(t, t) == s {
        (t, 0)
    } else {
        (0, residue_pow((p - s) % p, (p + 1) / 4))
    }
}

/// The value of `√n` for an integer `n ≥ 1`, multiplicative in `n`:
/// `√n = Π √qᵉ` over the primes `q < 1000` dividing `n` (with `e` reduced:
/// `√(q²·m) = q·√m`) times the root of the cofactor, so that `√2·√3` and
/// `√6` agree (`(√2 + √3)² − 5 − 2√6` has residue 0).  `√n² = n` always.
fn residue_integer_sqrt(n: &BigInt) -> Residue {
    use num_traits::ToPrimitive;
    let p = RESIDUE_PRIME;
    let Some(mut rest) = n.to_u64() else {
        // Beyond 64 bits: the root of the residue (still `√n² = n`).
        return residue_sqrt(n.mod_floor(&BigInt::from(p)).to_u64().unwrap_or(0));
    };
    let mut value: Residue = (1, 0);
    let mut q: u64 = 2;
    while q < 1000 && q.saturating_mul(q) <= rest {
        let mut e = 0u32;
        while rest.is_multiple_of(q) {
            rest /= q;
            e += 1;
        }
        if e > 0 {
            value = residue2_mul(value, (residue_pow(q, u64::from(e / 2)), 0));
            if e % 2 == 1 {
                value = residue2_mul(value, residue_sqrt(q));
            }
        }
        q += if q == 2 { 1 } else { 2 };
    }
    if rest > 1 {
        value = residue2_mul(value, residue_sqrt(rest % p));
    }
    value
}

/// Is the value of the generator `g` the plain `ρ_g^D` (not an exponential,
/// `e`, a rational power or a number, whose values are derived)?
fn is_plain_generator(arena: &Arena, g: ExprId) -> bool {
    !matches!(
        arena.node(g),
        ExprNode::Num(_)
            | ExprNode::Add(_)
            | ExprNode::Mul(_)
            | ExprNode::Neg(_)
            | ExprNode::Pow(_, _)
            | ExprNode::Exp(_)
            | ExprNode::E
            | ExprNode::Sin(_)
            | ExprNode::Cos(_)
            | ExprNode::Tan(_)
            | ExprNode::Sinh(_)
            | ExprNode::Cosh(_)
            | ExprNode::Tanh(_)
    )
}

/// The residue of the generator `id` (a node that is not a number, sum,
/// product, negation or integer power):
///
/// * `exp(Σ cⱼ·mⱼ)` is `Π ρ_{mⱼ}^(cⱼ·D)`, one unit per monomial `mⱼ` of the
///   exponent (its factors without the rational coefficient `cⱼ`; `m = 1`
///   for a number), and `e` is `ρ_1^D`: the identities `exp(2x) = exp(x)²`,
///   `exp(x + y) = exp(x)·exp(y)`, `exp(2) = e²` hold among the residues;
/// * `g^q` for a rational non-integer `q` and a plain generator `g` is
///   `ρ_g^(q·D)`: `(√x)² = x`, `√x·x^(3/2) = x²` hold (principal powers
///   satisfy `g^a·g^b = g^(a+b)` and `(g^a)^n = g^(a·n)` for integer `n`);
/// * `i` is `i` and `n^(m/2)` for a positive integer `n` is `(√n)^m` with
///   the square root taken in `F_{p²}` ([`residue_integer_sqrt`]): `(√2 +
///   √3)² − 5 − 2√6` and `(√2 + √3·i)² + 1 − 2√6·i` (what `subs` makes of
///   `(√x + √y)² − x − y − 2√x·√y`) have residue 0;
/// * `sin u`, `cos u`, `tan u` are `(t − t⁻¹)/(2i)`, `(t + t⁻¹)/2` and their
///   quotient for the angle unit `t = Π σ_{mⱼ}^(cⱼ·D)` of `u = Σ cⱼ·mⱼ`
///   (the value of `e^{iu}`, multiplicative in `u` as the exponentials,
///   with units `σ` of their own), and `sinh u`, `cosh u`, `tanh u` the same
///   in the unit `h` of `exp(u)` with `(h ∓ h⁻¹)/2`: `sin²u + cos²u − 1`,
///   `tan u·cos u − sin u`, `sin 2u − 2·sin u·cos u`, `cosh²u − sinh²u −
///   1` and `2·cosh u − eᵘ − e⁻ᵘ` have residue 0 whatever `u` is (also a
///   constant such as `besselj(10⁵, 1)`, which is never evaluated).
///   `t² ≠ −1` in `F_p` (`p ≡ 3 mod 4`), so `cos` and `cosh` are never 0
///   and `tan`, `tanh` always have a value.  Their non-integer powers are
///   generators of their own (as those of `exp`);
/// * any other generator, and the above when a coefficient times `D` has no
///   value ([`residue_exponent`]), is `ρ_id^D` for its own key.
///
/// The residue map is so a ring homomorphism on the expressions built from
/// the generators that respects those identities: an expression zero by
/// them has residue 0; a nonzero one rarely does.
fn generator_residue(arena: &Arena, id: ExprId) -> Residue {
    let plain = |g: ExprId| {
        residue_pow(
            residue_unit(u64::from(g.0) ^ RESIDUE_GENERATOR_SALT),
            RESIDUE_ROOT_SCALE,
        )
    };
    let derived = match arena.node(id) {
        ExprNode::ImaginaryUnit => return (0, 1),
        ExprNode::E => Some(residue_pow(
            residue_unit(RESIDUE_EXPONENT_SALT),
            RESIDUE_ROOT_SCALE,
        )),
        ExprNode::Exp(u) => exp_residue(arena, *u),
        ExprNode::Sin(u) | ExprNode::Cos(u) | ExprNode::Tan(u) => {
            if let Some(t) = exponent_unit(arena, *u, RESIDUE_ANGLE_SALT) {
                return circular_residue(arena.node(id), t);
            }
            None
        }
        ExprNode::Sinh(u) | ExprNode::Cosh(u) | ExprNode::Tanh(u) => {
            if let Some(h) = exp_residue(arena, *u) {
                return circular_residue(arena.node(id), h);
            }
            None
        }
        ExprNode::Pow(g, q) if is_plain_generator(arena, *g) => arena.as_num(*q).and_then(|q| {
            residue_scaled_power(residue_unit(u64::from(g.0) ^ RESIDUE_GENERATOR_SALT), q)
        }),
        ExprNode::Pow(n, q) => {
            if let (Some(n), Some(q)) = (arena.as_num(*n), arena.as_num(*q))
                && let Some(value) = numeric_radical_residue(n, q)
            {
                return value;
            }
            None
        }
        _ => None,
    };
    (derived.unwrap_or_else(|| plain(id)), 0)
}

/// The residue of `n^q` for a positive integer `n` and `q = m/2`: `(√n)^m`
/// ([`residue_integer_sqrt`]); `None` for any other number or exponent.
fn numeric_radical_residue(
    n: &crate::base::numeric::Q,
    q: &crate::base::numeric::Q,
) -> Option<Residue> {
    use num_traits::ToPrimitive;
    if !n.is_integer() || !n.is_positive() || *q.denom() != BigInt::from(2) {
        return None;
    }
    let root = residue_integer_sqrt(n.numer());
    let m = q.numer().to_i64()?;
    let base = if m < 0 { residue2_inv(root)? } else { root };
    Some(residue2_pow(base, u128::from(m.unsigned_abs())))
}

/// The residue of `sin u`, `cos u`, `tan u` from the angle unit `t` of `u`,
/// or of `sinh u`, `cosh u`, `tanh u` from the unit `t` of `exp(u)` (see
/// [`generator_residue`]); `t` is a unit of `F_p`, so `t + t⁻¹ ≠ 0`.
fn circular_residue(node: &ExprNode, t: u64) -> Residue {
    let p = RESIDUE_PRIME;
    // 2⁻¹ = (p + 1)/2
    let half = p.div_ceil(2);
    let inv = residue_pow(t, p - 2);
    let difference = (t + p - inv) % p;
    let sum = (t + inv) % p;
    let quotient = || residue_mul(difference, residue_pow(sum, p - 2));
    match node {
        // (t − t⁻¹)/(2i) = −i·(t − t⁻¹)/2
        ExprNode::Sin(_) => (0, residue_mul((p - difference) % p, half)),
        ExprNode::Cos(_) | ExprNode::Cosh(_) => (residue_mul(sum, half), 0),
        // sin/cos = −i·(t − t⁻¹)/(t + t⁻¹)
        ExprNode::Tan(_) => (0, (p - quotient()) % p),
        ExprNode::Sinh(_) => (residue_mul(difference, half), 0),
        _ => (quotient(), 0),
    }
}

/// The residue of `exp(u)` (see [`generator_residue`]), `None` when a
/// coefficient of the exponent times `D` is not an integer.
fn exp_residue(arena: &Arena, u: ExprId) -> Option<u64> {
    exponent_unit(arena, u, RESIDUE_EXPONENT_SALT)
}

/// `Π ρ_{mⱼ}^(cⱼ·D)` for the terms `cⱼ·mⱼ` of `u`, the units `ρ` keyed by
/// the monomials `mⱼ` with `salt` (`ρ_1` for a number): the residue of
/// `exp(u)` for [`RESIDUE_EXPONENT_SALT`], the angle unit of `sin u` for
/// [`RESIDUE_ANGLE_SALT`].  `None` when a coefficient times `D` has no
/// value ([`residue_exponent`]).
fn exponent_unit(arena: &Arena, u: ExprId, salt: u64) -> Option<u64> {
    let terms: &[ExprId] = match arena.node(u) {
        ExprNode::Add(children) => children,
        _ => std::slice::from_ref(&u),
    };
    let mut value = 1u64;
    for &t in terms {
        let (mut t, mut sign) = (t, crate::base::numeric::Q::one());
        if let ExprNode::Neg(inner) = arena.node(t) {
            t = *inner;
            sign = -sign;
        }
        let (coeff, key) = match arena.node(t) {
            ExprNode::Num(nid) => (arena.num(*nid).clone(), salt),
            ExprNode::Mul(children) => {
                let (coeff, rest) = match children.first().and_then(|&c| arena.as_num(c)) {
                    Some(q) => (q.clone(), &children[1..]),
                    None => (crate::base::numeric::Q::one(), &children[..]),
                };
                let key = rest
                    .iter()
                    .fold(salt, |h, c| residue_hash(h ^ u64::from(c.0)));
                (coeff, key)
            }
            _ => (
                crate::base::numeric::Q::one(),
                residue_hash(salt ^ u64::from(t.0)),
            ),
        };
        value = residue_mul(
            value,
            residue_scaled_power(residue_unit(key), &(coeff * sign))?,
        );
    }
    Some(value)
}

/// `false` when `expr` certainly does not vanish identically as a rational
/// function of its generators (see [`vanishes_identically`]): its value in
/// the field `F_{p²}` (`p = 2⁶¹ − 1`, [`Residue`]) at a point where every
/// generator takes a pseudo-random residue ([`generator_residue`]: a hash
/// of its `ExprId`, with exponentials, rational powers of a generator, `i`
/// and square roots of integers related as they are) is not 0.  A rational
/// function that vanishes identically is 0 at every point where its
/// denominators are invertible; a nonzero one of degree `d` is 0 at a
/// fraction of at most about `d/(2.6·10¹³)` of the points.  `true`
/// (undecided) when the residue is 0, or when a denominator or an inverted
/// subexpression is 0 there.
///
/// The exponentials make `exp(2x) − exp(x)²` and `exp(2) − exp(1)²` (which
/// the canonical form keeps: SymPy folds `exp(x)**2` to `exp(2*x)`, symplex
/// only in `eval`) residue 0, as `x·(x + 1) − x² − x`; the confirmation by
/// multiplying out and `eval` ([`vanishes_identically`]) then sees them.
/// Up to 0.33 every generator was independent (and the residues were
/// modulo `p`), so `(exp(2x) − exp(x)²)/(exp(2x) − exp(x)²)` was `1` and
/// `(exp(2) − exp(1)²)/0` was `zoo` (both `nan`).
pub(crate) fn may_vanish_identically(arena: &Arena, expr: ExprId) -> bool {
    let values = residues_at_random_point(arena, expr);
    !values
        .get(&expr)
        .copied()
        .flatten()
        .is_some_and(|r| r != (0, 0))
}

/// Is the base of some negative integer power in `expr`, outside every
/// generator, possibly identically zero ([`may_vanish_identically`])?
/// `false` certifies that no denominator of `expr` as a rational function
/// of its generators vanishes identically.  One pass over `expr`.
pub(crate) fn may_have_vanishing_denominator(arena: &Arena, expr: ExprId) -> bool {
    let values = residues_at_random_point(arena, expr);
    values.keys().any(|&id| match arena.node(id) {
        ExprNode::Pow(base, e) => {
            arena
                .as_num(*e)
                .is_some_and(|q| q.is_integer() && q.is_negative())
                && !values
                    .get(base)
                    .copied()
                    .flatten()
                    .is_some_and(|r| r != (0, 0))
        }
        _ => false,
    })
}

/// The residues in `F_{p²}` ([`Residue`], `p` = [`RESIDUE_PRIME`]) of
/// `expr` and of its structural subexpressions (numbers, sums, products,
/// negations, integer powers) at the point where each generator (any other
/// node) takes its residue ([`generator_residue`]); `None` for a node where
/// a rational literal's denominator or an inverted value is 0 there.  An
/// explicit post-order over the structural nodes; generators are leaves.
fn residues_at_random_point(arena: &Arena, expr: ExprId) -> FxHashMap<ExprId, Option<Residue>> {
    let p = RESIDUE_PRIME;
    let inv = |a: u64| (a != 0).then(|| residue_pow(a, p - 2));
    let big_mod = |n: &BigInt, m: u64| -> u64 {
        use num_traits::ToPrimitive;
        n.mod_floor(&BigInt::from(m)).to_u64().unwrap_or(0)
    };
    let integer_exponent = |e: ExprId| arena.as_num(e).filter(|q| q.is_integer());
    let structural = |id: ExprId| match arena.node(id) {
        ExprNode::Num(_) | ExprNode::Add(_) | ExprNode::Mul(_) | ExprNode::Neg(_) => true,
        ExprNode::Pow(_, e) => integer_exponent(*e).is_some(),
        _ => false,
    };
    let mut values: FxHashMap<ExprId, Option<Residue>> = FxHashMap::default();
    let mut stack: Vec<(ExprId, bool)> = vec![(expr, false)];
    while let Some((id, children_done)) = stack.pop() {
        if values.contains_key(&id) {
            continue;
        }
        if !structural(id) {
            // The same residue for the same subexpression (hash-consing),
            // unrelated ones for different generators.
            values.insert(id, Some(generator_residue(arena, id)));
            continue;
        }
        let node = arena.node(id);
        if !children_done {
            stack.push((id, true));
            match node {
                ExprNode::Pow(base, _) => stack.push((*base, false)),
                n => stack.extend(n.children().into_iter().map(|c| (c, false))),
            }
            continue;
        }
        let get = |c: &ExprId| values.get(c).copied().flatten();
        let value = match node {
            ExprNode::Num(nid) => {
                let q = arena.num(*nid);
                inv(big_mod(q.denom(), p)).map(|d| (residue_mul(big_mod(q.numer(), p), d), 0))
            }
            ExprNode::Add(children) => children
                .iter()
                .try_fold((0u64, 0u64), |acc, c| get(c).map(|v| residue2_add(acc, v))),
            ExprNode::Mul(children) => children
                .iter()
                .try_fold((1u64, 0u64), |acc, c| get(c).map(|v| residue2_mul(acc, v))),
            ExprNode::Neg(c) => get(c).map(residue2_neg),
            ExprNode::Pow(base, e) => match (get(base), integer_exponent(*e)) {
                (Some(b), Some(k)) => {
                    let n = k.to_integer();
                    let b = if n.is_negative() {
                        residue2_inv(b)
                    } else {
                        Some(b)
                    };
                    // b^(p²−1) = 1 for b ≠ 0 in F_{p²}.
                    b.map(|b| match (b, n.is_zero()) {
                        (_, true) => (1, 0),
                        ((0, 0), false) => (0, 0),
                        _ => {
                            let e =
                                num_traits::ToPrimitive::to_u128(&n.abs()).unwrap_or_else(|| {
                                    let order = BigInt::from(p) * BigInt::from(p) - BigInt::one();
                                    num_traits::ToPrimitive::to_u128(&n.abs().mod_floor(&order))
                                        .unwrap_or(0)
                                });
                            residue2_pow(b, e)
                        }
                    })
                }
                _ => None,
            },
            _ => None,
        };
        values.insert(id, value);
    }
    values
}

fn product_or_one(arena: &mut Arena, factors: &[ExprId]) -> ExprId {
    match factors.len() {
        0 => arena.one,
        1 => factors[0],
        _ => arena.mul(factors),
    }
}

/// Combine `Σ nᵢ/dᵢ` into one `(numerator, denominator)` pair.
fn combine_fraction_sum(arena: &mut Arena, parts: &[(ExprId, ExprId)]) -> (ExprId, ExprId) {
    let one = arena.one;
    if parts.iter().all(|&(_, d)| d == one) {
        let numers: SmallVec<[ExprId; 6]> = parts.iter().map(|&(n, _)| n).collect();
        return (arena.add(&numers), one);
    }
    let mut unique_denoms: Vec<ExprId> = Vec::new();
    for &(_, d) in parts {
        if d != one && !unique_denoms.contains(&d) {
            unique_denoms.push(d);
        }
    }
    match try_together_poly_lcm(arena, parts, &unique_denoms) {
        Some(nd) => nd,
        None => together_product_fallback(arena, parts, &unique_denoms),
    }
}

/// Attempt to combine fractions using polynomial LCM of the denominators.
///
/// Returns `Some((numerator, denominator))` on success, `None` if polynomial
/// conversion fails for any denominator (e.g. multiple variables,
/// transcendental denoms).
fn try_together_poly_lcm(
    arena: &mut Arena,
    parts: &[(ExprId, ExprId)],
    unique_denoms: &[ExprId],
) -> Option<(ExprId, ExprId)> {
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
        // A denominator that vanishes identically (`x(x + 1) − x² − x`, or
        // the literal 0 of `1/(x/(x + 1) − x/(x + 1))`, whose inner sum
        // cancelled) has no LCM and divides nothing: keep the product form,
        // where the arena folds a literal zero denominator into `zoo`.
        if p.is_zero() {
            return None;
        }
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
    Some((numer_sum, common_denom_expr))
}

/// Fallback: use the product of distinct denominators as the common denominator.
fn together_product_fallback(
    arena: &mut Arena,
    parts: &[(ExprId, ExprId)],
    unique_denoms: &[ExprId],
) -> (ExprId, ExprId) {
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
    (numer_sum, common_denom)
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

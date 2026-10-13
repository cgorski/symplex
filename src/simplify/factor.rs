//! Polynomial factoring over ℤ for symbolic expressions.
//!
//! This module implements [`factor`], [`factor_auto`] and [`factor_list`],
//! which factor a polynomial expression over ℤ into a product of irreducible
//! factors with their multiplicities.
//!
//! # Algorithm
//!
//! 1. Convert the expression to a univariate [`Poly`] in the given variable.
//!    If that succeeds, call [`Poly::factor_over_z`] (content extraction,
//!    Yun's square-free decomposition, Berlekamp–Zassenhaus — see
//!    [`crate::poly::factor_zassenhaus`]).
//! 2. Otherwise, if the expression is a polynomial in a handful of symbols,
//!    convert it to a [`MultiPoly`](crate::poly::multipoly::MultiPoly) and
//!    factor by Kronecker substitution or Hensel lifting
//!    ([`factor_multivariate_many`]).  As in SymPy's `factor`, a
//!    polynomial whose coefficients hold the imaginary unit is factored
//!    over `ℚ(i)`, and other constants (`√2`, `π`) are generators like the
//!    symbols (`√2·x + √2 = √2·(x + 1)`).
//! 3. Convert the factors back to expressions and build the product.
//!
//! [`factor_extension`] factors over an algebraic number field `ℚ(α)`
//! (SymPy's `factor(f, extension=…)`) by Trager's norm method
//! ([`numfield`]).
//!
//! Every factorization is verified by multiplying back before it is
//! returned; expressions that are not polynomial are returned unchanged.

use num_traits::One;

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::Q;
use crate::base::walk;
use crate::poly::Poly;
use crate::poly::factor_zassenhaus::{factor_multivariate, factor_multivariate_many};
use crate::poly::numfield::{self, AlgNum, ExtField, KPoly};
use crate::poly::polybridge;
use crate::poly::traits::Ring;

/// Number of symbols up to which the multivariate path converts with
/// `expr_to_multipoly` and factors by Kronecker substitution first (the
/// route of every release up to 0.42, kept as is).  Beyond it (up to 0.42
/// such polynomials were returned unchanged) the conversion has a size
/// budget and the factorization goes to Hensel lifting directly, within
/// the cost budget of
/// [`factor_multivariate_many`].
const MAX_MULTIVARIATE_SYMBOLS: usize = 4;

/// Content (a number: rational, or algebraic over an extension) plus
/// `(factor expression, multiplicity)` pairs.
type FactorParts = (ExprId, Vec<(ExprId, u32)>);

/// Which inputs the multivariate path accepts.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Route {
    /// The route of the internal callers (the solvers, `dsolve`), kept as
    /// it was up to 0.42 so that their outputs do not change: rational
    /// coefficients, at most [`MAX_MULTIVARIATE_SYMBOLS`] symbols.
    Internal,
    /// The public methods: also more symbols (Hensel lifting) and constant
    /// coefficients (`ℚ(i)`, constant generators; see
    /// [`factor_with_constants`]).
    Public,
}

/// Factor a polynomial expression in `var` into a product of irreducible
/// factors over ℤ (the internal route: rational coefficients, at most four
/// symbols; see [`factor_public`]).
///
/// Returns the expression unchanged if it is not polynomial in `var`, if it
/// is constant in `var`, or if it is already irreducible with unit content.
pub(crate) fn factor(arena: &mut Arena, expr: ExprId, var: ExprId) -> ExprId {
    match factor_parts(arena, expr, Some(var), Route::Internal) {
        Some((content, factors)) if is_nontrivial(arena, content, &factors) => {
            build_factored_expr(arena, content, &factors)
        }
        _ => expr,
    }
}

/// `Ex::factor`: [`factor`], and beyond it more than four symbols (Hensel
/// lifting) and constant coefficients as SymPy's `factor` takes them:
/// over `ℚ(i)` when `i` is the only constant, other constants as
/// generators (`√2·x² + √2·x = √2·x·(x + 1)`).  Up to 0.42 these were
/// returned unchanged.
pub(crate) fn factor_public(arena: &mut Arena, expr: ExprId, var: ExprId) -> ExprId {
    match factor_parts(arena, expr, Some(var), Route::Public) {
        Some((content, factors)) if is_nontrivial(arena, content, &factors) => {
            build_factored_expr(arena, content, &factors)
        }
        _ => expr,
    }
}

/// Factor a polynomial expression with the variable(s) inferred from its
/// free symbols.
///
/// With exactly one free symbol this is [`factor`] in that symbol; with
/// two or more symbols the multivariate path is used (beyond four symbols
/// by Hensel lifting within a cost budget; up to 0.42 such expressions
/// were returned unchanged).  Expressions with no free symbols, beyond the
/// budget, or with non-polynomial structure are returned unchanged.
pub(crate) fn factor_auto(arena: &mut Arena, expr: ExprId) -> ExprId {
    match factor_parts(arena, expr, None, Route::Public) {
        Some((content, factors)) if is_nontrivial(arena, content, &factors) => {
            build_factored_expr(arena, content, &factors)
        }
        _ => expr,
    }
}

/// Factor and return the pieces: `(content, [(factor, multiplicity), …])`
/// such that `expr = content · ∏ factorᵢ^multᵢ`.
///
/// If `var` is `None` the variables are inferred as in [`factor_auto`].
/// For inputs that cannot be factored (non-polynomial, constant, …) the
/// result is `(1, [(expr, 1)])`.  The internal route (see
/// [`factor_list_public`]).
pub(crate) fn factor_list(
    arena: &mut Arena,
    expr: ExprId,
    var: Option<ExprId>,
) -> (ExprId, Vec<(ExprId, u32)>) {
    match factor_parts(arena, expr, var, Route::Internal) {
        Some((content, factors)) => (content, factors),
        None => (arena.one, vec![(expr, 1)]),
    }
}

/// `Ex::factor_list` / `Ex::factor_list_all`: [`factor_list`] with the
/// inputs of [`factor_public`]; the content may be an algebraic number
/// (`2·i` for `2ix + 2`).
pub(crate) fn factor_list_public(
    arena: &mut Arena,
    expr: ExprId,
    var: Option<ExprId>,
) -> (ExprId, Vec<(ExprId, u32)>) {
    match factor_parts(arena, expr, var, Route::Public) {
        Some((content, factors)) => (content, factors),
        None => (arena.one, vec![(expr, 1)]),
    }
}

/// Shared driver: returns the content and the irreducible factors (as
/// expressions) or `None` if the expression is not a non-constant
/// polynomial in the chosen variable(s).
fn factor_parts(
    arena: &mut Arena,
    expr: ExprId,
    var: Option<ExprId>,
    route: Route,
) -> Option<FactorParts> {
    let mut syms = walk::free_symbols(arena, expr);
    syms.sort_by(|&a, &b| arena.canonical_cmp(a, b));
    syms.dedup();

    match var {
        Some(v) => {
            if let Some(poly) = polybridge::expr_to_poly(arena, expr, v) {
                return factor_univariate(arena, &poly, v);
            }
            if !syms.contains(&v) {
                return None;
            }
            match route {
                Route::Internal => factor_multi_internal(arena, expr, &syms),
                Route::Public => factor_with_constants(arena, expr, &syms, Some(v)),
            }
        }
        None => match syms.len() {
            0 => None,
            1 => {
                let v = syms[0];
                match (polybridge::expr_to_poly(arena, expr, v), route) {
                    (Some(poly), _) => factor_univariate(arena, &poly, v),
                    (None, Route::Internal) => None,
                    (None, Route::Public) => factor_with_constants(arena, expr, &syms, Some(v)),
                }
            }
            _ => match route {
                Route::Internal => factor_multi_internal(arena, expr, &syms),
                Route::Public => factor_with_constants(arena, expr, &syms, None),
            },
        },
    }
}

/// The multivariate path of the internal route (unchanged since 0.39).
fn factor_multi_internal(arena: &mut Arena, expr: ExprId, syms: &[ExprId]) -> Option<FactorParts> {
    if syms.len() > MAX_MULTIVARIATE_SYMBOLS {
        return None;
    }
    factor_multi(arena, expr, syms)
}

/// The multivariate path, with SymPy's default domain for coefficients
/// that are not rational: a univariate polynomial (in `var`) whose only
/// other generator is `i` is factored over `ℚ(i)` (SymPy's `ZZ_I`); other
/// constant generators (`√2`, `π`, `e`, and `i` beside them) become
/// variables of the multivariate factorization, as SymPy's generators.
/// Generators that hold a symbol (`sin x`, `√y`) are refused as before.
fn factor_with_constants(
    arena: &mut Arena,
    expr: ExprId,
    syms: &[ExprId],
    var: Option<ExprId>,
) -> Option<FactorParts> {
    let gens = numfield::generators(arena, &[expr]);
    let constants: Vec<ExprId> = gens
        .into_iter()
        .filter(|&g| !matches!(arena.node(g), ExprNode::Symbol(_)))
        .collect();
    if constants.is_empty() {
        return factor_multi(arena, expr, syms);
    }
    if walk::has_unevaluated(arena, expr)
        || constants.iter().any(|&g| {
            !walk::free_symbols(arena, g).is_empty()
                || matches!(
                    arena.node(g),
                    ExprNode::Infinity
                        | ExprNode::NegInfinity
                        | ExprNode::ComplexInfinity
                        | ExprNode::NaN
                )
        })
    {
        return None;
    }
    let i = arena.i_unit();
    if let (Some(v), [only]) = (var, constants.as_slice())
        && *only == i
        && syms.len() == 1
    {
        return factor_gaussian_default(arena, expr, v);
    }
    let mut vars: Vec<ExprId> = syms.to_vec();
    vars.extend(constants);
    factor_multi(arena, expr, &vars)
}

fn factor_univariate(arena: &mut Arena, poly: &Poly, var: ExprId) -> Option<FactorParts> {
    if poly.is_zero() || poly.is_constant() {
        return None;
    }
    let (content, factors) = poly.factor_over_z();
    let factor_ids = factors
        .iter()
        .map(|(f, m)| (polybridge::poly_to_expr(arena, f, var), *m))
        .collect();
    Some((arena.num_ratio(content), factor_ids))
}

fn factor_multi(arena: &mut Arena, expr: ExprId, syms: &[ExprId]) -> Option<FactorParts> {
    if syms.len() < 2 {
        return None;
    }
    let (content, factors) = if syms.len() <= MAX_MULTIVARIATE_SYMBOLS {
        let mp = polybridge::expr_to_multipoly(arena, expr, syms)?;
        if mp.is_zero() || mp.total_degree().unwrap_or(0) == 0 {
            return None;
        }
        factor_multivariate(&mp)?
    } else {
        let mp = crate::poly::paramfield::to_multipoly(arena, expr, syms)?;
        if mp.is_zero() || mp.total_degree().unwrap_or(0) == 0 {
            return None;
        }
        factor_multivariate_many(&mp)?
    };
    let factor_ids = factors
        .iter()
        .map(|(f, m)| (polybridge::multipoly_to_expr(arena, f, syms), *m))
        .collect();
    Some((arena.num_ratio(content), factor_ids))
}

/// SymPy's default for a polynomial with Gaussian coefficients: a factor
/// common to all terms is taken out first (SymPy's `factor` runs
/// `together`, whose `gcd_terms` extracts it), and when what remains has
/// rational coefficients it is factored over ℚ (`i·x² + 9i = i·(x² + 9)`);
/// otherwise over `ℚ(i)` (`x² + 2ix − 1 = (x + i)²`).
fn factor_gaussian_default(arena: &mut Arena, expr: ExprId, var: ExprId) -> Option<FactorParts> {
    let i = arena.i_unit();
    let mut ef = ExtField::for_atoms(arena, &[i])?;
    let f = ef.kpoly_of(arena, expr, var)?;
    let lc = f.leading_coeff()?.clone();
    let monic = f.make_monic();
    let rational: Option<Vec<Q>> = monic
        .coeffs()
        .iter()
        .map(|c| c.as_rational().cloned())
        .collect();
    if let Some(coeffs) = rational {
        let rational = Poly::from_coeffs(coeffs);
        let (content, factors) = rational.factor_over_z();
        let content = Ring::mul(&lc, &AlgNum::Rat(content));
        let factor_ids = factors
            .iter()
            .map(|(g, m)| (polybridge::poly_to_expr(arena, g, var), *m))
            .collect();
        return Some((ef.element_to_expr(arena, &content), factor_ids));
    }
    factor_over_extension(arena, expr, var, &[])
}

/// Factor `expr`, a polynomial in `var` with algebraic coefficients, over
/// the field generated by the algebraic numbers of its coefficients and of
/// `extension` (SymPy's `factor(f, extension=[…])`), by Trager's norm
/// method.  The factors are made monic and then scaled to coefficients in
/// `ℤ[α]` with no common integer factor (`x² − 1/2 = (2x − √2)(2x + √2)/4`,
/// SymPy's form).  `None` when the field cannot be built (an atom that is
/// not a recognised algebraic number, a degree above
/// [`MAX_FIELD_DEGREE`](numfield::MAX_FIELD_DEGREE)), the expression is not
/// such a polynomial, or a budget is exceeded.
fn factor_over_extension(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    extension: &[ExprId],
) -> Option<FactorParts> {
    // The algebraic numbers of the coefficients once multiplied out (as
    // SymPy builds its domain), and those of `extension`.
    let expanded = crate::transforms::expand::expand(arena, expr);
    let gens = numfield::generators(arena, &[expanded]);
    let mut atoms: Vec<ExprId> = Vec::new();
    for g in gens.into_iter().chain(extension.iter().copied()) {
        if g == var || atoms.contains(&g) || arena.as_num(g).is_some() {
            continue;
        }
        if walk::contains(arena, g, var) {
            return None;
        }
        atoms.push(g);
    }
    if atoms.is_empty() {
        let poly = polybridge::expr_to_poly(arena, expanded, var)?;
        return factor_univariate(arena, &poly, var);
    }
    let mut ef = ExtField::for_atoms(arena, &atoms)?;
    let f = ef.kpoly_of(arena, expr, var)?;
    let fac = ef.field.factor(&f)?;
    if !fac.complete {
        tracing::warn!("factor_extension: a norm factorization was not certified complete");
    }
    let mut content = fac.lc.clone();
    let mut factors: Vec<(ExprId, u32)> = Vec::with_capacity(fac.factors.len());
    for (h, m) in &fac.factors {
        let lq = ef.primitive_scale(h);
        let h2: KPoly = h.scale(&AlgNum::Rat(lq.clone()));
        let lm = Ring::pow_usize(&lq, *m as usize);
        content = Ring::mul(&content, &AlgNum::Rat(<Q as One>::one() / lm));
        factors.push((ef.kpoly_to_expr(arena, &h2, var), *m));
    }
    let content_id = ef.element_to_expr(arena, &content);
    Some((content_id, factors))
}

/// [`factor`] over the algebraic number field generated by `extension`
/// together with the algebraic numbers in the coefficients (SymPy's
/// `factor(f, x, extension=…)`; an empty `extension` is SymPy's
/// `extension=True`).  Returns `expr` unchanged when it is not such a
/// polynomial in `var`, when it is irreducible over that field, or beyond
/// the budgets of [`factor_over_extension`].
pub(crate) fn factor_extension(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    extension: &[ExprId],
) -> ExprId {
    match factor_over_extension(arena, expr, var, extension) {
        Some((content, factors)) if is_nontrivial(arena, content, &factors) => {
            build_factored_expr(arena, content, &factors)
        }
        _ => expr,
    }
}

/// A factorization is worth reporting if it has more than one factor, a
/// repeated factor, or a non-unit content.
fn is_nontrivial(arena: &Arena, content: ExprId, factors: &[(ExprId, u32)]) -> bool {
    factors.len() > 1
        || factors.iter().any(|(_, m)| *m > 1)
        || !arena.as_num(content).is_some_and(One::is_one)
}

/// Build `content · ∏ factor^mult` as an expression.
fn build_factored_expr(arena: &mut Arena, content: ExprId, factors: &[(ExprId, u32)]) -> ExprId {
    let mut parts: Vec<ExprId> = Vec::with_capacity(factors.len() + 1);

    if !arena.as_num(content).is_some_and(One::is_one) {
        parts.push(content);
    }

    for &(fexpr, mult) in factors {
        if mult == 1 {
            parts.push(fexpr);
        } else {
            let exp = arena.int(mult as i64);
            parts.push(arena.pow(fexpr, exp));
        }
    }

    match parts.len() {
        0 => arena.one,
        1 => parts[0],
        _ => arena.mul(&parts),
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

    fn powi(a: &mut Arena, base: ExprId, n: i64) -> ExprId {
        let e = a.int(n);
        a.pow(base, e)
    }

    #[test]
    fn factor_x_squared_minus_one() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x^2 - 1 = (x - 1)(x + 1)
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let one = a.one;
        let expr = a.sub(x2, one);
        let result = factor(&mut a, expr, x);
        let s = display(&a, result);
        // Should be a product involving (x - 1) and (x + 1) or equivalent
        // The canonical form might be (x + 1)*(-1 + x) or similar
        assert!(!s.contains("x^2"), "should be factored (no x^2): {s}");
    }

    #[test]
    fn factor_quadratic_two_roots() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x^2 - 5x + 6 = (x - 2)(x - 3)
        let two = a.int(2);
        let five = a.int(5);
        let six = a.int(6);
        let x2 = a.pow(x, two);
        let five_x = a.mul(&[five, x]);
        let minus_5x = a.neg(five_x);
        let expr = a.add(&[x2, minus_5x, six]);
        let result = factor(&mut a, expr, x);
        let s = display(&a, result);
        assert!(!s.contains("x^2"), "should be factored (no x^2): {s}");
    }

    #[test]
    fn factor_already_linear() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x + 1 — already linear, should stay unchanged
        let one = a.one;
        let expr = a.add(&[x, one]);
        let result = factor(&mut a, expr, x);
        assert_eq!(display(&a, result), "x + 1");
    }

    #[test]
    fn factor_no_rational_roots() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x^2 + 1 — no real rational roots
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let one = a.one;
        let expr = a.add(&[x2, one]);
        let result = factor(&mut a, expr, x);
        // Should remain unfactored
        assert_eq!(display(&a, result), "x^2 + 1");
    }

    #[test]
    fn factor_cubic_all_rational() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
        let three = a.int(3);
        let x3 = a.pow(x, three);
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let six_a = a.int(6);
        let six_x2 = a.mul(&[six_a, x2]);
        let eleven = a.int(11);
        let eleven_x = a.mul(&[eleven, x]);
        let six = a.int(6);
        let neg_six_x2 = a.neg(six_x2);
        let neg_six = a.neg(six);
        let expr = a.add(&[x3, neg_six_x2, eleven_x, neg_six]);
        let result = factor(&mut a, expr, x);
        let s = display(&a, result);
        assert!(
            !s.contains("x^3") && !s.contains("x^2"),
            "should be fully factored: {s}"
        );
    }

    #[test]
    fn factor_non_polynomial_unchanged() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.sin(x);
        let result = factor(&mut a, expr, x);
        assert_eq!(display(&a, result), "sin(x)");
    }

    #[test]
    fn factor_constant_unchanged() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let five = a.int(5);
        let result = factor(&mut a, five, x);
        assert_eq!(display(&a, result), "5");
    }

    #[test]
    fn factor_x12_minus_1_fully() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let x12 = powi(&mut a, x, 12);
        let one = a.one;
        let expr = a.sub(x12, one);
        let (content, factors) = factor_list(&mut a, expr, Some(x));
        assert_eq!(display(&a, content), "1");
        assert_eq!(factors.len(), 6, "τ(12) = 6 cyclotomic factors");
        let factored = factor(&mut a, expr, x);
        let s = display(&a, factored);
        assert!(s.contains("x^4 - x^2 + 1"), "should contain Φ12: {s}");
    }

    #[test]
    fn factor_multivariate_difference_of_squares() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let x2 = powi(&mut a, x, 2);
        let y2 = powi(&mut a, y, 2);
        let expr = a.sub(x2, y2);
        let result = factor(&mut a, expr, x);
        let s = display(&a, result);
        assert!(
            s.contains("x - y") && s.contains("x + y"),
            "x^2 - y^2 should factor: {s}"
        );
    }

    #[test]
    fn factor_auto_single_symbol() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let x2 = powi(&mut a, x, 2);
        let one = a.one;
        let expr = a.sub(x2, one);
        let result = factor_auto(&mut a, expr);
        assert!(!display(&a, result).contains("x^2"));
    }

    #[test]
    fn factor_auto_multivariate_with_content() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        // x^2 y - y = y (x - 1)(x + 1)
        let x2 = powi(&mut a, x, 2);
        let x2y = a.mul(&[x2, y]);
        let expr = a.sub(x2y, y);
        let (content, factors) = factor_list(&mut a, expr, None);
        assert_eq!(display(&a, content), "1");
        assert_eq!(factors.len(), 3);
        let names: Vec<String> = factors.iter().map(|(f, _)| display(&a, *f)).collect();
        assert!(names.iter().any(|n| n == "y"), "{names:?}");
    }

    #[test]
    fn factor_list_non_polynomial() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.sin(x);
        let (content, factors) = factor_list(&mut a, expr, Some(x));
        assert_eq!(content, a.one);
        assert_eq!(factors, vec![(expr, 1)]);
    }
}

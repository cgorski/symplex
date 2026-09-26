//! Symbolic equation solving.
//!
//! This module implements [`solve`] and [`solve_classified`], which find
//! the values of a variable that make an expression equal to zero.
//!
//! # Supported equation types
//!
//! - **Linear:** `a*x + b = 0` → `x = -b/a` (numeric or symbolic `a`, `b`)
//! - **Quadratic:** `a*x² + b*x + c = 0` → quadratic formula (numeric or
//!   symbolic coefficients)
//! - **Cubic / quartic:** Cardano and Ferrari on the irreducible factors
//! - **Binomial:** `a*xⁿ + b = 0` → all `n` complex roots via roots of unity
//! - **Higher degree:** exact factorisation over ℤ (Berlekamp–Zassenhaus);
//!   rational roots from the linear factors, radicals for the quadratic to
//!   quartic ones, `RootOf` placeholders only for irreducible factors of
//!   degree ≥ 5
//! - **Transcendental:** `exp`, `ln`, `sin`, `cos`, `tan`, hyperbolic and
//!   inverse functions, `|·|`, constant-base exponentials, all by inversion
//!   peeling (principal branches, or full periodic families in
//!   [`solve_general`])
//! - **Change of variable:** equations polynomial in `f(x)` for some `f`
//! - **Lambert W:** mixed polynomial–exponential forms
//!
//! # Design
//!
//! The solver works by:
//! 1. Classifying degenerate cases (identity `0 = 0`, contradiction `c = 0`).
//! 2. Converting the expression to a [`Poly`] in the given variable.
//! 3. Applying degree-specific solvers (linear, quadratic, …).
//! 4. Falling back to symbolic-coefficient and transcendental strategies.
//!
//! [`solve`] returns a bare `Vec<Solution>` for callers that only care
//! about explicit roots; [`solve_classified`] additionally distinguishes
//! *identity* (every value is a solution) and *no solution* outcomes.

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode, SymbolId};
use crate::base::numeric::Q;
use crate::poly::Poly;
use crate::poly::polybridge;

/// A solution to an equation, with the variable and its value.
#[derive(Clone, Debug)]
pub struct Solution {
    /// The value of the variable that satisfies the equation.
    pub value: ExprId,
}

/// Structured outcome of solving `expr = 0` for a variable.
#[derive(Clone, Debug)]
pub(crate) enum SolveOutcome {
    /// A (possibly empty) finite list of explicit solutions.
    ///
    /// An empty list means the solver could not find any root in the
    /// searched domain — it does **not** mean the equation is
    /// contradictory (see [`SolveOutcome::NoSolution`]).
    Solutions(Vec<Solution>),
    /// The equation reduces to `0 = 0`: every value of the variable is a
    /// solution.
    Identity,
    /// The equation is provably unsatisfiable (e.g. reduces to `1 = 0`, or
    /// `exp(x) = 0`).  Carries a human-readable reason.
    NoSolution(String),
}

impl SolveOutcome {
    /// Explicit solutions, or an empty list for identity / no-solution.
    pub(crate) fn into_solutions(self) -> Vec<Solution> {
        match self {
            SolveOutcome::Solutions(s) => s,
            SolveOutcome::Identity | SolveOutcome::NoSolution(_) => Vec::new(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Public entry points
// ═══════════════════════════════════════════════════════════════════════════

/// Solve `expr = 0` for `var`.
///
/// Returns a vector of solutions (values of `var` that make `expr`
/// zero).  Returns an empty vector if:
/// - The expression is not polynomial in `var` and no transcendental
///   strategy applies.
/// - The expression is identically zero (every value is a solution) or a
///   nonzero constant (no solution).  Use [`solve_classified`] to tell
///   these cases apart.
///
/// Solutions are returned as symbolic expressions ([`ExprId`]) in the
/// arena, fully canonicalized, each **distinct** root once whatever its
/// multiplicity (`(x − 1)² = 0` gives `[1]`).  Only principal branches of
/// periodic functions are returned; see [`solve_general`] for full
/// families.
pub(crate) fn solve(arena: &mut Arena, expr: ExprId, var: ExprId) -> Vec<Solution> {
    solve_classified(arena, expr, var).into_solutions()
}

/// Solve `expr = 0` for `var`, distinguishing identities and
/// contradictions from "no roots found".
pub(crate) fn solve_classified(arena: &mut Arena, expr: ExprId, var: ExprId) -> SolveOutcome {
    solve_impl(arena, expr, var, None)
}

/// Solve `expr = 0` for `var`, returning **general** solution families for
/// periodic functions.
///
/// `param` must be a fresh symbol (ideally carrying the *integer*
/// assumption); it is used as the free integer parameter `n` in
/// `asin(c) + 2πn`, `±acos(c) + 2πn`, `atan(c) + πn`, etc.  Equations
/// without periodic structure return the same results as [`solve`].
pub(crate) fn solve_general(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    param: ExprId,
) -> SolveOutcome {
    solve_impl(arena, expr, var, Some(param))
}

/// Shared implementation of [`solve_classified`] and [`solve_general`].
fn solve_impl(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    period: Option<ExprId>,
) -> SolveOutcome {
    match solve_raw(arena, expr, var, period) {
        SolveOutcome::Solutions(s) => {
            let had_candidates = !s.is_empty();
            let s = drop_certain_non_roots(arena, expr, var, s);
            if had_candidates && s.is_empty() {
                return SolveOutcome::NoSolution(
                    "every candidate solution fails the equation (a principal branch does not reach the right-hand side)"
                        .into(),
                );
            }
            finalize_solutions(arena, s)
        }
        other => other,
    }
}

/// Remove the candidates that certainly do not solve `expr = 0`.
///
/// Inversion peeling takes principal branches (`asin(f) = c → f = sin c`,
/// `ln f = c → f = e^c`, `f^(p/q) = c → f = u^q`), and a branch reproduces
/// `c` only when `c` lies in the range of the inverted function: `asin x = π`
/// gave `x = 0`, `√x = −2` gave `4`, `atan x = 2` gave `tan 2`,
/// `ln(atan x) = 1` gave `tan e`.  Rather than a range rule per function
/// (and per composition), every candidate of a non-polynomial equation is
/// substituted back, as SymPy's `solve` does with `checksol`: one at which
/// the equation evaluates to a certified non-zero number, or is undefined
/// (`|x|/x = 0` at `x = 0`), is dropped.  Candidates with free parameters,
/// and equations `evalf` cannot decide, are kept.  Polynomial equations
/// over ℚ are solved exactly and are not re-checked.
fn drop_certain_non_roots(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    solutions: Vec<Solution>,
) -> Vec<Solution> {
    if solutions.is_empty() || polybridge::expr_to_poly(arena, expr, var).is_some() {
        return solutions;
    }
    solutions
        .into_iter()
        .filter(|s| !certainly_not_a_root(arena, expr, var, s.value))
        .collect()
}

/// Tolerance of [`certainly_not_a_root`]: a residual certified beyond it
/// is not a rounding residue.
const ROOT_CHECK_TOLERANCE: f64 = 1e-10;

/// Does `expr` at `var = candidate` certainly not vanish?  `true` when the
/// substituted equation, free of symbols, is undefined (`zoo`, `NaN`, `±∞`)
/// or evaluates to a number of magnitude above [`ROOT_CHECK_TOLERANCE`];
/// `false` when it cannot be decided.
fn certainly_not_a_root(arena: &mut Arena, expr: ExprId, var: ExprId, candidate: ExprId) -> bool {
    use crate::base::walk;
    if !walk::free_symbols(arena, candidate).is_empty() || walk::has_unevaluated(arena, candidate) {
        return false;
    }
    let at = crate::transforms::subs::subs(arena, expr, var, candidate);
    let at = crate::transforms::eval::eval(arena, at);
    if !walk::free_symbols(arena, at).is_empty() || walk::has_unevaluated(arena, at) {
        return false;
    }
    let undefined = walk::post_order_ids(arena, at).into_iter().any(|id| {
        matches!(
            arena.node(id),
            ExprNode::Infinity | ExprNode::NegInfinity | ExprNode::ComplexInfinity | ExprNode::NaN
        )
    });
    if undefined {
        return true;
    }
    match crate::transforms::evalf::evalf_complex(arena, at, ROOT_CHECK_DIGITS) {
        Ok(z) => crate::transforms::evalf::abs_to_f64(&z).is_some_and(|m| m > ROOT_CHECK_TOLERANCE),
        Err(_) => false,
    }
}

/// Digits at which [`certainly_not_a_root`] evaluates a residual.
const ROOT_CHECK_DIGITS: u32 = 20;

/// Strategy dispatch without the final clean-up pass (see [`solve_impl`]).
fn solve_raw(arena: &mut Arena, expr: ExprId, var: ExprId, period: Option<ExprId>) -> SolveOutcome {
    // Step 0: degenerate cases.
    if arena.is_zero_structural(expr) {
        return SolveOutcome::Identity;
    }
    if !expr_contains_var(arena, expr, var) {
        return classify_constant(arena, expr, var);
    }

    // Pre-check: if expr is a Mul, solve each var-dependent factor
    // independently.  x*(x-1)*(x+2) = 0 → union of factor solutions.
    if let ExprNode::Mul(ref children) = arena.node(expr).clone() {
        let mut solutions: Vec<Solution> = Vec::new();
        let mut any_identity = false;
        for &child in children {
            if !expr_contains_var(arena, child, var) {
                // Constant factor: the canonicalizer already folds literal
                // zeros, so a surviving constant factor is treated as nonzero.
                continue;
            }
            match solve_raw(arena, child, var, period) {
                SolveOutcome::Solutions(child_solutions) => {
                    for sol in child_solutions {
                        if !solutions.iter().any(|s| s.value == sol.value) {
                            solutions.push(sol);
                        }
                    }
                }
                SolveOutcome::Identity => any_identity = true,
                SolveOutcome::NoSolution(_) => {}
            }
        }
        if any_identity {
            return SolveOutcome::Identity;
        }
        if !solutions.is_empty() {
            return SolveOutcome::Solutions(solutions);
        }
        // Otherwise fall through and treat the product as a whole.
    }

    // Step 1: Convert to polynomial with rational coefficients.
    if let Some(poly) = polybridge::expr_to_poly(arena, expr, var) {
        if poly.is_zero() {
            return SolveOutcome::Identity;
        }
        if poly.is_constant() {
            let c = arena.display(expr).to_string();
            return SolveOutcome::NoSolution(format!("equation reduces to {c} = 0"));
        }
        return SolveOutcome::Solutions(solve_rational_poly(arena, var, &poly));
    }

    // Step 2: Not polynomial over ℚ.  Try symbolic linear solver first:
    // handles a*x + b = 0 where a, b are symbolic expressions.
    if let Some(solutions) = try_solve_linear_symbolic(arena, expr, var)
        && !solutions.is_empty()
    {
        return SolveOutcome::Solutions(solutions);
    }

    // Step 2b: polynomial in `var` with symbolic coefficients (degree ≤ 2,
    // or binomial a·xⁿ + b).
    if let Some(solutions) = try_solve_symbolic_poly(arena, expr, var)
        && !solutions.is_empty()
    {
        return SolveOutcome::Solutions(solutions);
    }

    // Step 3: transcendental solving via inversion peeling.
    // Handles: exp(x)=c, ln(x)=c, sin(x)=c, sqrt(x)=c, etc.
    let mut domain_empty = false;
    match try_solve_by_inversion(arena, expr, var, period) {
        Some(solutions) if !solutions.is_empty() => {
            return SolveOutcome::Solutions(solutions);
        }
        Some(_) => domain_empty = true, // peeling proved "no real solution"
        None => {}
    }

    // Step 4: change-of-variable: if expression is polynomial in f(x) for
    // some f, substitute t = f(x), solve the polynomial, back-substitute.
    if let Some(solutions) = try_change_of_variable(arena, expr, var, period)
        && !solutions.is_empty()
    {
        return SolveOutcome::Solutions(solutions);
    }

    // Step 4b: radicals of the variable itself (`√x = x − 2`).
    if let Some(solutions) = try_radical_substitution(arena, expr, var)
        && !solutions.is_empty()
    {
        return SolveOutcome::Solutions(solutions);
    }

    // Step 5: LambertW for mixed polynomial-exponential equations:
    // x·exp(x) = c, x·exp(a·x) = c, exp(x) + x = c, etc.
    let var_sym_opt = match arena.node(var) {
        ExprNode::Symbol(sid) => Some(*sid),
        _ => None,
    };
    if let Some(var_sym) = var_sym_opt
        && let Some(solutions) = try_solve_lambert(arena, expr, var, var_sym)
        && !solutions.is_empty()
    {
        return SolveOutcome::Solutions(solutions);
    }

    if domain_empty {
        return SolveOutcome::NoSolution(
            "equation has no real solutions (range restriction of exp/sin/cos/cosh/abs)".into(),
        );
    }
    SolveOutcome::Solutions(Vec::new())
}

/// Post-process a list of candidate solutions: evaluate exact special
/// values (`asin(1/2)` → `π/6`), drop infinite / undefined candidates
/// (e.g. `1/x = 0` → `zoo`), and deduplicate.
fn finalize_solutions(arena: &mut Arena, solutions: Vec<Solution>) -> SolveOutcome {
    let had_candidates = !solutions.is_empty();
    let mut out: Vec<Solution> = Vec::with_capacity(solutions.len());
    for s in solutions {
        let v = crate::transforms::eval::eval(arena, s.value);
        let infinite = matches!(
            arena.node(v),
            ExprNode::Infinity | ExprNode::NegInfinity | ExprNode::ComplexInfinity | ExprNode::NaN
        );
        if infinite {
            continue;
        }
        if !out.iter().any(|o| o.value == v) {
            out.push(Solution { value: v });
        }
    }
    if had_candidates && out.is_empty() {
        return SolveOutcome::NoSolution(
            "every candidate solution is infinite or undefined (e.g. 1/x = 0)".into(),
        );
    }
    SolveOutcome::Solutions(out)
}

/// Dispatch a nonconstant polynomial with rational coefficients to the
/// degree-specific solvers.
fn solve_rational_poly(arena: &mut Arena, var: ExprId, poly: &Poly) -> Vec<Solution> {
    let degree = poly.degree().unwrap_or(0);
    tracing::debug!(degree = degree, "polynomial degree determined");
    match degree {
        0 => Vec::new(),
        1 => solve_linear(arena, poly),
        2 => solve_quadratic(arena, poly),
        3 => solve_cubic(arena, var, poly),
        4 => solve_quartic(arena, var, poly),
        _ => solve_rational_roots(arena, var, poly),
    }
}

/// Classify an expression that does **not** depend on the solve variable.
///
/// Decides whether `expr = 0` is an identity, a contradiction, or
/// undecidable (symbolic constant).  Symbolic constants that are not
/// provably zero are reported as `NoSolution` with an explanatory reason
/// — the equation holds only for special values of the parameters.
fn classify_constant(arena: &mut Arena, expr: ExprId, var: ExprId) -> SolveOutcome {
    let var_name = arena.display(var).to_string();
    let evaled = crate::transforms::eval::eval(arena, expr);
    if arena.is_zero_structural(evaled) {
        return SolveOutcome::Identity;
    }
    if let Some(r) = arena.as_num(evaled)
        && !r.is_zero()
    {
        return SolveOutcome::NoSolution(format!("equation reduces to {r} = 0"));
    }
    // Try harder: expand + eval.
    let expanded = crate::transforms::expand::expand(arena, evaled);
    let expanded = crate::transforms::eval::eval(arena, expanded);
    if arena.is_zero_structural(expanded) {
        return SolveOutcome::Identity;
    }
    if let Some(r) = arena.as_num(expanded)
        && !r.is_zero()
    {
        return SolveOutcome::NoSolution(format!("equation reduces to {r} = 0"));
    }
    // Purely numeric constant (no free symbols): decide numerically.
    if crate::base::walk::free_symbols(arena, expanded).is_empty()
        && let Ok(s) = crate::transforms::evalf::evalf(arena, expanded, 20)
    {
        let mag = parse_evalf_magnitude(&s);
        if let Some(m) = mag {
            if m > 1e-9 {
                let shown = arena.display(expr).to_string();
                return SolveOutcome::NoSolution(format!(
                    "equation reduces to the nonzero constant {shown} = 0"
                ));
            }
            if m < 1e-15 {
                return SolveOutcome::Identity;
            }
        }
    }
    let shown = arena.display(expr).to_string();
    SolveOutcome::NoSolution(format!(
        "expression {shown} does not depend on {var_name} and is not identically zero"
    ))
}

/// Parse the magnitude of an `evalf` output string (real or `a + b*I`).
pub(crate) fn parse_evalf_magnitude(s: &str) -> Option<f64> {
    let s = s.trim();
    if let Ok(v) = s.parse::<f64>() {
        return Some(v.abs());
    }
    // Complex: split on the last '+' or '-' that separates the parts.
    let body = s.replace('*', "");
    let body = body.trim_end_matches(['I', 'i']).trim();
    // Pure imaginary: "i", "-i", "2.5i".
    match body {
        "" | "+" => return Some(1.0),
        "-" => return Some(1.0),
        _ => {}
    }
    if let Ok(v) = body.parse::<f64>() {
        return Some(v.abs());
    }
    let mut split_at = None;
    for (i, ch) in body.char_indices().skip(1) {
        if (ch == '+' || ch == '-') && !body[..i].ends_with('e') && !body[..i].ends_with('E') {
            split_at = Some(i);
        }
    }
    let idx = split_at?;
    let re: f64 = body[..idx].trim().parse().ok()?;
    // The imaginary part is printed as "- 1.5e-3" (space after the sign).
    let im_text: String = body[idx..].chars().filter(|c| !c.is_whitespace()).collect();
    let im: f64 = match im_text.as_str() {
        "+" => 1.0,
        "-" => -1.0,
        t => t.parse().ok()?,
    };
    Some(re.hypot(im))
}

// ═══════════════════════════════════════════════════════════════════════════
// Local helper: check if an expression contains a given variable
// ═══════════════════════════════════════════════════════════════════════════

fn expr_contains_var(arena: &Arena, expr: ExprId, var: ExprId) -> bool {
    crate::base::walk::contains(arena, expr, var)
}

// ═══════════════════════════════════════════════════════════════════════════
// Symbolic-coefficient polynomial solving
// ═══════════════════════════════════════════════════════════════════════════

/// Extract ascending coefficients of `expr` viewed as a polynomial in
/// `var`, allowing arbitrary var-free symbolic coefficients.
///
/// Returns `None` if `var` appears in a non-polynomial position.
pub(crate) fn symbolic_poly_coeffs(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
) -> Option<Vec<ExprId>> {
    let expanded = crate::transforms::expand::expand(arena, expr);
    let terms: Vec<ExprId> = match arena.node(expanded).clone() {
        ExprNode::Add(children) => children.to_vec(),
        _ => vec![expanded],
    };
    let mut buckets: Vec<Vec<ExprId>> = Vec::new();
    for term in terms {
        let (deg, coeff) = term_degree_coeff(arena, term, var)?;
        if buckets.len() <= deg {
            buckets.resize_with(deg + 1, Vec::new);
        }
        buckets[deg].push(coeff);
    }
    let mut coeffs = Vec::with_capacity(buckets.len());
    for bucket in buckets {
        let c = match bucket.len() {
            0 => arena.zero,
            1 => bucket[0],
            _ => arena.add(&bucket),
        };
        coeffs.push(crate::transforms::eval::eval(arena, c));
    }
    // Trim leading zeros.
    while coeffs.len() > 1 && arena.is_zero_structural(*coeffs.last()?) {
        coeffs.pop();
    }
    Some(coeffs)
}

/// Split a single product term into `(degree in var, var-free coefficient)`.
fn term_degree_coeff(arena: &mut Arena, term: ExprId, var: ExprId) -> Option<(usize, ExprId)> {
    if term == var {
        return Some((1, arena.one));
    }
    if !expr_contains_var(arena, term, var) {
        return Some((0, term));
    }
    match arena.node(term).clone() {
        ExprNode::Pow(base, exp) if base == var => {
            let n = arena.as_num(exp)?.clone();
            if !n.is_integer() || n.is_negative() {
                return None;
            }
            let d: usize = n.to_integer().try_into().ok()?;
            Some((d, arena.one))
        }
        ExprNode::Neg(inner) => {
            let (d, c) = term_degree_coeff(arena, inner, var)?;
            Some((d, arena.neg(c)))
        }
        ExprNode::Mul(children) => {
            let mut deg = 0usize;
            let mut consts: Vec<ExprId> = Vec::new();
            for &child in &children {
                if !expr_contains_var(arena, child, var) {
                    consts.push(child);
                } else if child == var {
                    deg += 1;
                } else if let ExprNode::Pow(base, exp) = arena.node(child).clone()
                    && base == var
                {
                    let n = arena.as_num(exp)?.clone();
                    if !n.is_integer() || n.is_negative() {
                        return None;
                    }
                    let d: usize = n.to_integer().try_into().ok()?;
                    deg += d;
                } else {
                    return None;
                }
            }
            let c = match consts.len() {
                0 => arena.one,
                1 => consts[0],
                _ => arena.mul(&consts),
            };
            Some((deg, c))
        }
        _ => None,
    }
}

/// Solve a polynomial in `var` whose coefficients are symbolic but free
/// of `var`.  Handles degree 1, degree 2 (quadratic formula) and the
/// binomial form `a·xⁿ + b` (roots of unity).
fn try_solve_symbolic_poly(arena: &mut Arena, expr: ExprId, var: ExprId) -> Option<Vec<Solution>> {
    let coeffs = symbolic_poly_coeffs(arena, expr, var)?;
    solve_symbolic_coeffs(arena, &coeffs)
}

/// Solve from ascending symbolic coefficients (see [`try_solve_symbolic_poly`]).
///
/// The linear root `-b/a` and the quadratic discriminant are put into
/// rational normal form with `ratsimp`, so parametric coefficients that are
/// themselves fractions (`(3r - 1)/(j + 1) - (r + 1)/(2j)`) yield a single
/// cancelled fraction rather than a fraction of fractions.
fn solve_symbolic_coeffs(arena: &mut Arena, coeffs: &[ExprId]) -> Option<Vec<Solution>> {
    let degree = coeffs.len().checked_sub(1)?;
    match degree {
        0 => None,
        1 => {
            let a = coeffs[1];
            let b = coeffs[0];
            let neg_b = arena.neg(b);
            let v = arena.div(neg_b, a);
            let v = crate::transforms::eval::eval(arena, v);
            let v = crate::simplify::ratsimp::ratsimp(arena, v);
            Some(vec![Solution { value: v }])
        }
        2 => {
            let a = coeffs[2];
            let b = coeffs[1];
            let c = coeffs[0];
            if arena.is_zero_structural(b) {
                // a·x² + c = 0 → x = ±sqrt(-c/a)
                let neg_c = arena.neg(c);
                let ratio = arena.div(neg_c, a);
                let ratio = crate::transforms::eval::eval(arena, ratio);
                let root = arena.sqrt(ratio);
                let root = crate::transforms::eval::eval(arena, root);
                let neg_root = arena.neg(root);
                return Some(vec![Solution { value: root }, Solution { value: neg_root }]);
            }
            // x = (-b ± sqrt(b² - 4ac)) / (2a)
            let two = arena.int(2);
            let four = arena.int(4);
            let b_sq = arena.pow(b, two);
            let four_ac = arena.mul(&[four, a, c]);
            let disc = arena.sub(b_sq, four_ac);
            let disc = crate::transforms::eval::eval(arena, disc);
            let disc = crate::simplify::ratsimp::ratsimp(arena, disc);
            let sqrt_disc = arena.sqrt(disc);
            let neg_b = arena.neg(b);
            let two_a = arena.mul(&[two, a]);
            let num1 = arena.add(&[neg_b, sqrt_disc]);
            let num2 = arena.sub(neg_b, sqrt_disc);
            let x1 = arena.div(num1, two_a);
            let x2 = arena.div(num2, two_a);
            let x1 = crate::transforms::eval::eval(arena, x1);
            let x2 = crate::transforms::eval::eval(arena, x2);
            if x1 == x2 {
                Some(vec![Solution { value: x1 }])
            } else {
                Some(vec![Solution { value: x1 }, Solution { value: x2 }])
            }
        }
        _ => {
            // Binomial a·xⁿ + b = 0.
            let middle_zero = coeffs[1..degree]
                .iter()
                .all(|&c| arena.is_zero_structural(c));
            if !middle_zero {
                return None;
            }
            let a = coeffs[degree];
            let b = coeffs[0];
            let neg_b = arena.neg(b);
            let ratio = arena.div(neg_b, a);
            let ratio = crate::transforms::eval::eval(arena, ratio);
            Some(binomial_roots(arena, ratio, degree))
        }
    }
}

/// All `n` complex solutions of `xⁿ = c`: `c^(1/n) · e^{2πik/n}` for
/// `k = 0..n`, with the roots of unity written as `cos + i·sin`.
fn binomial_roots(arena: &mut Arena, c: ExprId, n: usize) -> Vec<Solution> {
    if arena.is_zero_structural(c) {
        return vec![Solution { value: arena.zero }];
    }
    // For a negative real constant use |c|^(1/n)·e^{iπ(2k+1)/n} so that the
    // roots come out as explicit real/imaginary combinations instead of
    // an unevaluated principal root of a negative number.
    let negative_real = arena.as_num(c).is_some_and(|r| r.is_negative());
    let (radicand, angle_offset) = if negative_real {
        (arena.neg(c), 1i64)
    } else {
        (c, 0i64)
    };
    let inv_n = arena.rational(1, n as i64);
    let magnitude = arena.pow(radicand, inv_n);
    let magnitude = crate::transforms::eval::eval(arena, magnitude);
    let pi = arena.pi;
    let i_unit = arena.i_unit;
    let mut roots: Vec<Solution> = Vec::with_capacity(n);
    for k in 0..n {
        // angle = π·(2k + offset)/n
        let numer = 2 * k as i64 + angle_offset;
        let root = if numer == 0 {
            magnitude
        } else {
            let frac = arena.rational(numer, n as i64);
            let angle = arena.mul(&[frac, pi]);
            let cos_a = arena.cos(angle);
            let sin_a = arena.sin(angle);
            let i_sin = arena.mul(&[i_unit, sin_a]);
            let omega = arena.add(&[cos_a, i_sin]);
            let prod = arena.mul(&[magnitude, omega]);
            let prod = crate::transforms::eval::eval(arena, prod);
            let prod = crate::transforms::expand::expand(arena, prod);
            crate::transforms::eval::eval(arena, prod)
        };
        if !roots.iter().any(|r| r.value == root) {
            roots.push(Solution { value: root });
        }
    }
    roots
}

/// Binomial shortcut for rational polynomials `a·xⁿ + b` (n ≥ 3): returns
/// all `n` roots via roots of unity.  `None` if the polynomial has any
/// middle terms.
fn try_solve_binomial_rational(arena: &mut Arena, poly: &Poly) -> Option<Vec<Solution>> {
    let n = poly.degree()?;
    if n < 3 {
        return None;
    }
    for i in 1..n {
        if !poly.coeff(i).is_zero() {
            return None;
        }
    }
    let a = poly.coeff(n);
    let b = poly.coeff(0);
    if a.is_zero() {
        return None;
    }
    let ratio = -b / a;
    let c = arena.num_ratio(ratio.clone());
    Some(binomial_roots(arena, c, n))
}

// ═══════════════════════════════════════════════════════════════════════════
// Transcendental solving via inversion peeling
// ═══════════════════════════════════════════════════════════════════════════

/// Try to solve `expr = 0` by algebraic inversion.
/// Restructures as `f(x) = c` and inverts `f`.
///
/// Returns `Some(vec![])` when a range restriction proves there is no
/// real solution (e.g. `exp(x) = -1`), and `None` when the structure is
/// not invertible.
fn try_solve_by_inversion(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    period: Option<ExprId>,
) -> Option<Vec<Solution>> {
    if !expr_contains_var(arena, expr, var) {
        return None;
    }
    let zero = arena.zero;
    solve_by_peeling(arena, expr, zero, var, period)
}

/// Peel layers off `lhs = rhs` to isolate `var`.
///
/// `rhs` never contains `var`.  When `period` is `Some(n)`, periodic
/// inversions (sin, cos, tan) emit full solution families in terms of the
/// integer parameter `n`; otherwise only principal branches are produced.
fn solve_by_peeling(
    arena: &mut Arena,
    lhs: ExprId,
    rhs: ExprId,
    var: ExprId,
    period: Option<ExprId>,
) -> Option<Vec<Solution>> {
    // Base case: lhs IS the variable
    if lhs == var {
        return Some(vec![Solution { value: rhs }]);
    }

    let node = arena.node(lhs).clone();
    match node {
        // f(x) + c = rhs → f(x) = rhs - c ; several var-terms → polynomial fallback
        ExprNode::Add(ref children) => {
            let mut dep = Vec::new();
            let mut indep = Vec::new();
            for &child in children {
                if expr_contains_var(arena, child, var) {
                    dep.push(child);
                } else {
                    indep.push(child);
                }
            }
            if dep.len() == 1 {
                let new_rhs = if indep.is_empty() {
                    rhs
                } else {
                    let sum_indep = arena.add(&indep);
                    arena.sub(rhs, sum_indep)
                };
                return solve_by_peeling(arena, dep[0], new_rhs, var, period);
            }
            // Multiple var-dependent terms: lhs - rhs = 0 may be polynomial.
            let diff = arena.sub(lhs, rhs);
            let diff = crate::transforms::eval::eval(arena, diff);
            if let Some(poly) = polybridge::expr_to_poly(arena, diff, var) {
                if poly.is_zero() || poly.is_constant() {
                    return None;
                }
                return Some(solve_rational_poly(arena, var, &poly));
            }
            if let Some(sols) = try_solve_linear_symbolic(arena, diff, var) {
                return Some(sols);
            }
            try_solve_symbolic_poly(arena, diff, var)
        }
        // c * f(x) = rhs → f(x) = rhs/c
        ExprNode::Mul(ref children) => {
            let mut dep = Vec::new();
            let mut indep = Vec::new();
            for &child in children {
                if expr_contains_var(arena, child, var) {
                    dep.push(child);
                } else {
                    indep.push(child);
                }
            }
            if dep.len() != 1 || indep.is_empty() {
                return None;
            }
            let coeff = arena.mul(&indep);
            let new_rhs = arena.div(rhs, coeff);
            solve_by_peeling(arena, dep[0], new_rhs, var, period)
        }
        // exp(f(x)) = rhs → f(x) = ln(rhs)
        // Domain check: exp(x) > 0 for all real x, so rhs must be strictly positive.
        ExprNode::Exp(inner) => {
            if rhs == arena.zero {
                tracing::debug!("solve_by_peeling: exp domain error, rhs = 0");
                return Some(vec![]);
            }
            if let Some(c) = arena.as_num(rhs)
                && !c.is_positive()
            {
                tracing::debug!("solve_by_peeling: exp domain error, rhs <= 0");
                return Some(vec![]);
            }
            let new_rhs = arena.ln(rhs);
            solve_by_peeling(arena, inner, new_rhs, var, period)
        }
        // ln(f(x)) = rhs → f(x) = exp(rhs)
        ExprNode::Ln(inner) => {
            let new_rhs = arena.exp(rhs);
            solve_by_peeling(arena, inner, new_rhs, var, period)
        }
        // sin(f(x)) = rhs → f(x) ∈ {asin(rhs), π - asin(rhs)} (+ 2πn)
        ExprNode::Sin(inner) => {
            // Domain check: sin(x) = c has no real solutions when |c| > 1
            if let Some(c) = arena.as_num(rhs)
                && c.abs() > Ratio::one()
            {
                tracing::debug!("solve_by_peeling: sin domain error, |c| > 1");
                return Some(vec![]);
            }
            tracing::debug!("solve_by_peeling: inverting sin, two branches");
            let asin_rhs = arena.asin(rhs);
            let pi = arena.pi;
            let pi_minus_asin = arena.sub(pi, asin_rhs);
            let (b1, b2) = match period {
                Some(n) => {
                    let two = arena.int(2);
                    let two_pi_n = arena.mul(&[two, pi, n]);
                    (
                        arena.add(&[asin_rhs, two_pi_n]),
                        arena.add(&[pi_minus_asin, two_pi_n]),
                    )
                }
                None => (asin_rhs, pi_minus_asin),
            };
            peel_two_branches(arena, inner, b1, b2, var, period)
        }
        // cos(f(x)) = rhs → f(x) ∈ {acos(rhs), -acos(rhs)} (+ 2πn)
        ExprNode::Cos(inner) => {
            // Domain check: cos(x) = c has no real solutions when |c| > 1
            if let Some(c) = arena.as_num(rhs)
                && c.abs() > Ratio::one()
            {
                tracing::debug!("solve_by_peeling: cos domain error, |c| > 1");
                return Some(vec![]);
            }
            tracing::debug!("solve_by_peeling: inverting cos, two branches");
            let acos_rhs = arena.acos(rhs);
            let neg_acos = arena.neg(acos_rhs);
            let (b1, b2) = match period {
                Some(n) => {
                    let two = arena.int(2);
                    let pi = arena.pi;
                    let two_pi_n = arena.mul(&[two, pi, n]);
                    (
                        arena.add(&[acos_rhs, two_pi_n]),
                        arena.add(&[neg_acos, two_pi_n]),
                    )
                }
                None => (acos_rhs, neg_acos),
            };
            peel_two_branches(arena, inner, b1, b2, var, period)
        }
        // tan(f(x)) = rhs → f(x) = atan(rhs) (+ πn)
        ExprNode::Tan(inner) => {
            let atan_rhs = arena.atan(rhs);
            let new_rhs = match period {
                Some(n) => {
                    let pi = arena.pi;
                    let pi_n = arena.mul(&[pi, n]);
                    arena.add(&[atan_rhs, pi_n])
                }
                None => atan_rhs,
            };
            solve_by_peeling(arena, inner, new_rhs, var, period)
        }
        // f(x)^n = rhs → f(x) = rhs^(1/n)
        // When n is a positive even integer, also consider f(x) = -(rhs^(1/n))
        //
        // a^f(x) = rhs → f(x) = ln(rhs) / ln(a)  (constant base, variable exponent)
        ExprNode::Pow(inner_base, inner_exp) => {
            if let Some(n) = arena.as_num(inner_exp) {
                let n = n.clone();
                if !n.is_zero() {
                    let branches = power_preimages(arena, rhs, &n)?;
                    return peel_branches(arena, inner_base, &branches, var, period);
                }
            }

            // a^f(x) = rhs where a is a constant (no var) and f(x) contains var.
            // Strategy: f(x) = ln(rhs) / ln(a).
            //
            // Integer shortcut: if a and rhs are positive integers and a^k == rhs
            // for some k, solve f(x) = k directly (gives exact answer like x = 3
            // instead of x = ln(8)/ln(2)).
            if !expr_contains_var(arena, inner_base, var)
                && expr_contains_var(arena, inner_exp, var)
            {
                tracing::debug!("solve_by_peeling: constant-base exponential a^f(x) = rhs");

                // Integer shortcut: try to find k such that base^k == rhs
                if let (Some(b), Some(r)) = (
                    arena.as_num(inner_base).cloned(),
                    arena.as_num(rhs).cloned(),
                ) && b.is_integer()
                    && r.is_integer()
                    && b > Ratio::one()
                    && r.is_positive()
                {
                    let b_int = b.to_integer();
                    let r_int = r.to_integer();
                    // Try small powers: b^1, b^2, ... up to b^64
                    let mut power = BigInt::one();
                    for k in 0u32..65 {
                        if power == r_int {
                            let k_expr = arena.int(k as i64);
                            tracing::debug!(
                                "solve_by_peeling: integer log shortcut, base^{k} = rhs"
                            );
                            return solve_by_peeling(arena, inner_exp, k_expr, var, period);
                        }
                        if power > r_int {
                            break;
                        }
                        power *= &b_int;
                    }
                }

                // General case: f(x) = ln(rhs) / ln(base)
                let ln_base = arena.ln(inner_base);
                let ln_rhs = arena.ln(rhs);
                let new_rhs = arena.div(ln_rhs, ln_base);
                return solve_by_peeling(arena, inner_exp, new_rhs, var, period);
            }

            None
        }
        // Neg(-f(x)) = rhs → f(x) = -rhs
        ExprNode::Neg(inner) => {
            let new_rhs = arena.neg(rhs);
            solve_by_peeling(arena, inner, new_rhs, var, period)
        }
        // ── Inverse trig peeling ──────────────────────────────────
        // asin(f(x)) = rhs → f(x) = sin(rhs)
        ExprNode::Asin(inner) => {
            let new_rhs = arena.sin(rhs);
            solve_by_peeling(arena, inner, new_rhs, var, period)
        }
        // acos(f(x)) = rhs → f(x) = cos(rhs)
        ExprNode::Acos(inner) => {
            let new_rhs = arena.cos(rhs);
            solve_by_peeling(arena, inner, new_rhs, var, period)
        }
        // atan(f(x)) = rhs → f(x) = tan(rhs)
        ExprNode::Atan(inner) => {
            let new_rhs = arena.tan(rhs);
            solve_by_peeling(arena, inner, new_rhs, var, period)
        }
        // ── Inverse hyperbolic peeling ────────────────────────────
        // sinh(f(x)) = rhs → f(x) = asinh(rhs)
        ExprNode::Sinh(inner) => {
            let new_rhs = arena.asinh(rhs);
            solve_by_peeling(arena, inner, new_rhs, var, period)
        }
        // cosh(f(x)) = rhs → f(x) = ±acosh(rhs)
        // Domain check: cosh(x) >= 1 for all real x, so rhs must be >= 1.
        ExprNode::Cosh(inner) => {
            if let Some(c) = arena.as_num(rhs)
                && *c < Ratio::one()
            {
                tracing::debug!("solve_by_peeling: cosh domain error, rhs < 1");
                return Some(vec![]);
            }
            let acosh_rhs = arena.acosh(rhs);
            let neg_acosh = arena.neg(acosh_rhs);
            peel_two_branches(arena, inner, acosh_rhs, neg_acosh, var, period)
        }
        // tanh(f(x)) = rhs → f(x) = atanh(rhs)
        ExprNode::Tanh(inner) => {
            let new_rhs = arena.atanh(rhs);
            solve_by_peeling(arena, inner, new_rhs, var, period)
        }
        // asinh(f) = rhs → f = sinh(rhs), acosh(f) = rhs → f = cosh(rhs),
        // atanh(f) = rhs → f = tanh(rhs): valid when rhs lies in the range
        // of the principal branch, which the final check of every candidate
        // decides (`acosh x = −1` has no solution: `cosh(−1)` gives
        // `acosh(cosh 1) = 1`).  Without these the kinks of `|asinh x|`
        // were invisible to the definite integrator.
        ExprNode::Asinh(inner) => {
            let new_rhs = arena.sinh(rhs);
            solve_by_peeling(arena, inner, new_rhs, var, period)
        }
        ExprNode::Acosh(inner) => {
            let new_rhs = arena.cosh(rhs);
            solve_by_peeling(arena, inner, new_rhs, var, period)
        }
        ExprNode::Atanh(inner) => {
            let new_rhs = arena.tanh(rhs);
            solve_by_peeling(arena, inner, new_rhs, var, period)
        }
        // ── Abs peeling ───────────────────────────────────────────
        // |f(x)| = rhs → f(x) = rhs OR f(x) = -rhs (when rhs ≥ 0)
        ExprNode::Abs(inner) => {
            // |f(x)| = negative has no solutions
            if let Some(r) = arena.as_num(rhs)
                && r.is_negative()
            {
                return Some(vec![]);
            }
            let neg_rhs = arena.neg(rhs);
            peel_two_branches(arena, inner, rhs, neg_rhs, var, period)
        }
        _ => None,
    }
}

/// Largest `|p|` for which `f^(p/q) = c` with a numeric `c` is inverted
/// through all `|p|` roots of `u^p = c` (beyond it, the principal
/// candidate only).
const MAX_POWER_PREIMAGES: i64 = 24;

/// The values `f` with `f^n = c` (`n ≠ 0` rational) that peeling continues
/// with; `Some(vec![])` when there is none (`f^(−k) = 0`).
///
/// * Integer `n`: every root of `f^|n| = c^(sign n)` (`binomial_roots`), as
///   the polynomial route returns all roots of `xⁿ = c`.  Before 0.30 only
///   `c^(1/n)` (and its negative for a positive even `n`) came back:
///   `x^(−2) = 3` lost `−1/√3`, `(x²)^(−3) = 1/2` lost `−2^(1/6)`.
/// * `n = p/q` in lowest terms, `q > 1`: `f^(p/q) = (f^(1/q))^p` with the
///   principal `u = f^(1/q)` in the sector `−π/q < arg u ≤ π/q`, so the
///   candidates are `u^q` for the roots `u` of `u^p = c`; those outside the
///   sector are spurious (`√x = −2` has none, `x^(1/3) = −3/4` neither)
///   and are dropped by the check of every candidate against the equation
///   ([`certainly_not_a_root`]).  For a `c` with free symbols that check
///   cannot run, and only the principal candidate `c^(q/p)` is kept, as
///   before.
fn power_preimages(arena: &mut Arena, c: ExprId, n: &Q) -> Option<Vec<ExprId>> {
    let p = n.numer().clone();
    let q = n.denom().clone();
    let c_zero = arena.is_zero_structural(c);
    if c_zero {
        // f^n = 0: f = 0 for n > 0, nothing for n < 0.
        return Some(if n.is_positive() {
            vec![arena.zero]
        } else {
            vec![]
        });
    }
    let symbolic = !crate::base::walk::free_symbols(arena, c).is_empty();
    let p_abs: i64 = num_traits::ToPrimitive::to_i64(&p.abs()).unwrap_or(i64::MAX);
    if q.is_one() || (!symbolic && p_abs <= MAX_POWER_PREIMAGES) {
        // u^|p| = c^(sign p), then f = u^q.
        let base = if p.is_negative() {
            let m1 = arena.neg_one;
            let inv = arena.pow(c, m1);
            crate::transforms::eval::eval(arena, inv)
        } else {
            c
        };
        if p_abs > MAX_INT_POWER_ROOTS {
            return None;
        }
        let roots = binomial_roots(arena, base, p_abs as usize);
        let q_id = arena.big_int(q);
        let mut out = Vec::with_capacity(roots.len());
        for r in roots {
            let f = arena.pow(r.value, q_id);
            let f = crate::transforms::expand::expand(arena, f);
            let f = crate::transforms::eval::eval(arena, f);
            if !out.contains(&f) {
                out.push(f);
            }
        }
        return Some(out);
    }
    // Principal candidate only: f = c^(1/n).
    let inv_n = Ratio::one() / n.clone();
    let inv_n_id = arena.num_ratio(inv_n);
    Some(vec![arena.pow(c, inv_n_id)])
}

/// Largest integer exponent `|n|` whose `n` roots are listed when peeling
/// `f^n = c`.
const MAX_INT_POWER_ROOTS: i64 = 64;

/// Continue peeling `inner` against each of several right-hand sides and
/// merge the (deduplicated) results (see [`peel_two_branches`]).
fn peel_branches(
    arena: &mut Arena,
    inner: ExprId,
    rhss: &[ExprId],
    var: ExprId,
    period: Option<ExprId>,
) -> Option<Vec<Solution>> {
    let mut solutions: Vec<Solution> = Vec::new();
    let mut saw_some = rhss.is_empty();
    for &rhs in rhss {
        if let Some(sols) = solve_by_peeling(arena, inner, rhs, var, period) {
            saw_some = true;
            for sol in sols {
                if !solutions.iter().any(|s| s.value == sol.value) {
                    solutions.push(sol);
                }
            }
        }
    }
    if solutions.is_empty() {
        if saw_some { Some(vec![]) } else { None }
    } else {
        Some(solutions)
    }
}

/// Continue peeling `inner` against two alternative right-hand sides and
/// merge the (deduplicated) results.
fn peel_two_branches(
    arena: &mut Arena,
    inner: ExprId,
    rhs1: ExprId,
    rhs2: ExprId,
    var: ExprId,
    period: Option<ExprId>,
) -> Option<Vec<Solution>> {
    let mut solutions = Vec::new();
    let mut saw_some = false;
    if let Some(sols) = solve_by_peeling(arena, inner, rhs1, var, period) {
        saw_some = true;
        solutions.extend(sols);
    }
    if let Some(sols) = solve_by_peeling(arena, inner, rhs2, var, period) {
        saw_some = true;
        for sol in sols {
            if !solutions.iter().any(|s| s.value == sol.value) {
                solutions.push(sol);
            }
        }
    }
    if solutions.is_empty() {
        // Distinguish "structure not invertible" (None) from "both branches
        // proved empty" (Some(vec![])).
        if saw_some { Some(vec![]) } else { None }
    } else {
        Some(solutions)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Linear solver: a*x + b = 0 → x = -b/a
// ═══════════════════════════════════════════════════════════════════════════

fn solve_linear(arena: &mut Arena, poly: &Poly) -> Vec<Solution> {
    let a = poly.coeff(1); // coefficient of x
    let b = poly.coeff(0); // constant term

    if a.is_zero() {
        return Vec::new(); // Degenerate: 0*x + b = 0.
    }

    // x = -b / a
    let value = -b / a;
    let value_id = arena.num_ratio(value.clone());

    vec![Solution { value: value_id }]
}

// ═══════════════════════════════════════════════════════════════════════════
// Quadratic solver: a*x² + b*x + c = 0
// ═══════════════════════════════════════════════════════════════════════════

fn solve_quadratic(arena: &mut Arena, poly: &Poly) -> Vec<Solution> {
    let a = poly.coeff(2);
    let b = poly.coeff(1);
    let c = poly.coeff(0);

    if a.is_zero() {
        // Actually linear.
        let linear = Poly::from_coeffs(vec![c, b]);
        return solve_linear(arena, &linear);
    }

    // Discriminant: b² - 4ac
    let discriminant = &b * &b - Ratio::from_integer(BigInt::from(4)) * &a * &c;

    if discriminant.is_zero() {
        // Double root: x = -b / (2a)
        let two_a = Ratio::from_integer(BigInt::from(2)) * &a;
        let value = -b / two_a;
        let value_id = arena.num_ratio(value.clone());
        return vec![Solution { value: value_id }];
    }

    if discriminant.is_negative() {
        // Complex roots: x = (-b ± i√|Δ|) / (2a)
        let abs_disc = -discriminant;
        let neg_b = arena.num_ratio((-&b).clone());
        let abs_disc_id = arena.num_ratio(abs_disc.clone());

        // Check if |Δ| is a perfect square
        let sqrt_abs_disc = if let Some(s) = rational_sqrt(&abs_disc) {
            arena.num_ratio(s.clone())
        } else {
            arena.sqrt(abs_disc_id)
        };

        let i_sqrt = arena.mul(&[arena.i_unit, sqrt_abs_disc]);
        let two_a_val = Ratio::from_integer(BigInt::from(2)) * &a;
        let two_a_id = arena.num_ratio(two_a_val.clone());
        let two_a_inv = {
            let neg_one = arena.neg_one;
            arena.pow(two_a_id, neg_one)
        };

        // x1 = (-b + i√|Δ|) / (2a)
        let sum1 = arena.add(&[neg_b, i_sqrt]);
        let x1 = arena.mul(&[sum1, two_a_inv]);

        // x2 = (-b - i√|Δ|) / (2a)
        let neg_i_sqrt = arena.neg(i_sqrt);
        let sum2 = arena.add(&[neg_b, neg_i_sqrt]);
        let x2 = arena.mul(&[sum2, two_a_inv]);

        return vec![Solution { value: x1 }, Solution { value: x2 }];
    }

    // Check if the discriminant is a perfect square (rational root).
    if let Some(sqrt_disc) = rational_sqrt(&discriminant) {
        let two_a = Ratio::from_integer(BigInt::from(2)) * &a;

        // x = (-b ± √Δ) / (2a)
        let x1 = (-&b + &sqrt_disc) / &two_a;
        let x2 = (-&b - &sqrt_disc) / &two_a;

        let x1_id = arena.num_ratio(x1.clone());
        let x2_id = arena.num_ratio(x2.clone());

        if x1 == x2 {
            vec![Solution { value: x1_id }]
        } else {
            vec![Solution { value: x1_id }, Solution { value: x2_id }]
        }
    } else {
        // Discriminant is not a perfect square — roots are irrational.
        // Express as (-b ± sqrt(discriminant)) / (2*a) symbolically.
        let neg_b = {
            let neg_b_val = -&b;
            arena.num_ratio(neg_b_val.clone())
        };
        let sqrt_disc = {
            let disc_id = arena.num_ratio(discriminant.clone());
            arena.sqrt(disc_id)
        };
        let two_a_val = Ratio::from_integer(BigInt::from(2)) * &a;
        let two_a_id = arena.num_ratio(two_a_val.clone());
        let two_a_inv = {
            let neg_one = arena.neg_one;
            arena.pow(two_a_id, neg_one)
        };

        // x1 = (-b + sqrt(disc)) / (2a)
        let sum1 = arena.add(&[neg_b, sqrt_disc]);
        let x1 = arena.mul(&[sum1, two_a_inv]);

        // x2 = (-b - sqrt(disc)) / (2a)
        let neg_sqrt = arena.neg(sqrt_disc);
        let sum2 = arena.add(&[neg_b, neg_sqrt]);
        let x2 = arena.mul(&[sum2, two_a_inv]);

        vec![Solution { value: x1 }, Solution { value: x2 }]
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Cubic solver: a*x³ + b*x² + c*x + d = 0
// ═══════════════════════════════════════════════════════════════════════════

/// Solve a cubic polynomial. Tries rational roots first, then falls back
/// to Cardano's formula for irrational / complex roots.
fn solve_cubic(arena: &mut Arena, var: ExprId, poly: &Poly) -> Vec<Solution> {
    let a = poly.coeff(3);
    if a.is_zero() {
        let quadratic = Poly::from_coeffs(vec![poly.coeff(0), poly.coeff(1), poly.coeff(2)]);
        return solve_quadratic(arena, &quadratic);
    }

    // Try rational roots first — exact answers are preferable.
    let rational_attempt = solve_rational_roots(arena, var, poly);
    if !rational_attempt.is_empty() {
        return rational_attempt;
    }

    // No rational roots — use Cardano's formula.
    solve_cubic_cardano(arena, poly)
}

/// Pure Cardano's formula (no rational-root attempt) to avoid re-entry loops.
fn solve_cubic_cardano(arena: &mut Arena, poly: &Poly) -> Vec<Solution> {
    let a = poly.coeff(3);
    let b = poly.coeff(2);
    let c = poly.coeff(1);
    let d = poly.coeff(0);

    if a.is_zero() {
        let quadratic = Poly::from_coeffs(vec![d, c, b]);
        return solve_quadratic(arena, &quadratic);
    }

    // Binomial a·x³ + d = 0: cleaner roots-of-unity form than Cardano.
    if let Some(roots) = try_solve_binomial_rational(arena, poly) {
        return roots;
    }

    let a_id = arena.num_ratio(a.clone());
    let b_id = arena.num_ratio(b.clone());
    let c_id = arena.num_ratio(c.clone());
    let d_id = arena.num_ratio(d.clone());

    let three = arena.int(3);
    let nine = arena.int(9);
    let twenty_seven = arena.int(27);
    let two = arena.int(2);
    let four = arena.int(4);

    // Depress to t³ + pt + q = 0  via  x = t - b/(3a)
    //   p = (3ac - b²) / (3a²)
    //   q = (2b³ - 9abc + 27a²d) / (27a³)

    // p_num = 3ac - b²
    let tmp1 = arena.mul(&[three, a_id, c_id]);
    let tmp2 = arena.mul(&[b_id, b_id]);
    let p_num = arena.sub(tmp1, tmp2);
    let p_den = arena.mul(&[three, a_id, a_id]);
    let p = arena.div(p_num, p_den);

    // q_num = 2b³ - 9abc + 27a²d
    let tmp3 = arena.mul(&[two, b_id, b_id, b_id]);
    let tmp4 = arena.mul(&[nine, a_id, b_id, c_id]);
    let tmp4n = arena.neg(tmp4);
    let tmp5 = arena.mul(&[twenty_seven, a_id, a_id, d_id]);
    let q_num = arena.add(&[tmp3, tmp4n, tmp5]);
    let q_den = arena.mul(&[twenty_seven, a_id, a_id, a_id]);
    let q = arena.div(q_num, q_den);

    // Discriminant: Δ = q²/4 + p³/27
    let qq = arena.mul(&[q, q]);
    let qq_over4 = arena.div(qq, four);
    let ppp = arena.mul(&[p, p, p]);
    let ppp_over27 = arena.div(ppp, twenty_seven);
    let disc = arena.add(&[qq_over4, ppp_over27]);

    // √Δ
    let half = arena.rational(1, 2);
    let sqrt_disc = arena.pow(disc, half);

    // Cardano: t = cbrt(-q/2 + √Δ) + cbrt(-q/2 - √Δ)
    let neg_q = arena.neg(q);
    let neg_q_half = arena.div(neg_q, two);

    let u_arg = arena.add(&[neg_q_half, sqrt_disc]);
    let v_arg = arena.sub(neg_q_half, sqrt_disc);

    // Cardano's formula needs the *real* cube roots of the two radicands
    // (their product must be −p/3).  A `Pow(negative, 1/3)` node is
    // evaluated on the principal complex branch by `evalf`, which would
    // silently produce wrong roots, so a radicand that is provably
    // negative is written as −cbrt(|radicand|).  The signs follow from the
    // exact rational data: for Δ ≥ 0, √Δ ≥ |q|/2 exactly when p ≥ 0.
    let (p_rat, q_rat, disc_rat) = {
        let three_r = Ratio::from_integer(BigInt::from(3));
        let nine_r = Ratio::from_integer(BigInt::from(9));
        let two_r = Ratio::from_integer(BigInt::from(2));
        let four_r = Ratio::from_integer(BigInt::from(4));
        let twenty_seven_r = Ratio::from_integer(BigInt::from(27));
        let p_r = (&three_r * &a * &c - &b * &b) / (&three_r * &a * &a);
        let q_r = (&two_r * &b * &b * &b - &nine_r * &a * &b * &c + &twenty_seven_r * &a * &a * &d)
            / (&twenty_seven_r * &a * &a * &a);
        let disc_r = &q_r * &q_r / &four_r + &p_r * &p_r * &p_r / &twenty_seven_r;
        (p_r, q_r, disc_r)
    };
    let (u_sign, v_sign) = cardano_radicand_signs(&p_rat, &q_rat, &disc_rat);

    let u = real_cbrt_with_sign(arena, u_arg, u_sign);
    let v = real_cbrt_with_sign(arena, v_arg, v_sign);

    let t1 = arena.add(&[u, v]);

    // Shift back: x = t - b/(3a)
    let three_a = arena.mul(&[three, a_id]);
    let shift = arena.div(b_id, three_a);
    let x1 = arena.sub(t1, shift);

    // Other two roots via cube roots of unity:
    //   ω  = (-1 + i√3)/2
    //   ω² = (-1 - i√3)/2
    let omega_re = arena.rational(-1, 2);
    let sqrt3 = arena.pow(three, half);
    let sqrt3_half = arena.div(sqrt3, two);
    let i_unit = arena.i_unit;
    let omega_im = arena.mul(&[sqrt3_half, i_unit]);
    let omega = arena.add(&[omega_re, omega_im]);
    let omega2 = arena.sub(omega_re, omega_im);

    let ou = arena.mul(&[omega, u]);
    let o2v = arena.mul(&[omega2, v]);
    let t2 = arena.add(&[ou, o2v]);
    let o2u = arena.mul(&[omega2, u]);
    let ov = arena.mul(&[omega, v]);
    let t3 = arena.add(&[o2u, ov]);

    let x2 = arena.sub(t2, shift);
    let x3 = arena.sub(t3, shift);

    // Simplify all roots through eval
    let x1s = crate::transforms::eval::eval(arena, x1);
    let x2s = crate::transforms::eval::eval(arena, x2);
    let x3s = crate::transforms::eval::eval(arena, x3);

    vec![
        Solution { value: x1s },
        Solution { value: x2s },
        Solution { value: x3s },
    ]
}

/// Signs of the Cardano radicands `−q/2 + √Δ` and `−q/2 − √Δ` for the
/// depressed cubic `t³ + pt + q` with `Δ = q²/4 + p³/27`.
///
/// Returns `(sign_u, sign_v)` with values in `{-1, 0, 1}`; `0` is also used
/// when `Δ < 0` (complex radicands, principal branch is correct there).
fn cardano_radicand_signs(p: &Q, q: &Q, disc: &Q) -> (i8, i8) {
    use std::cmp::Ordering;
    if disc.is_negative() {
        return (0, 0);
    }
    let neg_q_sign: i8 = match q.cmp(&Ratio::zero()) {
        Ordering::Less => 1,
        Ordering::Equal => 0,
        Ordering::Greater => -1,
    };
    if disc.is_zero() {
        // Both radicands equal −q/2.
        return (neg_q_sign, neg_q_sign);
    }
    // Δ > 0: √Δ > |q|/2 ⇔ p > 0; √Δ = |q|/2 ⇔ p = 0; √Δ < |q|/2 ⇔ p < 0.
    let p_sign = match p.cmp(&Ratio::zero()) {
        Ordering::Less => -1i8,
        Ordering::Equal => 0,
        Ordering::Greater => 1,
    };
    let u_sign = if neg_q_sign >= 0 {
        // −q/2 ≥ 0 and √Δ > 0 ⇒ positive.
        1
    } else {
        // −q/2 < 0: sign decided by whether √Δ exceeds |q|/2.
        p_sign
    };
    let v_sign = if neg_q_sign <= 0 {
        -1
    } else {
        // −q/2 > 0: −q/2 − √Δ is positive iff √Δ < q/2 ⇔ p < 0.
        -p_sign
    };
    (u_sign, v_sign)
}

/// Build the real cube root of `arg` given its known sign: `cbrt(arg)` for
/// a positive radicand, `−cbrt(−arg)` for a negative one (so that no cube
/// root of a negative real is ever emitted), `0` for a zero radicand.  An
/// unknown sign (`0` for a complex radicand) falls back to the principal
/// branch `arg^(1/3)`.
fn real_cbrt_with_sign(arena: &mut Arena, arg: ExprId, sign: i8) -> ExprId {
    let third = arena.rational(1, 3);
    match sign {
        1 => arena.pow(arg, third),
        -1 => {
            let neg_arg = arena.neg(arg);
            let neg_arg = crate::transforms::eval::eval(arena, neg_arg);
            let root = arena.pow(neg_arg, third);
            arena.neg(root)
        }
        _ => {
            // Zero radicand (exactly), or complex radicand (principal branch).
            let ev = crate::transforms::eval::eval(arena, arg);
            if arena.is_zero_structural(ev) {
                return arena.zero;
            }
            arena.pow(arg, third)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Quartic solver: Ferrari's method  a*x⁴ + b*x³ + c*x² + d*x + e = 0
// ═══════════════════════════════════════════════════════════════════════════

/// Solve a quartic polynomial. Tries rational roots first, then falls back
/// to Ferrari's method.
fn solve_quartic(arena: &mut Arena, var: ExprId, poly: &Poly) -> Vec<Solution> {
    let a = poly.coeff(4);
    if a.is_zero() {
        let cubic = Poly::from_coeffs(vec![
            poly.coeff(0),
            poly.coeff(1),
            poly.coeff(2),
            poly.coeff(3),
        ]);
        return solve_cubic(arena, var, &cubic);
    }

    // Try rational roots first.
    let rational_attempt = solve_rational_roots(arena, var, poly);
    if !rational_attempt.is_empty() {
        return rational_attempt;
    }

    // No rational roots — use Ferrari's method.
    solve_quartic_ferrari(arena, poly)
}

/// Pure Ferrari's method (no rational-root attempt) to avoid re-entry loops.
fn solve_quartic_ferrari(arena: &mut Arena, poly: &Poly) -> Vec<Solution> {
    let a = poly.coeff(4);
    let b = poly.coeff(3);
    let c = poly.coeff(2);
    let d = poly.coeff(1);
    let e = poly.coeff(0);

    if a.is_zero() {
        let cubic = Poly::from_coeffs(vec![e.clone(), d, c, b]);
        return solve_cubic_cardano(arena, &cubic);
    }

    // Binomial a·x⁴ + e = 0: cleaner roots-of-unity form than Ferrari.
    if let Some(roots) = try_solve_binomial_rational(arena, poly) {
        return roots;
    }

    // Depress to t⁴ + pt² + qt + r = 0  via  x = t - b/(4a)
    //   p = (8ac - 3b²) / (8a²)
    //   q = (b³ - 4abc + 8a²d) / (8a³)
    //   r = (-3b⁴ + 256a³e - 64a²bd + 16ab²c) / (256a⁴)

    // Exact rational depressed coefficients (also used for the biquadratic
    // shortcut below and for the resolvent-root filter).
    let (p_rat, q_rat, r_rat) = {
        let r3 = Ratio::from_integer(BigInt::from(3));
        let r4 = Ratio::from_integer(BigInt::from(4));
        let r8 = Ratio::from_integer(BigInt::from(8));
        let r16 = Ratio::from_integer(BigInt::from(16));
        let r64 = Ratio::from_integer(BigInt::from(64));
        let r256 = Ratio::from_integer(BigInt::from(256));
        let a2 = &a * &a;
        let a3 = &a2 * &a;
        let a4 = &a3 * &a;
        let b2 = &b * &b;
        let p_r = (&r8 * &a * &c - &r3 * &b2) / (&r8 * &a2);
        let q_r = (&b2 * &b - &r4 * &a * &b * &c + &r8 * &a2 * &d) / (&r8 * &a3);
        let r_r = (-&r3 * &b2 * &b2 + &r256 * &a3 * &e - &r64 * &a2 * &b * &d
            + &r16 * &a * &b2 * &c)
            / (&r256 * &a4);
        (p_r, q_r, r_r)
    };

    // Biquadratic t⁴ + pt² + r = 0 (q = 0): Ferrari's factorisation
    // degenerates (k = √(2m − p) = 0 for the rational resolvent root
    // m = p/2, giving 0/0), so solve the quadratic in s = t² and take
    // t = ±√s instead.
    if q_rat.is_zero() {
        let quad = Poly::from_coeffs(vec![r_rat.clone(), p_rat.clone(), Ratio::one()]);
        let s_roots = solve_quadratic(arena, &quad);
        let four_a = Ratio::from_integer(BigInt::from(4)) * &a;
        let shift_rat = &b / &four_a;
        let shift = arena.num_ratio(shift_rat.clone());
        let half = arena.rational(1, 2);
        let mut out: Vec<Solution> = Vec::new();
        for s in s_roots {
            let s_ev = crate::transforms::eval::eval(arena, s.value);
            let t_pos = arena.pow(s_ev, half);
            let t_neg = arena.neg(t_pos);
            for t in [t_pos, t_neg] {
                let x = arena.sub(t, shift);
                let x = crate::transforms::eval::eval(arena, x);
                if !out.iter().any(|o| o.value == x) {
                    out.push(Solution { value: x });
                }
            }
        }
        return out;
    }

    let a_id = arena.num_ratio(a.clone());
    let b_id = arena.num_ratio(b.clone());
    let c_id = arena.num_ratio(c.clone());
    let d_id = arena.num_ratio(d.clone());
    let e_id = arena.num_ratio(e.clone());

    let two = arena.int(2);
    let three = arena.int(3);
    let four = arena.int(4);
    let eight = arena.int(8);
    let sixteen = arena.int(16);
    let sixty_four = arena.int(64);
    let two_fifty_six = arena.int(256);

    // p = (8ac - 3b²) / (8a²)
    let tmp_8ac = arena.mul(&[eight, a_id, c_id]);
    let tmp_3bb = arena.mul(&[three, b_id, b_id]);
    let p_num = arena.sub(tmp_8ac, tmp_3bb);
    let p_den = arena.mul(&[eight, a_id, a_id]);
    let p = arena.div(p_num, p_den);

    // q = (b³ - 4abc + 8a²d) / (8a³)
    let tmp_bbb = arena.mul(&[b_id, b_id, b_id]);
    let tmp_4abc = arena.mul(&[four, a_id, b_id, c_id]);
    let tmp_4abc_n = arena.neg(tmp_4abc);
    let tmp_8aad = arena.mul(&[eight, a_id, a_id, d_id]);
    let q_num = arena.add(&[tmp_bbb, tmp_4abc_n, tmp_8aad]);
    let q_den = arena.mul(&[eight, a_id, a_id, a_id]);
    let q = arena.div(q_num, q_den);

    // r = (-3b⁴ + 256a³e - 64a²bd + 16ab²c) / (256a⁴)
    let tmp_3b4 = arena.mul(&[three, b_id, b_id, b_id, b_id]);
    let tmp_3b4_n = arena.neg(tmp_3b4);
    let tmp_256a3e = arena.mul(&[two_fifty_six, a_id, a_id, a_id, e_id]);
    let tmp_64a2bd = arena.mul(&[sixty_four, a_id, a_id, b_id, d_id]);
    let tmp_64a2bd_n = arena.neg(tmp_64a2bd);
    let tmp_16ab2c = arena.mul(&[sixteen, a_id, b_id, b_id, c_id]);
    let r_num = arena.add(&[tmp_3b4_n, tmp_256a3e, tmp_64a2bd_n, tmp_16ab2c]);
    let r_den = arena.mul(&[two_fifty_six, a_id, a_id, a_id, a_id]);
    let r = arena.div(r_num, r_den);

    // Resolvent cubic:  8m³ - 4pm² - 8rm + (4pr - q²) = 0
    // We need the coefficients as rationals to build a Poly.
    let resolvent_c3 = eight;
    let neg_four = arena.neg(four);
    let resolvent_c2 = arena.mul(&[neg_four, p]);
    let neg_eight = arena.neg(eight);
    let resolvent_c1 = arena.mul(&[neg_eight, r]);
    let tmp_4pr = arena.mul(&[four, p, r]);
    let tmp_qq = arena.mul(&[q, q]);
    let tmp_qq_n = arena.neg(tmp_qq);
    let resolvent_c0 = arena.add(&[tmp_4pr, tmp_qq_n]);

    let resolvent_c3_e = crate::transforms::eval::eval(arena, resolvent_c3);
    let resolvent_c2_e = crate::transforms::eval::eval(arena, resolvent_c2);
    let resolvent_c1_e = crate::transforms::eval::eval(arena, resolvent_c1);
    let resolvent_c0_e = crate::transforms::eval::eval(arena, resolvent_c0);

    let rc3 = match arena.as_num(resolvent_c3_e) {
        Some(v) => v.clone(),
        None => return Vec::new(),
    };
    let rc2 = match arena.as_num(resolvent_c2_e) {
        Some(v) => v.clone(),
        None => return Vec::new(),
    };
    let rc1 = match arena.as_num(resolvent_c1_e) {
        Some(v) => v.clone(),
        None => return Vec::new(),
    };
    let rc0 = match arena.as_num(resolvent_c0_e) {
        Some(v) => v.clone(),
        None => return Vec::new(),
    };

    let resolvent_poly = Poly::from_coeffs(vec![rc0, rc1, rc2, rc3]);

    // We need 2m - p ≠ 0 for a non-degenerate Ferrari factorization; with
    // q ≠ 0 (guaranteed above) m = p/2 is never a resolvent root.
    debug_assert!(!q_rat.is_zero());

    // Try rational roots of the resolvent cubic first.
    // This avoids the *casus irreducibilis* problem where Cardano's formula
    // produces complex cube roots for real rational roots, yielding
    // unsimplifiable expressions like cbrt(±i·√(1/27)).
    let m = if let Some(m_rat) = find_preferred_resolvent_root(&resolvent_poly, &p_rat) {
        arena.num_ratio(m_rat.clone())
    } else {
        let m_solutions = solve_cubic_cardano(arena, &resolvent_poly);
        if m_solutions.is_empty() {
            return Vec::new();
        }
        m_solutions[0].value
    };

    // Factor into two quadratics via √(2m − p):
    //   t² + k·t + (m − q/(2k)) = 0
    //   t² − k·t + (m + q/(2k)) = 0
    // where k = √(2m − p)
    let half = arena.rational(1, 2);
    let two_m = arena.mul(&[two, m]);
    let two_m_minus_p = arena.sub(two_m, p);
    let k = arena.pow(two_m_minus_p, half);

    let two_k = arena.mul(&[two, k]);
    let q_over_2k = arena.div(q, two_k);

    // shift = b/(4a)
    let four_a = arena.mul(&[four, a_id]);
    let shift = arena.div(b_id, four_a);

    // Quadratic 1:  t² + kt + (m - q/(2k)) = 0
    let s1 = arena.sub(m, q_over_2k);
    let kk = arena.mul(&[k, k]);
    let four_s1 = arena.mul(&[four, s1]);
    let disc1 = arena.sub(kk, four_s1);
    let sqrt_disc1 = arena.pow(disc1, half);
    let neg_k = arena.neg(k);

    let sum1a = arena.add(&[neg_k, sqrt_disc1]);
    let t1a = arena.div(sum1a, two);
    let diff1b = arena.sub(neg_k, sqrt_disc1);
    let t1b = arena.div(diff1b, two);

    let x1 = arena.sub(t1a, shift);
    let x2 = arena.sub(t1b, shift);

    // Quadratic 2:  t² - kt + (m + q/(2k)) = 0
    let s2 = arena.add(&[m, q_over_2k]);
    let kk2 = arena.mul(&[k, k]);
    let four_s2 = arena.mul(&[four, s2]);
    let disc2 = arena.sub(kk2, four_s2);
    let sqrt_disc2 = arena.pow(disc2, half);

    let sum2a = arena.add(&[k, sqrt_disc2]);
    let t2a = arena.div(sum2a, two);
    let diff2b = arena.sub(k, sqrt_disc2);
    let t2b = arena.div(diff2b, two);

    let x3 = arena.sub(t2a, shift);
    let x4 = arena.sub(t2b, shift);

    // Simplify all roots
    let x1s = crate::transforms::eval::eval(arena, x1);
    let x2s = crate::transforms::eval::eval(arena, x2);
    let x3s = crate::transforms::eval::eval(arena, x3);
    let x4s = crate::transforms::eval::eval(arena, x4);

    vec![
        Solution { value: x1s },
        Solution { value: x2s },
        Solution { value: x3s },
        Solution { value: x4s },
    ]
}

/// Find a rational root of the resolvent cubic that yields a non-degenerate
/// Ferrari factorization (i.e. `2m − p ≠ 0`, so that `k = √(2m−p) ≠ 0`).
///
/// Uses the Rational Root Theorem: for a polynomial with integer coefficients,
/// every rational root `p/q` satisfies `p | a₀` and `q | aₙ`.
///
/// Returns `None` if no rational root is found (Cardano fallback will be used).
fn find_preferred_resolvent_root(resolvent: &Poly, p_rat: &Q) -> Option<Q> {
    let (int_poly, _scale) = clear_denominators(resolvent);
    let a0 = int_poly.coeff(0).to_integer();
    let an = {
        let v = int_poly.leading_coeff()?;
        v.to_integer()
    };

    let two_r = Ratio::from_integer(BigInt::from(2));
    let mut fallback: Option<Q> = None;

    if a0.is_zero() {
        // m = 0 is a root.  Record it but keep looking for a non-degenerate one.
        let zero = Ratio::zero();
        if &two_r * &zero - p_rat != Ratio::zero() {
            return Some(zero);
        }
        fallback = Some(zero);

        // Divide out m and check the remaining quadratic for rational roots.
        let reduced_coeffs: Vec<Q> = resolvent.coeffs().iter().skip(1).cloned().collect();
        let reduced = Poly::from_coeffs(reduced_coeffs);
        let aq = reduced.coeff(2);
        let bq = reduced.coeff(1);
        let cq = reduced.coeff(0);
        if !aq.is_zero() {
            let disc = &bq * &bq - Ratio::from_integer(BigInt::from(4)) * &aq * &cq;
            if let Some(sqrt_d) = rational_sqrt(&disc) {
                let two_aq = Ratio::from_integer(BigInt::from(2)) * &aq;
                for candidate in [(-&bq + &sqrt_d) / &two_aq, (-&bq - &sqrt_d) / &two_aq] {
                    if &two_r * &candidate - p_rat != Ratio::zero() {
                        return Some(candidate);
                    }
                    if fallback.is_none() {
                        fallback = Some(candidate);
                    }
                }
            }
        }
    } else {
        let divs_a0 = divisors(&a0.abs());
        let divs_an = divisors(&an.abs());

        for p_div in &divs_a0 {
            for q_div in &divs_an {
                for &sign in &[1i64, -1i64] {
                    let candidate = Ratio::new(p_div * BigInt::from(sign), q_div.clone());
                    if resolvent.eval(&candidate).is_zero() {
                        // Prefer a root where 2m - p ≠ 0.
                        if &two_r * &candidate - p_rat != Ratio::zero() {
                            return Some(candidate);
                        }
                        if fallback.is_none() {
                            fallback = Some(candidate);
                        }
                    }
                }
            }
        }
    }

    fallback
}

// ═══════════════════════════════════════════════════════════════════════════
// Exact factorisation over ℤ for higher-degree polynomials
// ═══════════════════════════════════════════════════════════════════════════

/// The roots of a polynomial over ℚ through its exact factorisation over
/// ℤ ([`Poly::factor_over_z`]: content, Yun's square-free decomposition,
/// Berlekamp–Zassenhaus).
///
/// The content is dropped first, so `c·p` has exactly the roots of `p` for
/// every rational `c ≠ 0`, and every irreducible factor is solved once,
/// whatever its multiplicity: a linear factor *is* a rational root, a
/// quadratic, cubic or quartic factor goes to the radical formulas, and a
/// `RootOf(f, k)` placeholder is emitted only for an irreducible `f` of
/// degree ≥ 5, with `k` running over the roots of *that factor* — never
/// over a reducible or non-square-free polynomial.  The result lists the
/// **distinct** roots.
///
/// This replaced a Rational Root Theorem search whose divisor enumeration
/// gave up (silently, returning `{1, |n|}`) for a coefficient above `10⁶`,
/// so that `4·p(x)` lost the rational roots `p(x)` had, and which divided a
/// found root out once only, leaving its multiplicity in the cofactor
/// handed to `RootOf`.
///
/// Used for degree ≥ 5 and as the first pass of the cubic and quartic
/// solvers.  A binomial `a·xⁿ + b` of degree ≥ 5 keeps its roots-of-unity
/// form (before factoring, so `x⁵ − 32` is not split into a linear and a
/// quartic factor).
fn solve_rational_roots(arena: &mut Arena, var: ExprId, poly: &Poly) -> Vec<Solution> {
    let degree = match poly.degree() {
        Some(d) if d >= 1 => d,
        _ => return Vec::new(),
    };

    // Binomial a·xⁿ + b = 0 (n ≥ 5): all n roots explicitly via roots of
    // unity.  Cubics/quartics get here via their own solvers, which fall
    // back to the binomial form only after rational-root extraction.
    if degree > 4
        && let Some(roots) = try_solve_binomial_rational(arena, poly)
    {
        return roots;
    }

    let (_content, factors) = poly.factor_over_z();
    let mut roots: Vec<Solution> = Vec::new();
    for (factor, _multiplicity) in &factors {
        let found = match factor.degree() {
            None | Some(0) => Vec::new(),
            Some(1) => solve_linear(arena, factor),
            Some(2) => solve_quadratic(arena, factor),
            // Irreducible over ℚ: no rational root to extract, straight to
            // the radical formulas (which are invariant under the scaling
            // that made the factor primitive).
            Some(3) => solve_cubic_cardano(arena, factor),
            Some(4) => solve_quartic_ferrari(arena, factor),
            Some(d) => {
                if let Some(explicit) = try_solve_binomial_rational(arena, factor) {
                    explicit
                } else {
                    let factor_expr = polybridge::poly_to_expr(arena, factor, var);
                    (0..d)
                        .map(|i| {
                            let idx = arena.int(i as i64);
                            Solution {
                                value: arena.intern(ExprNode::RootOf(factor_expr, var, idx)),
                            }
                        })
                        .collect()
                }
            }
        };
        for sol in found {
            if !roots.iter().any(|s| s.value == sol.value) {
                roots.push(sol);
            }
        }
    }

    tracing::debug!(
        factors = factors.len(),
        roots = roots.len(),
        "roots via factorisation over ℤ"
    );
    roots
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Try to compute the exact square root of a non-negative rational.
///
/// Returns `Some(√r)` if `r` is a perfect square (both numerator and
/// denominator are perfect squares), or `None` otherwise.
fn rational_sqrt(r: &Q) -> Option<Q> {
    if r.is_negative() {
        return None;
    }
    if r.is_zero() {
        return Some(Ratio::zero());
    }

    let n = r.numer().abs();
    let d = r.denom().abs();

    let sqrt_n = n.sqrt();
    let sqrt_d = d.sqrt();

    if &sqrt_n * &sqrt_n == n && &sqrt_d * &sqrt_d == d {
        Some(Ratio::new(sqrt_n, sqrt_d))
    } else {
        None
    }
}

/// Clear denominators of a polynomial's coefficients.
///
/// Returns the integer-coefficient polynomial and the scale factor.
fn clear_denominators(poly: &Poly) -> (Poly, BigInt) {
    if poly.is_zero() {
        return (Poly::zero(), BigInt::one());
    }

    // Compute LCM of all denominators.
    let mut lcm = BigInt::one();
    for c in poly.coeffs() {
        let d = c.denom().abs();
        lcm = num_integer::lcm(lcm, d);
    }

    // Multiply all coefficients by the LCM.
    let scale = Ratio::from_integer(lcm.clone());
    let scaled = poly.scale(&scale);

    (scaled, lcm)
}

/// Compute all positive divisors of `n` (a positive BigInt).
///
/// Returns them in ascending order.  For n=0, returns `[1]` as a
/// fallback.
fn divisors(n: &BigInt) -> Vec<BigInt> {
    if n.is_zero() {
        return vec![BigInt::one()];
    }

    let n_abs = n.abs();

    // For small numbers, brute-force trial division.
    // For large numbers, this would be slow — but our polynomials
    // typically have small coefficients.
    let limit: u64 = match (&n_abs).try_into() {
        Ok(v) if v <= 1_000_000u64 => v,
        _ => {
            // Very large coefficient — just return 1 and n.
            return vec![BigInt::one(), n_abs];
        }
    };

    let mut result = Vec::new();
    let mut i = 1u64;
    while i * i <= limit {
        if limit.is_multiple_of(i) {
            result.push(BigInt::from(i));
            if i * i != limit {
                result.push(BigInt::from(limit / i));
            }
        }
        i += 1;
    }

    result.sort();
    result
}

// ═══════════════════════════════════════════════════════════════════════════
// Change-of-variable solving
// ═══════════════════════════════════════════════════════════════════════════

/// Try solving by detecting that the expression is polynomial in some f(x).
/// E.g., `exp(2x) - 3*exp(x) + 2` is polynomial in `t = exp(x)`: `t² - 3t + 2`.
fn try_change_of_variable(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    period: Option<ExprId>,
) -> Option<Vec<Solution>> {
    let var_sym = match arena.node(var) {
        ExprNode::Symbol(sid) => *sid,
        _ => return None,
    };

    // Collect candidate generators: inner functions applied to var
    let candidates = collect_generators(arena, expr, var, var_sym);

    for generator in candidates {
        // Create a temporary variable for substitution
        let t = arena.symbol("__t_subst");

        // --- Simple case: direct structural substitution of generator → t ---
        // This handles e.g. sin(x)^2 - sin(x) where walk_and_rebuild
        // naturally replaces sin(x) inside Pow(sin(x), 2) as well.
        let substituted = arena.subs_structural(expr, generator, t);

        if !expr_contains_var(arena, substituted, var) {
            let t_solutions = solve(arena, substituted, t);

            if !t_solutions.is_empty() {
                // Back-substitute: for each t = c, solve generator(var) = c.
                // Use solve_by_peeling directly to avoid recursing into
                // try_change_of_variable again (which would infinite-loop).
                let mut var_solutions: Vec<Solution> = Vec::new();
                for t_sol in &t_solutions {
                    if let Some(back_sols) =
                        solve_by_peeling(arena, generator, t_sol.value, var, period)
                    {
                        for s in back_sols {
                            if !var_solutions.iter().any(|v| v.value == s.value) {
                                var_solutions.push(s);
                            }
                        }
                    }
                }
                if !var_solutions.is_empty() {
                    return Some(var_solutions);
                }
            }
        }

        // --- Advanced: detect exp(n*x) = exp(x)^n pattern ---
        // exp(2*x) is Exp(Mul([2, x])) which does NOT structurally
        // contain exp(x) = Exp(x), so simple substitution misses it.
        // We rewrite exp(k*x) → exp(x)^k first, then substitute.
        if let ExprNode::Exp(inner) = arena.node(generator).clone()
            && inner == var
        {
            let rewritten = rewrite_exp_powers(arena, expr, var, generator);
            if rewritten != expr {
                let substituted2 = arena.subs_structural(rewritten, generator, t);
                if !expr_contains_var(arena, substituted2, var) {
                    let t_solutions = solve(arena, substituted2, t);
                    if !t_solutions.is_empty() {
                        let mut var_solutions: Vec<Solution> = Vec::new();
                        for t_sol in &t_solutions {
                            if let Some(back_sols) =
                                solve_by_peeling(arena, generator, t_sol.value, var, period)
                            {
                                for s in back_sols {
                                    if !var_solutions.iter().any(|v| v.value == s.value) {
                                        var_solutions.push(s);
                                    }
                                }
                            }
                        }
                        if !var_solutions.is_empty() {
                            return Some(var_solutions);
                        }
                    }
                }
            }
        }
    }

    None
}

/// Equations in `x` and rational powers `x^(p/q)` of `x` itself
/// (`√x = x − 2`, `x^(2/3) + x^(1/3) = 2`): with `L` the lcm of the
/// denominators and `t = x^(1/L)` (principal), `x^(p/q) = t^(pL/q)` and
/// `x = t^L`, so the equation becomes one in `t` alone; each root `t₀`
/// gives the candidate `x = t₀^L`.  The identity holds only for `t` in the
/// principal sector (`−π/L < arg t ≤ π/L`), so a root outside it yields a
/// spurious `x` (`t = −1` gives `x = 1`, where `√1 = 1 ≠ 1 − 2`); the
/// check of every candidate against the equation removes those.
///
/// `None` when `x` occurs other than bare or as the base of a rational
/// power, or when no power has a denominator.
fn try_radical_substitution(arena: &mut Arena, expr: ExprId, var: ExprId) -> Option<Vec<Solution>> {
    let post = crate::base::walk::post_order_ids(arena, expr);
    let mut lcm = BigInt::one();
    let mut powers: Vec<(ExprId, Q)> = Vec::new();
    for &id in &post {
        if !expr_contains_var(arena, id, var) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Pow(b, e) if *b == var => {
                let r = arena.as_num(*e)?.clone();
                if !r.is_integer() {
                    lcm = num_integer::Integer::lcm(&lcm, r.denom());
                }
                powers.push((id, r));
            }
            // `x` elsewhere must be bare: a power's exponent or a function
            // argument that is not `x` or `x^(p/q)` is not handled here.
            ExprNode::Pow(_, e) if expr_contains_var(arena, *e, var) => return None,
            ExprNode::Symbol(_)
            | ExprNode::Add(_)
            | ExprNode::Mul(_)
            | ExprNode::Neg(_)
            | ExprNode::Pow(..) => {}
            _ => return None,
        }
    }
    if lcm.is_one() || lcm > BigInt::from(MAX_RADICAL_INDEX) {
        return None;
    }
    let t = arena.symbol("__t_radical");
    let l_id = arena.big_int(lcm.clone());
    let mut substituted = expr;
    for (pow_id, r) in &powers {
        let k = r * Ratio::from_integer(lcm.clone());
        let k_id = arena.num_ratio(k);
        let tk = arena.pow(t, k_id);
        substituted = arena.subs_structural(substituted, *pow_id, tk);
    }
    let t_l = arena.pow(t, l_id);
    substituted = crate::transforms::subs::subs(arena, substituted, var, t_l);
    let substituted = crate::transforms::eval::eval(arena, substituted);
    if expr_contains_var(arena, substituted, var) {
        return None;
    }
    let t_solutions = solve(arena, substituted, t);
    let mut out: Vec<Solution> = Vec::new();
    for ts in t_solutions {
        let x0 = arena.pow(ts.value, l_id);
        let x0 = crate::transforms::expand::expand(arena, x0);
        let x0 = crate::transforms::eval::eval(arena, x0);
        if !out.iter().any(|s| s.value == x0) {
            out.push(Solution { value: x0 });
        }
    }
    Some(out)
}

/// Largest root index `L` of [`try_radical_substitution`].
const MAX_RADICAL_INDEX: i64 = 12;

/// Collect candidate generator functions: exp(x), sin(x), cos(x), ln(x), x^(k), etc.
fn collect_generators(arena: &Arena, expr: ExprId, var: ExprId, var_sym: SymbolId) -> Vec<ExprId> {
    let mut generators = Vec::new();
    let mut visited = std::collections::HashSet::new();
    collect_gens_recursive(arena, expr, var, var_sym, &mut generators, &mut visited);
    generators
}

#[allow(clippy::only_used_in_recursion)]
fn collect_gens_recursive(
    arena: &Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
    gens: &mut Vec<ExprId>,
    visited: &mut std::collections::HashSet<ExprId>,
) {
    if !visited.insert(expr) {
        return;
    }
    match arena.node(expr).clone() {
        ExprNode::Exp(inner) if inner == var && !gens.contains(&expr) => {
            gens.push(expr);
        }
        ExprNode::Sin(inner) | ExprNode::Cos(inner) | ExprNode::Tan(inner)
            if inner == var && !gens.contains(&expr) =>
        {
            gens.push(expr);
        }
        ExprNode::Ln(inner) if inner == var && !gens.contains(&expr) => {
            gens.push(expr);
        }
        ExprNode::Pow(base, exp)
            if base == var && !expr_contains_var(arena, exp, var)
            // x^(1/n) or x^k type
            && !gens.contains(&expr) =>
        {
            gens.push(expr);
        }
        ExprNode::Add(ref children) | ExprNode::Mul(ref children) => {
            for &c in children {
                collect_gens_recursive(arena, c, var, var_sym, gens, visited);
            }
        }
        ExprNode::Pow(base, exp) => {
            collect_gens_recursive(arena, base, var, var_sym, gens, visited);
            collect_gens_recursive(arena, exp, var, var_sym, gens, visited);
        }
        ExprNode::Neg(inner)
        | ExprNode::Exp(inner)
        | ExprNode::Ln(inner)
        | ExprNode::Sin(inner)
        | ExprNode::Cos(inner)
        | ExprNode::Tan(inner) => {
            collect_gens_recursive(arena, inner, var, var_sym, gens, visited);
        }
        _ => {}
    }
}

/// Rewrite `exp(k*x)` as `exp(x)^k` for small integer k throughout the expression.
fn rewrite_exp_powers(arena: &mut Arena, expr: ExprId, var: ExprId, gen_exp_x: ExprId) -> ExprId {
    let mut result = expr;
    // Check small positive integer multiples: exp(k*x) → exp(x)^k
    for k in 2i64..=6 {
        let k_id = arena.int(k);
        let k_var = arena.mul(&[k_id, var]);
        let exp_k_var = arena.exp(k_var);
        let gen_pow_k = arena.pow(gen_exp_x, k_id);
        result = arena.subs_structural(result, exp_k_var, gen_pow_k);
    }
    // Also handle negative multiples: exp(-k*x) → exp(x)^(-k)
    for k in [-1i64, -2, -3] {
        let k_id = arena.int(k);
        let k_var = arena.mul(&[k_id, var]);
        let exp_k_var = arena.exp(k_var);
        let gen_pow_k = arena.pow(gen_exp_x, k_id);
        result = arena.subs_structural(result, exp_k_var, gen_pow_k);
    }
    result
}

// ═══════════════════════════════════════════════════════════════════════════
// LambertW solving for mixed polynomial-exponential equations
// ═══════════════════════════════════════════════════════════════════════════

/// Try to solve equations involving mixed polynomial and exponential or
/// logarithmic terms with the Lambert W function (after SymPy's
/// `_solve_lambert` / `_lambert`, `sympy/solvers/bivariate.py`, BSD):
///
/// - `A·x·exp(B·x) + D = 0`  →  `x = W_k(−B·D/A)/B`
/// - `A·exp(B·x) + C·x + D = 0`  →  `x = −W_k(A·B/(C·exp(B·D/C)))/B − D/C`
/// - `A·x·ln(B·x) + C·x + D = 0`  →  `x = exp(W_k(−(B·D/A)·exp(C/A)) − C/A)/B`
/// - `A·ln(B·x) + C·x + D = 0`  →  `x = (A/C)·W_k((C/(A·B))·exp(−D/A))`
/// - `A·x^x + D = 0`  →  `x = exp(W_k(ln(−D/A)))`
///
/// (with `ln(B·x)` for a constant `B` solved in `B·x`).  The branches
/// `W_k` returned are those of [`lambert_branches`]: the principal one, and
/// `W₋₁` as well when the argument lies in `(−1/e, 0)`, where both are real
/// and the equation has two real roots.  Before 0.30 only `W₀` was taken:
/// `solve(exp(x) − 2x − π, x)` gave `−1.4540` and missed `1.9526`
/// (`−π/2 − W₋₁(−e^{−π/2}/2)`), and the logarithmic forms and `x^x` were
/// not recognised at all.
fn try_solve_lambert(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    _var_sym: SymbolId,
) -> Option<Vec<Solution>> {
    tracing::debug!("solve: trying LambertW");

    // Get additive terms
    let terms: Vec<ExprId> = match arena.node(expr).clone() {
        ExprNode::Add(children) => children.to_vec(),
        _ => vec![expr],
    };

    // Classify each term into categories
    let mut constant_terms: Vec<ExprId> = Vec::new();
    let mut linear_coeffs: Vec<ExprId> = Vec::new(); // A for A*var
    let mut exp_terms: Vec<LambertExp> = Vec::new(); // A*exp(B*var)
    let mut var_exp_terms: Vec<LambertExp> = Vec::new(); // A*var*exp(B*var)
    let mut var_ln_terms: Vec<LambertLn> = Vec::new(); // A*var*ln(B*var)
    let mut ln_terms: Vec<LambertLn> = Vec::new(); // A*ln(B*var)
    let mut pow_terms: Vec<ExprId> = Vec::new(); // A for A*var^var

    for &term in &terms {
        if !expr_contains_var(arena, term, var) {
            constant_terms.push(term);
            continue;
        }

        match classify_lambert_term(arena, term, var) {
            Some(LambertTermClass::Linear(coeff)) => linear_coeffs.push(coeff),
            Some(LambertTermClass::ExpVar(e)) => exp_terms.push(e),
            Some(LambertTermClass::VarExpVar(e)) => var_exp_terms.push(e),
            Some(LambertTermClass::VarLnVar(ln)) => var_ln_terms.push(ln),
            Some(LambertTermClass::LnVar(ln)) => ln_terms.push(ln),
            Some(LambertTermClass::PowVarVar(coeff)) => pow_terms.push(coeff),
            None => return None,
        }
    }

    // Build constant sum (D)
    let d = match constant_terms.len() {
        0 => arena.zero,
        1 => constant_terms[0],
        _ => arena.add(&constant_terms),
    };
    let c = match linear_coeffs.len() {
        0 => None,
        1 => Some(linear_coeffs[0]),
        _ => Some(arena.add(&linear_coeffs)),
    };
    let counts = (
        exp_terms.len(),
        var_exp_terms.len(),
        var_ln_terms.len(),
        ln_terms.len(),
        pow_terms.len(),
    );

    let solutions: Vec<ExprId> = match (counts, c) {
        // ── A*var*exp(B*var) + D = 0 ─────────────────────────────────
        //   B*var*exp(B*var) = -B*D/A  →  var = W(-B*D/A)/B
        ((0, 1, 0, 0, 0), None) => {
            let LambertExp { coeff: a, rate: b } = var_exp_terms[0];
            let neg_d = arena.neg(d);
            let neg_d_over_a = arena.div(neg_d, a);
            let b_arg = arena.mul(&[b, neg_d_over_a]);
            lambert_branches(arena, b_arg)
                .into_iter()
                .map(|w| arena.div(w, b))
                .collect()
        }
        // ── A*exp(B*var) + C*var + D = 0 ─────────────────────────────
        //   u = -(B*var + B*D/C):  u·exp(u) = A·B/(C·exp(B·D/C))
        //   var = -W(…)/B - D/C
        ((1, 0, 0, 0, 0), Some(c)) => {
            let LambertExp {
                coeff: a_exp,
                rate: b,
            } = exp_terms[0];
            let b_d = arena.mul(&[b, d]);
            let b_d_over_c = arena.div(b_d, c);
            let exp_bd_c = arena.exp(b_d_over_c);
            let c_exp_bd_c = arena.mul(&[c, exp_bd_c]);
            let a_b = arena.mul(&[a_exp, b]);
            let w_arg = arena.div(a_b, c_exp_bd_c);
            let d_over_c = arena.div(d, c);
            lambert_branches(arena, w_arg)
                .into_iter()
                .map(|w| {
                    let neg_w = arena.neg(w);
                    let neg_w_over_b = arena.div(neg_w, b);
                    arena.sub(neg_w_over_b, d_over_c)
                })
                .collect()
        }
        // ── A*var*ln(B*var) + C*var + D = 0 ──────────────────────────
        //   B*var = e^t:  (t + C/A)·e^{t + C/A} = -(B·D/A)·e^{C/A}
        //   var = exp(W(-(B·D/A)·e^{C/A}) - C/A)/B
        ((0, 0, 1, 0, 0), c) => {
            let LambertLn { coeff: a, scale: b } = var_ln_terms[0];
            let c_over_a = match c {
                Some(c) => arena.div(c, a),
                None => arena.zero,
            };
            let neg_bd = arena.mul(&[arena.neg_one, b, d]);
            let neg_bd_over_a = arena.div(neg_bd, a);
            let exp_c_a = arena.exp(c_over_a);
            let w_arg = arena.mul(&[neg_bd_over_a, exp_c_a]);
            lambert_branches(arena, w_arg)
                .into_iter()
                .map(|w| {
                    let t = arena.sub(w, c_over_a);
                    let y = arena.exp(t);
                    arena.div(y, b)
                })
                .collect()
        }
        // ── A*ln(B*var) + C*var + D = 0 ──────────────────────────────
        //   var·e^{(C/A)·var} = e^{-D/A}/B
        //   var = (A/C)·W((C/(A·B))·e^{-D/A})
        ((0, 0, 0, 1, 0), Some(c)) => {
            let LambertLn { coeff: a, scale: b } = ln_terms[0];
            let ab = arena.mul(&[a, b]);
            let c_over_ab = arena.div(c, ab);
            let a_over_c = arena.div(a, c);
            let neg_d = arena.neg(d);
            let neg_d_over_a = arena.div(neg_d, a);
            let exp_part = arena.exp(neg_d_over_a);
            let w_arg = arena.mul(&[c_over_ab, exp_part]);
            lambert_branches(arena, w_arg)
                .into_iter()
                .map(|w| arena.mul(&[a_over_c, w]))
                .collect()
        }
        // ── A*var^var + D = 0 ────────────────────────────────────────
        //   var·ln(var) = ln(-D/A)  →  var = exp(W(ln(-D/A)))
        ((0, 0, 0, 0, 1), None) => {
            let a = pow_terms[0];
            let neg_d = arena.neg(d);
            let r = arena.div(neg_d, a);
            let ln_r = arena.ln(r);
            lambert_branches(arena, ln_r)
                .into_iter()
                .map(|w| arena.exp(w))
                .collect()
        }
        _ => return None,
    };
    Some(
        solutions
            .into_iter()
            .map(|s| Solution {
                value: crate::transforms::eval::eval(arena, s),
            })
            .collect(),
    )
}

/// The Lambert W values `W_k(arg)` a Lambert solution takes: the principal
/// branch, and `W₋₁(arg)` as well when `arg` is certainly in `(−1/e, 0)`,
/// where both branches are real (SymPy's `_lambert` keeps `k = −1` only
/// when `LambertW(arg, -1)` is known to be real; here the interval is
/// decided with certified signs).  At `arg = −1/e` the two coincide, and
/// for `arg ≥ 0` or a symbolic `arg` only `W₀` is taken — the other
/// branches give the complex roots, infinitely many, which `solve` does
/// not enumerate.
fn lambert_branches(arena: &mut Arena, arg: ExprId) -> Vec<ExprId> {
    let arg = crate::transforms::eval::eval(arena, arg);
    let principal = arena.lambertw(arg);
    let mut out = vec![principal];
    if crate::base::walk::free_symbols(arena, arg).is_empty()
        && crate::poly::algebraic::sign_checked(arena, arg) == Some(-1)
    {
        let neg_one = arena.neg_one;
        let inv_e = arena.exp(neg_one);
        let shifted = arena.add(&[arg, inv_e]);
        if crate::poly::algebraic::sign_checked(arena, shifted) == Some(1) {
            out.push(arena.lambertw_branch(arg, neg_one));
        }
    }
    out
}

/// `A·exp(B·var)` (and `A·var·exp(B·var)`): the coefficient `A` and the
/// rate `B`.
#[derive(Clone, Copy)]
struct LambertExp {
    coeff: ExprId,
    rate: ExprId,
}

/// `A·ln(B·var)` (and `A·var·ln(B·var)`): the coefficient `A` and the
/// scale `B`.
#[derive(Clone, Copy)]
struct LambertLn {
    coeff: ExprId,
    scale: ExprId,
}

/// Classification of a single additive term for LambertW analysis.
enum LambertTermClass {
    /// `A * var` — linear in the solve variable.
    Linear(ExprId),
    /// `A * exp(B * var)` — exponential in the solve variable.
    ExpVar(LambertExp),
    /// `A * var * exp(B * var)` — mixed polynomial-exponential.
    VarExpVar(LambertExp),
    /// `A * var * ln(B * var)`.
    VarLnVar(LambertLn),
    /// `A * ln(B * var)`.
    LnVar(LambertLn),
    /// `A * var^var`.
    PowVarVar(ExprId),
}

/// Classify a single additive term (known to contain `var`) into a
/// LambertW-relevant category, or return `None` if unrecognizable.
fn classify_lambert_term(arena: &mut Arena, term: ExprId, var: ExprId) -> Option<LambertTermClass> {
    // `-(inner)`: classify `inner` and negate the coefficient.  This
    // handles cases like `-(x*exp(x))` that remain as Neg nodes rather than
    // being absorbed into a Mul with -1.
    let (term, negate) = match arena.node(term).clone() {
        ExprNode::Neg(inner) => (inner, true),
        _ => (term, false),
    };
    let factors: Vec<ExprId> = match arena.node(term).clone() {
        ExprNode::Mul(children) => children.to_vec(),
        _ => vec![term],
    };
    let mut const_factors: Vec<ExprId> = Vec::new();
    let mut has_var = false;
    let mut exp_inner: Option<ExprId> = None;
    let mut ln_inner: Option<ExprId> = None;
    let mut pow_var_var = false;
    for &child in &factors {
        if !expr_contains_var(arena, child, var) {
            const_factors.push(child);
            continue;
        }
        if child == var {
            if has_var {
                return None; // var appears twice → var²
            }
            has_var = true;
            continue;
        }
        match arena.node(child).clone() {
            ExprNode::Exp(inner) if exp_inner.is_none() => exp_inner = Some(inner),
            ExprNode::Ln(inner) if ln_inner.is_none() => ln_inner = Some(inner),
            ExprNode::Pow(base, exp) if base == var && exp == var && !pow_var_var => {
                pow_var_var = true;
            }
            _ => return None, // unrecognized var-dependent factor
        }
    }
    let mut coeff = match const_factors.len() {
        0 => arena.one,
        1 => const_factors[0],
        _ => arena.mul(&const_factors),
    };
    if negate {
        coeff = arena.neg(coeff);
    }
    let class = match (has_var, exp_inner, ln_inner, pow_var_var) {
        (true, None, None, false) => LambertTermClass::Linear(coeff),
        (has_var, Some(inner), None, false) => {
            let rate = extract_var_coeff_in_product(arena, inner, var)?;
            let e = LambertExp { coeff, rate };
            if has_var {
                LambertTermClass::VarExpVar(e)
            } else {
                LambertTermClass::ExpVar(e)
            }
        }
        (has_var, None, Some(inner), false) => {
            let scale = extract_var_coeff_in_product(arena, inner, var)?;
            let l = LambertLn { coeff, scale };
            if has_var {
                LambertTermClass::VarLnVar(l)
            } else {
                LambertTermClass::LnVar(l)
            }
        }
        (false, None, None, true) => LambertTermClass::PowVarVar(coeff),
        _ => return None,
    };
    Some(class)
}

// ═══════════════════════════════════════════════════════════════════════════
// Symbolic linear solver: a*x + b = 0 where a, b may be symbolic
// ═══════════════════════════════════════════════════════════════════════════

/// Try to solve a linear equation with symbolic coefficients.
///
/// Given `expr = 0`, solve for `var` when `expr` is linear in `var` but
/// the coefficients may be symbolic (not just numbers).
/// For example: `k*x - F = 0` → `x = F/k`.
///
/// Returns `Some(solutions)` if `expr` is linear in `var`, `None` otherwise.
pub(crate) fn try_solve_linear_symbolic(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
) -> Option<Vec<Solution>> {
    // Get the Add children (or treat expr as a single-term sum).
    let terms: Vec<ExprId> = match arena.node(expr).clone() {
        ExprNode::Add(children) => children.to_vec(),
        _ => vec![expr],
    };

    let mut coeff_parts: Vec<ExprId> = Vec::new(); // coefficients of var
    let mut const_parts: Vec<ExprId> = Vec::new(); // terms without var

    for &term in &terms {
        if !expr_contains_var(arena, term, var) {
            // Term doesn't contain var — it's part of the constant.
            const_parts.push(term);
            continue;
        }

        // Term contains var — try to extract a linear coefficient.
        {
            let coeff = extract_var_coeff_in_product(arena, term, var)?;
            coeff_parts.push(coeff);
        }
    }

    if coeff_parts.is_empty() {
        return None;
    }

    // Build total coefficient: sum of all var-coefficients.
    let coeff = if coeff_parts.len() == 1 {
        coeff_parts[0]
    } else {
        arena.add(&coeff_parts)
    };

    // Build constant: sum of all non-var terms.
    let constant = if const_parts.is_empty() {
        arena.zero
    } else if const_parts.len() == 1 {
        const_parts[0]
    } else {
        arena.add(&const_parts)
    };

    // Solution: var = -constant / coeff, as a single cancelled fraction
    // when the coefficients are themselves fractions.
    let neg_const = arena.neg(constant);
    let solution = arena.div(neg_const, coeff);
    let solution = crate::simplify::ratsimp::ratsimp(arena, solution);

    Some(vec![Solution { value: solution }])
}

fn extract_var_coeff_in_product(arena: &mut Arena, expr: ExprId, var: ExprId) -> Option<ExprId> {
    if expr == var {
        return Some(arena.one);
    }
    match arena.node(expr).clone() {
        ExprNode::Mul(children) => {
            let mut const_parts: Vec<ExprId> = Vec::new();
            let mut found_var = false;
            for &child in &children {
                if child == var {
                    if found_var {
                        return None;
                    }
                    found_var = true;
                } else if !expr_contains_var(arena, child, var) {
                    const_parts.push(child);
                } else {
                    return None;
                }
            }
            if !found_var {
                return None;
            }
            match const_parts.len() {
                0 => Some(arena.one),
                1 => Some(const_parts[0]),
                _ => Some(arena.mul(&const_parts)),
            }
        }
        _ => None,
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

    fn solution_strings(a: &Arena, solutions: &[Solution]) -> Vec<String> {
        solutions.iter().map(|s| display(a, s.value)).collect()
    }

    // ── Linear ──────────────────────────────────────────────────────

    #[test]
    fn solve_linear_simple() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x - 3 = 0 → x = 3
        let three = a.int(3);
        let expr = a.sub(x, three);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1);
        assert_eq!(display(&a, solutions[0].value), "3");
    }

    #[test]
    fn solve_linear_with_coefficient() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // 2*x - 6 = 0 → x = 3
        let two = a.int(2);
        let six = a.int(6);
        let two_x = a.mul(&[two, x]);
        let expr = a.sub(two_x, six);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1);
        assert_eq!(display(&a, solutions[0].value), "3");
    }

    #[test]
    fn solve_linear_rational_solution() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // 3*x - 1 = 0 → x = 1/3
        let three = a.int(3);
        let one = a.one;
        let three_x = a.mul(&[three, x]);
        let expr = a.sub(three_x, one);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1);
        assert_eq!(display(&a, solutions[0].value), "1/3");
    }

    #[test]
    fn solve_linear_negative_solution() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x + 5 = 0 → x = -5
        let five = a.int(5);
        let expr = a.add(&[x, five]);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1);
        assert_eq!(display(&a, solutions[0].value), "-5");
    }

    // ── Quadratic ───────────────────────────────────────────────────

    #[test]
    fn solve_quadratic_two_roots() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x^2 - 5x + 6 = 0 → x = 2, x = 3
        let two = a.int(2);
        let five = a.int(5);
        let six = a.int(6);
        let x_sq = a.pow(x, two);
        let five_x = a.mul(&[five, x]);
        let neg_five_x = a.neg(five_x);
        let expr = a.add(&[x_sq, neg_five_x, six]);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 2);
        let vals: Vec<String> = solution_strings(&a, &solutions);
        assert!(
            vals.contains(&"2".to_string()),
            "should have root 2: {vals:?}"
        );
        assert!(
            vals.contains(&"3".to_string()),
            "should have root 3: {vals:?}"
        );
    }

    #[test]
    fn solve_quadratic_double_root() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x^2 - 2x + 1 = 0 → x = 1 (double root)
        let two = a.int(2);
        let one = a.one;
        let x_sq = a.pow(x, two);
        let two_x = a.mul(&[two, x]);
        let neg_two_x = a.neg(two_x);
        let expr = a.add(&[x_sq, neg_two_x, one]);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1);
        assert_eq!(display(&a, solutions[0].value), "1");
    }

    #[test]
    fn solve_quadratic_no_real_roots() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x^2 + 1 = 0 → complex roots ±i
        let two = a.int(2);
        let one = a.one;
        let x_sq = a.pow(x, two);
        let expr = a.add(&[x_sq, one]);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 2, "x²+1=0 should have 2 complex roots");
        let vals: Vec<String> = solution_strings(&a, &solutions);
        assert!(
            vals.iter().all(|v| v.contains("I")),
            "roots should contain I: {vals:?}"
        );
    }

    #[test]
    fn solve_quadratic_irrational_roots() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x^2 - 2 = 0 → x = ±√2
        let two = a.int(2);
        let x_sq = a.pow(x, two);
        let expr = a.sub(x_sq, two);
        let solutions = solve(&mut a, expr, x);
        // Should return symbolic sqrt(2) and -sqrt(2).
        assert_eq!(solutions.len(), 2, "x²-2=0 should have 2 solutions");
        let vals: Vec<String> = solution_strings(&a, &solutions);
        // Check that one contains sqrt and the other is negative.
        let has_sqrt = vals.iter().any(|v| v.contains("sqrt"));
        assert!(has_sqrt, "should contain sqrt(2): {vals:?}");
    }

    #[test]
    fn solve_x_squared_minus_one() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x^2 - 1 = 0 → x = 1, x = -1
        let two = a.int(2);
        let one = a.one;
        let x_sq = a.pow(x, two);
        let expr = a.sub(x_sq, one);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 2);
        let vals: Vec<String> = solution_strings(&a, &solutions);
        assert!(vals.contains(&"1".to_string()));
        assert!(vals.contains(&"-1".to_string()));
    }

    // ── Cubic via rational roots ────────────────────────────────────

    #[test]
    fn solve_cubic_all_rational() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
        let three = a.int(3);
        let six = a.int(6);
        let eleven = a.int(11);
        let two = a.int(2);

        let x3 = a.pow(x, three);
        let x2 = a.pow(x, two);
        let six_x2 = a.mul(&[six, x2]);
        let eleven_x = a.mul(&[eleven, x]);

        let neg_six_x2 = a.neg(six_x2);
        let neg_six = a.neg(six);
        let expr = a.add(&[x3, neg_six_x2, eleven_x, neg_six]);

        let solutions = solve(&mut a, expr, x);
        let vals: Vec<String> = solution_strings(&a, &solutions);
        assert!(
            vals.contains(&"1".to_string()),
            "should have root 1: {vals:?}"
        );
        assert!(
            vals.contains(&"2".to_string()),
            "should have root 2: {vals:?}"
        );
        assert!(
            vals.contains(&"3".to_string()),
            "should have root 3: {vals:?}"
        );
    }

    // ── Edge cases ──────────────────────────────────────────────────

    #[test]
    fn solve_constant_nonzero_no_solutions() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // 5 = 0 → no solutions.
        let five = a.int(5);
        let solutions = solve(&mut a, five, x);
        assert!(solutions.is_empty());
        assert!(matches!(
            solve_classified(&mut a, five, x),
            SolveOutcome::NoSolution(_)
        ));
    }

    #[test]
    fn solve_zero_expression() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // 0 = 0 → infinite solutions (bare `solve` returns empty).
        let zero = a.zero;
        let solutions = solve(&mut a, zero, x);
        assert!(solutions.is_empty());
        assert!(matches!(
            solve_classified(&mut a, zero, x),
            SolveOutcome::Identity
        ));
    }

    #[test]
    fn solve_classified_identity_after_eval() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // sin(0) + 0*x is independent of x and evaluates to 0.
        let zero = a.zero;
        let sin0 = a.sin(zero);
        assert!(matches!(
            solve_classified(&mut a, sin0, x),
            SolveOutcome::Identity
        ));
    }

    #[test]
    fn solve_classified_exp_eq_zero_no_solution() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let e = a.exp(x);
        assert!(matches!(
            solve_classified(&mut a, e, x),
            SolveOutcome::NoSolution(_)
        ));
    }

    #[test]
    fn solve_classified_polynomial_solutions() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let four = a.int(4);
        let expr = a.sub(x2, four);
        match solve_classified(&mut a, expr, x) {
            SolveOutcome::Solutions(s) => assert_eq!(s.len(), 2),
            other => panic!("expected solutions, got {other:?}"),
        }
    }

    // ── General (periodic) solutions ────────────────────────────────────

    #[test]
    fn solve_general_sin_half_has_period_param() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let n = sym(&mut a, "n");
        let half = a.rational(1, 2);
        let sx = a.sin(x);
        let expr = a.sub(sx, half);
        let out = solve_general(&mut a, expr, x, n);
        let sols = out.into_solutions();
        assert_eq!(sols.len(), 2, "two families expected");
        for s in &sols {
            assert!(
                expr_contains_var(&a, s.value, n),
                "family should mention n: {}",
                display(&a, s.value)
            );
            assert!(display(&a, s.value).contains("pi"));
        }
    }

    #[test]
    fn solve_general_tan_has_pi_n() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let n = sym(&mut a, "n");
        let one = a.one;
        let tx = a.tan(x);
        let expr = a.sub(tx, one);
        let sols = solve_general(&mut a, expr, x, n).into_solutions();
        assert_eq!(sols.len(), 1);
        let s = display(&a, sols[0].value);
        assert!(s.contains("n") && s.contains("pi"), "got {s}");
    }

    #[test]
    fn solve_general_polynomial_unchanged() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let n = sym(&mut a, "n");
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let one = a.one;
        let expr = a.sub(x2, one);
        let sols = solve_general(&mut a, expr, x, n).into_solutions();
        assert_eq!(sols.len(), 2);
        for s in &sols {
            assert!(!expr_contains_var(&a, s.value, n));
        }
    }

    #[test]
    fn solve_sin_linear_argument() {
        // sin(2x + 1) = 1/2 requires peeling through an Add node.
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let one = a.one;
        let two_x = a.mul(&[two, x]);
        let arg = a.add(&[two_x, one]);
        let s = a.sin(arg);
        let half = a.rational(1, 2);
        let expr = a.sub(s, half);
        let sols = solve(&mut a, expr, x);
        assert_eq!(sols.len(), 2, "got {:?}", solution_strings(&a, &sols));
    }

    #[test]
    fn solve_symbolic_quadratic_coefficients() {
        // x^2 - k = 0 with symbolic k → ±sqrt(k)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let k = sym(&mut a, "k");
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let expr = a.sub(x2, k);
        let sols = solve(&mut a, expr, x);
        assert_eq!(sols.len(), 2, "got {:?}", solution_strings(&a, &sols));
        for s in &sols {
            assert!(expr_contains_var(&a, s.value, k));
        }
    }

    #[test]
    fn solve_binomial_quintic_roots_of_unity() {
        // x^5 - 2 = 0 → five explicit roots, no RootOf.
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let five = a.int(5);
        let x5 = a.pow(x, five);
        let two = a.int(2);
        let expr = a.sub(x5, two);
        let sols = solve(&mut a, expr, x);
        assert_eq!(sols.len(), 5);
        for s in &sols {
            assert!(!matches!(a.node(s.value), ExprNode::RootOf(..)));
        }
    }

    #[test]
    fn solve_sin_x_eq_zero_via_change_of_variable() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // sin(x) = 0 → change-of-variable with t = sin(x) finds t = 0,
        // then back-substitutes sin(x) = 0 → x ∈ {asin(0), π - asin(0)} = {0, π}.
        let expr = a.sin(x);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(
            solutions.len(),
            2,
            "sin(x)=0 should have 2 solutions (two branches), got {}",
            solutions.len()
        );
        let vals: Vec<String> = solutions.iter().map(|s| display(&a, s.value)).collect();
        // asin(0) is not auto-evaluated, so expect symbolic forms
        assert!(
            vals.iter().any(|v| v == "0" || v.contains("asin(0)")),
            "should have root 0 or asin(0): {vals:?}"
        );
        assert!(
            vals.iter()
                .any(|v| v == "pi" || v.contains("pi") || v.contains("asin")),
            "should have root involving pi or asin: {vals:?}"
        );
    }

    #[test]
    fn solve_with_zero_root() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x^2 - x = x*(x-1) = 0 → x = 0, x = 1
        let two = a.int(2);
        let x_sq = a.pow(x, two);
        let expr = a.sub(x_sq, x);
        let solutions = solve(&mut a, expr, x);
        let vals: Vec<String> = solution_strings(&a, &solutions);
        assert!(
            vals.contains(&"0".to_string()),
            "should have root 0: {vals:?}"
        );
        assert!(
            vals.contains(&"1".to_string()),
            "should have root 1: {vals:?}"
        );
    }

    // ── Verification ────────────────────────────────────────────────

    #[test]
    fn solve_and_verify_quadratic() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x^2 - 5x + 6 = 0
        let two = a.int(2);
        let five = a.int(5);
        let six = a.int(6);
        let x_sq = a.pow(x, two);
        let five_x = a.mul(&[five, x]);
        let neg_five_x = a.neg(five_x);
        let expr = a.add(&[x_sq, neg_five_x, six]);

        let solutions = solve(&mut a, expr, x);
        // Verify each solution by substitution.
        for sol in &solutions {
            let val = crate::transforms::subs::subs(&mut a, expr, x, sol.value);
            assert!(
                a.is_zero_structural(val),
                "substituting x={} should give 0, got {}",
                display(&a, sol.value),
                display(&a, val)
            );
        }
    }

    // ── Transcendental / inversion peeling ──────────────────────────

    #[test]
    fn solve_exp_x_eq_5() {
        // exp(x) - 5 = 0 → x = ln(5)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let five = a.int(5);
        let exp_x = a.exp(x);
        let expr = a.sub(exp_x, five);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1, "exp(x)-5=0 should have 1 solution");
        let val = display(&a, solutions[0].value);
        assert!(
            val.contains("ln") || val.contains("log"),
            "solution should be ln(5): {val}"
        );
    }

    #[test]
    fn solve_ln_x_eq_2() {
        // ln(x) - 2 = 0 → x = exp(2) = e^2
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let ln_x = a.ln(x);
        let expr = a.sub(ln_x, two);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1, "ln(x)-2=0 should have 1 solution");
        let val = display(&a, solutions[0].value);
        assert!(
            val.contains("exp") || val.contains("e") || val.contains("E"),
            "solution should be exp(2): {val}"
        );
    }

    #[test]
    fn solve_sin_x_eq_half() {
        // sin(x) - 1/2 = 0 → x ∈ {asin(1/2), π - asin(1/2)}
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let half = {
            let nid = a.intern_num(Ratio::new(BigInt::from(1), BigInt::from(2)));
            a.intern(ExprNode::Num(nid))
        };
        let sin_x = a.sin(x);
        let expr = a.sub(sin_x, half);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(
            solutions.len(),
            2,
            "sin(x)-1/2=0 should have 2 solutions (two branches)"
        );
        // Solutions are evaluated: asin(1/2) → π/6 and π - asin(1/2) → 5π/6.
        let val0 = display(&a, solutions[0].value);
        let val1 = display(&a, solutions[1].value);
        assert!(
            val0.contains("pi") || val0.contains("asin"),
            "first solution should be pi/6 (or asin(1/2)): {val0}"
        );
        assert!(
            val1.contains("pi") || val1.contains("asin"),
            "second solution should be 5*pi/6 (or pi - asin(1/2)): {val1}"
        );
        assert_ne!(val0, val1);
    }

    #[test]
    fn solve_sqrt_x_eq_3() {
        // sqrt(x) - 3 = 0 → x = 9
        // sqrt(x) is x^(1/2)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let three = a.int(3);
        let sqrt_x = a.sqrt(x);
        let expr = a.sub(sqrt_x, three);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1, "sqrt(x)-3=0 should have 1 solution");
        let val = display(&a, solutions[0].value);
        assert_eq!(val, "9", "solution should be 9: {val}");
    }

    #[test]
    fn solve_mul_factors() {
        // x*(x-1)*(x+2) = 0 → roots 0, 1, -2
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.one;
        let two = a.int(2);
        let x_minus_1 = a.sub(x, one);
        let x_plus_2 = a.add(&[x, two]);
        let expr = a.mul(&[x, x_minus_1, x_plus_2]);
        let solutions = solve(&mut a, expr, x);
        assert!(
            solutions.len() >= 3,
            "x*(x-1)*(x+2)=0 should have 3 roots, got {}",
            solutions.len()
        );
        let vals: Vec<String> = solution_strings(&a, &solutions);
        assert!(
            vals.contains(&"0".to_string()),
            "should have root 0: {vals:?}"
        );
        assert!(
            vals.contains(&"1".to_string()),
            "should have root 1: {vals:?}"
        );
        assert!(
            vals.contains(&"-2".to_string()),
            "should have root -2: {vals:?}"
        );
    }

    #[test]
    fn solve_2_exp_x_minus_6() {
        // 2*exp(x) - 6 = 0 → exp(x) = 3 → x = ln(3)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let six = a.int(6);
        let exp_x = a.exp(x);
        let two_exp_x = a.mul(&[two, exp_x]);
        let expr = a.sub(two_exp_x, six);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1, "2*exp(x)-6=0 should have 1 solution");
        let val = display(&a, solutions[0].value);
        assert!(
            val.contains("ln") || val.contains("log"),
            "solution should be ln(3): {val}"
        );
    }

    #[test]
    fn solve_and_verify_cubic() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x^3 - 6x^2 + 11x - 6 = 0
        let three = a.int(3);
        let six = a.int(6);
        let eleven = a.int(11);
        let two = a.int(2);

        let x3 = a.pow(x, three);
        let x2 = a.pow(x, two);
        let six_x2 = a.mul(&[six, x2]);
        let eleven_x = a.mul(&[eleven, x]);

        let neg_six_x2 = a.neg(six_x2);
        let neg_six = a.neg(six);
        let expr = a.add(&[x3, neg_six_x2, eleven_x, neg_six]);

        let solutions = solve(&mut a, expr, x);
        for sol in &solutions {
            let val = crate::transforms::subs::subs(&mut a, expr, x, sol.value);
            assert!(
                a.is_zero_structural(val),
                "substituting x={} should give 0, got {}",
                display(&a, sol.value),
                display(&a, val)
            );
        }
    }

    // ── Change-of-variable ──────────────────────────────────────────

    #[test]
    fn solve_exp_2x_minus_3_exp_x_plus_2() {
        // exp(2x) - 3*exp(x) + 2 = 0
        // Let t = exp(x): t² - 3t + 2 = (t-1)(t-2) = 0
        // t = 1 → x = ln(1) = 0
        // t = 2 → x = ln(2)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let three = a.int(3);

        let two_x = a.mul(&[two, x]);
        let exp_2x = a.exp(two_x);
        let exp_x = a.exp(x);
        let three_exp_x = a.mul(&[three, exp_x]);
        let neg_three_exp_x = a.neg(three_exp_x);
        let expr = a.add(&[exp_2x, neg_three_exp_x, two]);

        let solutions = solve(&mut a, expr, x);
        assert!(
            solutions.len() >= 2,
            "exp(2x)-3*exp(x)+2=0 should have 2 solutions, got {}: {:?}",
            solutions.len(),
            solution_strings(&a, &solutions)
        );
        let vals: Vec<String> = solution_strings(&a, &solutions);
        // ln(1) may or may not simplify to 0 depending on canonicalization
        assert!(
            vals.contains(&"0".to_string()) || vals.contains(&"ln(1)".to_string()),
            "should have root 0 or ln(1) (from exp(x)=1): {vals:?}"
        );
        let has_ln2 = vals.iter().any(|v| v.contains("ln") || v.contains("log"));
        assert!(has_ln2, "should have root ln(2): {vals:?}");
    }

    #[test]
    fn solve_sin_squared_minus_sin() {
        // sin(x)^2 - sin(x) = 0
        // Let t = sin(x): t² - t = t(t-1) = 0
        // t = 0 → sin(x) = 0 → x = asin(0) = 0
        // t = 1 → sin(x) = 1 → x = asin(1)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);

        let sin_x = a.sin(x);
        let sin_x_sq = a.pow(sin_x, two);
        let expr = a.sub(sin_x_sq, sin_x);

        let solutions = solve(&mut a, expr, x);
        assert!(
            solutions.len() >= 2,
            "sin(x)^2-sin(x)=0 should have ≥2 solutions, got {}: {:?}",
            solutions.len(),
            solution_strings(&a, &solutions)
        );
        let vals: Vec<String> = solution_strings(&a, &solutions);
        // One root should be 0 (from sin(x)=0 → x=asin(0)=0)
        let has_zero = vals.contains(&"0".to_string());
        // The other should involve asin (from sin(x)=1 → x=asin(1))
        let has_asin = vals
            .iter()
            .any(|v| v.contains("asin") || v.contains("arcsin"));
        assert!(
            has_zero || has_asin,
            "should have root 0 or asin(1): {vals:?}"
        );
    }

    #[test]
    fn solve_exp_quadratic_one_valid_root() {
        // exp(2x) - 5*exp(x) + 6 = 0
        // Let t = exp(x): t² - 5t + 6 = (t-2)(t-3) = 0
        // t = 2 → x = ln(2)
        // t = 3 → x = ln(3)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let five = a.int(5);
        let six = a.int(6);

        let two_x = a.mul(&[two, x]);
        let exp_2x = a.exp(two_x);
        let exp_x = a.exp(x);
        let five_exp_x = a.mul(&[five, exp_x]);
        let neg_five_exp_x = a.neg(five_exp_x);
        let expr = a.add(&[exp_2x, neg_five_exp_x, six]);

        let solutions = solve(&mut a, expr, x);
        assert!(
            solutions.len() >= 2,
            "exp(2x)-5*exp(x)+6=0 should have 2 solutions, got {}: {:?}",
            solutions.len(),
            solution_strings(&a, &solutions)
        );
        let vals: Vec<String> = solution_strings(&a, &solutions);
        let has_ln = vals.iter().all(|v| v.contains("ln") || v.contains("log"));
        assert!(has_ln, "all roots should involve ln: {vals:?}");
    }

    // ── Helper tests ────────────────────────────────────────────────

    #[test]
    fn rational_sqrt_perfect() {
        let r = Ratio::new(BigInt::from(9), BigInt::from(4));
        let s = rational_sqrt(&r).unwrap();
        assert_eq!(s, Ratio::new(BigInt::from(3), BigInt::from(2)));
    }

    #[test]
    fn rational_sqrt_not_perfect() {
        let r = Ratio::from_integer(BigInt::from(2));
        assert!(rational_sqrt(&r).is_none());
    }

    #[test]
    fn rational_sqrt_zero() {
        let r = Ratio::zero();
        let s = rational_sqrt(&r).unwrap();
        assert!(s.is_zero());
    }

    #[test]
    fn divisors_of_12() {
        let d = divisors(&BigInt::from(12));
        assert_eq!(
            d,
            vec![1, 2, 3, 4, 6, 12]
                .into_iter()
                .map(BigInt::from)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn divisors_of_1() {
        let d = divisors(&BigInt::from(1));
        assert_eq!(d, vec![BigInt::from(1)]);
    }

    #[test]
    fn clear_denominators_works() {
        let poly = Poly::from_coeffs(vec![
            Ratio::new(BigInt::from(1), BigInt::from(2)),
            Ratio::new(BigInt::from(1), BigInt::from(3)),
        ]);
        let (int_poly, lcm) = clear_denominators(&poly);
        assert_eq!(lcm, BigInt::from(6));
        // 1/2 * 6 = 3, 1/3 * 6 = 2
        assert_eq!(int_poly.coeff(0), Ratio::from_integer(BigInt::from(3)));
        assert_eq!(int_poly.coeff(1), Ratio::from_integer(BigInt::from(2)));
    }

    // ── Complex quadratic roots ─────────────────────────────────────

    #[test]
    fn solve_x2_plus_1() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x² + 1 = 0 → roots ±i
        let two = a.int(2);
        let one = a.one;
        let x_sq = a.pow(x, two);
        let expr = a.add(&[x_sq, one]);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 2, "x²+1=0 should have 2 complex roots");
        let vals: Vec<String> = solution_strings(&a, &solutions);
        assert!(
            vals.iter().all(|v| v.contains("I")),
            "roots should contain I: {vals:?}"
        );
    }

    #[test]
    fn solve_x2_plus_2x_plus_5() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x² + 2x + 5 = 0 → roots -1 ± 2i
        let two = a.int(2);
        let five = a.int(5);
        let x_sq = a.pow(x, two);
        let two_x = a.mul(&[two, x]);
        let expr = a.add(&[x_sq, two_x, five]);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 2, "x²+2x+5=0 should have 2 complex roots");
        let vals: Vec<String> = solution_strings(&a, &solutions);
        // roots are -1 ± 2i
        for v in &vals {
            assert!(v.contains("I"), "root should contain I: {v}");
        }
    }

    #[test]
    fn solve_x2_plus_4() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x² + 4 = 0 → roots ±2i
        let two = a.int(2);
        let four = a.int(4);
        let x_sq = a.pow(x, two);
        let expr = a.add(&[x_sq, four]);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 2, "x²+4=0 should have 2 complex roots");
        let vals: Vec<String> = solution_strings(&a, &solutions);
        assert!(
            vals.iter().all(|v| v.contains("I")),
            "roots should contain I: {vals:?}"
        );
    }

    #[test]
    fn solve_even_power_peeling_both_roots() {
        // (2x + 1)^2 = 9  →  2x + 1 = ±3  →  x = 1 or x = -2
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.int(1);
        let two = a.int(2);
        let nine = a.int(9);
        let two_x = a.mul(&[two, x]);
        let inner = a.add(&[two_x, one]); // 2x + 1
        let squared = a.pow(inner, two); // (2x + 1)^2
        let neg_nine = a.neg(nine);
        let expr = a.add(&[squared, neg_nine]); // (2x + 1)^2 - 9
        let solutions = solve(&mut a, expr, x);
        let mut vals: Vec<String> = solution_strings(&a, &solutions);
        vals.sort();
        assert_eq!(vals.len(), 2, "expected 2 solutions, got {vals:?}");
        assert_eq!(vals, vec!["-2", "1"], "solutions: {vals:?}");
    }

    // ── LambertW solver ─────────────────────────────────────────────

    #[test]
    fn solve_lambert_x_exp_x_eq_1() {
        // x·exp(x) - 1 = 0  →  x = W(1)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.one;
        let exp_x = a.exp(x);
        let x_exp_x = a.mul(&[x, exp_x]);
        let expr = a.sub(x_exp_x, one);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1, "x·exp(x)=1 should have 1 solution");
        let val = display(&a, solutions[0].value);
        assert!(val.contains("W("), "solution should be W(1): {val}");
    }

    #[test]
    fn solve_lambert_x_exp_x_eq_5() {
        // x·exp(x) - 5 = 0  →  x = W(5)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let five = a.int(5);
        let exp_x = a.exp(x);
        let x_exp_x = a.mul(&[x, exp_x]);
        let expr = a.sub(x_exp_x, five);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1, "x·exp(x)=5 should have 1 solution");
        let val = display(&a, solutions[0].value);
        assert!(val.contains("W("), "solution should be W(5): {val}");
    }

    #[test]
    fn solve_lambert_x_exp_x_eq_0() {
        // x·exp(x) = 0  →  factored as Mul([x, Exp(x)]);
        // the Mul pre-check solves x=0 from the x factor.
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let exp_x = a.exp(x);
        let expr = a.mul(&[x, exp_x]);
        let solutions = solve(&mut a, expr, x);
        assert!(!solutions.is_empty(), "x·exp(x)=0 should have a solution");
        let vals: Vec<String> = solution_strings(&a, &solutions);
        assert!(
            vals.contains(&"0".to_string()),
            "should have root 0: {vals:?}"
        );
    }

    #[test]
    fn solve_lambert_2x_exp_x_eq_4() {
        // 2·x·exp(x) - 4 = 0  →  x·exp(x) = 2  →  x = W(2)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let four = a.int(4);
        let exp_x = a.exp(x);
        let two_x_exp_x = a.mul(&[two, x, exp_x]);
        let expr = a.sub(two_x_exp_x, four);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1, "2·x·exp(x)=4 should have 1 solution");
        let val = display(&a, solutions[0].value);
        assert!(val.contains("W("), "solution should involve W: {val}");
    }

    #[test]
    fn solve_lambert_x_exp_2x_eq_3() {
        // x·exp(2·x) - 3 = 0
        // Multiply by 2: 2·x·exp(2·x) = 6 → 2·x = W(6) → x = W(6)/2
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let three = a.int(3);
        let two_x = a.mul(&[two, x]);
        let exp_2x = a.exp(two_x);
        let x_exp_2x = a.mul(&[x, exp_2x]);
        let expr = a.sub(x_exp_2x, three);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1, "x·exp(2x)=3 should have 1 solution");
        let val = display(&a, solutions[0].value);
        assert!(val.contains("W("), "solution should involve W: {val}");
    }

    #[test]
    fn solve_lambert_exp_x_plus_x_eq_2() {
        // exp(x) + x - 2 = 0  →  Pattern 2 (A=1, B=1, C=1, D=-2)
        // Solution: x = 2 - W(exp(2))
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let exp_x = a.exp(x);
        let sum = a.add(&[exp_x, x]);
        let expr = a.sub(sum, two);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1, "exp(x)+x-2=0 should have 1 solution");
        let val = display(&a, solutions[0].value);
        assert!(val.contains("W("), "solution should involve W: {val}");
    }

    #[test]
    fn solve_lambert_neg_exp_x_minus_x_plus_2() {
        // -exp(x) - x + 2 = 0  is the same equation as exp(x) + x - 2 = 0
        // Pattern 2 with A=-1, B=1, C=-1, D=2
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let exp_x = a.exp(x);
        let neg_exp_x = a.neg(exp_x);
        let neg_x = a.neg(x);
        let expr = a.add(&[neg_exp_x, neg_x, two]);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1, "-exp(x)-x+2=0 should have 1 solution");
        let val = display(&a, solutions[0].value);
        assert!(val.contains("W("), "solution should involve W: {val}");
    }

    #[test]
    fn solve_lambert_x_squared_exp_x_not_lambert() {
        // x²·exp(x) - 1 = 0: NOT a LambertW pattern (x² instead of x).
        // Solver should return empty gracefully.
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.one;
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let exp_x = a.exp(x);
        let x2_exp_x = a.mul(&[x2, exp_x]);
        let expr = a.sub(x2_exp_x, one);
        let solutions = solve(&mut a, expr, x);
        // Should not panic; may return empty since it's not a recognized pattern
        assert!(
            solutions.is_empty(),
            "x²·exp(x)=1 is not a simple LambertW pattern; got {} solutions",
            solutions.len()
        );
    }

    #[test]
    fn solve_lambert_x_exp_x_eq_e_gives_1() {
        // x·exp(x) - e = 0  →  x = W(e) = 1
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let e = a.e_const;
        let exp_x = a.exp(x);
        let x_exp_x = a.mul(&[x, exp_x]);
        let expr = a.sub(x_exp_x, e);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1, "x·exp(x)=e should have 1 solution");
        let val = display(&a, solutions[0].value);
        assert_eq!(val, "1", "W(e) should evaluate to 1: {val}");
    }

    #[test]
    fn solve_lambert_preserves_existing_polynomial() {
        // x^2 - 1 = 0 should still be solved by the polynomial path, not LambertW
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.one;
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let expr = a.sub(x2, one);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 2, "x²-1 should still give 2 roots");
    }

    #[test]
    fn solve_lambert_preserves_existing_transcendental() {
        // exp(x) - 5 = 0 should still be solved by inversion peeling
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let five = a.int(5);
        let exp_x = a.exp(x);
        let expr = a.sub(exp_x, five);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1, "exp(x)-5=0 should still give 1 root");
        let val = display(&a, solutions[0].value);
        assert!(
            val.contains("ln"),
            "solution should be ln(5), not lambertw: {val}"
        );
    }

    // ── Symbolic linear ─────────────────────────────────────────────

    #[test]
    fn solve_symbolic_linear_kx_minus_f() {
        let mut a = Arena::new();
        let k = sym(&mut a, "k");
        let x = sym(&mut a, "x");
        let f = sym(&mut a, "F");
        // k*x - F = 0, solve for x → x = F/k
        let kx = a.mul(&[k, x]);
        let expr = a.sub(kx, f);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1);
        let s = display(&a, solutions[0].value);
        assert!(s.contains('F') && s.contains('k'), "Expected F/k, got: {s}");
    }

    #[test]
    fn solve_symbolic_linear_bare_var() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let c = sym(&mut a, "c");
        // x + c = 0, solve for x → x = -c
        let expr = a.add(&[x, c]);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1);
        let s = display(&a, solutions[0].value);
        // Should be -c or (-1)*c or similar
        assert!(s.contains('c'), "Expected -c, got: {s}");
    }

    #[test]
    fn solve_symbolic_linear_multiple_terms() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let p = sym(&mut a, "a");
        let b = sym(&mut a, "b");
        let c = sym(&mut a, "c");
        // a*x + b*x + c = 0 → x = -c/(a+b)
        let ax = a.mul(&[p, x]);
        let bx = a.mul(&[b, x]);
        let expr = a.add(&[ax, bx, c]);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1);
        let s = display(&a, solutions[0].value);
        assert!(s.contains('c'), "Expected -c/(a+b), got: {s}");
    }

    // ── Bug 21 regression: exp/range domain checks ──────────────────

    #[test]
    fn solve_exp_x_eq_zero_no_solution() {
        // exp(x) = 0 has no real solution (exp(x) > 0 for all real x).
        // Regression: previously returned [ln(0)] instead of [].
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let exp_x = a.exp(x);
        let solutions = solve(&mut a, exp_x, x);
        assert!(
            solutions.is_empty(),
            "exp(x)=0 should have no solutions, got: {:?}",
            solution_strings(&a, &solutions)
        );
    }

    #[test]
    fn solve_exp_x_eq_negative_no_solution() {
        // exp(x) + 3 = 0  →  exp(x) = -3, no real solution.
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let three = a.int(3);
        let exp_x = a.exp(x);
        let expr = a.add(&[exp_x, three]);
        let solutions = solve(&mut a, expr, x);
        assert!(
            solutions.is_empty(),
            "exp(x)=-3 should have no solutions, got: {:?}",
            solution_strings(&a, &solutions)
        );
    }

    #[test]
    fn solve_sin_x_eq_2_no_solution() {
        // sin(x) = 2 has no real solution (sin range is [-1, 1]).
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let sin_x = a.sin(x);
        let expr = a.sub(sin_x, two);
        let solutions = solve(&mut a, expr, x);
        assert!(
            solutions.is_empty(),
            "sin(x)=2 should have no solutions, got: {:?}",
            solution_strings(&a, &solutions)
        );
    }

    #[test]
    fn solve_ln_x_eq_zero() {
        // ln(x) = 0 → x = exp(0) = 1.
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let ln_x = a.ln(x);
        let solutions = solve(&mut a, ln_x, x);
        assert_eq!(solutions.len(), 1, "ln(x)=0 should have 1 solution");
        let val = display(&a, solutions[0].value);
        assert!(
            val == "1" || val.contains("exp(0)"),
            "solution should be 1 or exp(0), got: {val}"
        );
    }

    #[test]
    fn solve_abs_x_plus_1_no_solution() {
        // |x| + 1 = 0 → |x| = -1, impossible since |x| >= 0.
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.one;
        let abs_x = a.abs(x);
        let expr = a.add(&[abs_x, one]);
        let solutions = solve(&mut a, expr, x);
        assert!(
            solutions.is_empty(),
            "|x|+1=0 should have no solutions, got: {:?}",
            solution_strings(&a, &solutions)
        );
    }
}

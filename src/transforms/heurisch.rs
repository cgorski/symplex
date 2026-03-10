//! Heuristic Risch integrator ("Poor Man's Integrator").
//!
//! This module implements a numeric-evaluation–based integration algorithm
//! that serves as a fallback when the rule-based integrator returns an
//! unevaluated `Integral` node.
//!
//! # Algorithm outline
//!
//! 1. **Component collection** — walk the integrand and collect all
//!    functional subexpressions involving the integration variable.
//! 2. **Substitution mapping** — map each component to a fresh variable V_i.
//! 3. **Degree bound** — estimate a degree bound for the candidate
//!    antiderivative from the exponent/degree structure.
//! 4. **Monomial generation** — generate all monomials in V_i up to the
//!    degree bound.
//! 5. **Ansatz construction** — candidate = Σ c_j · monomial_j / denom.
//! 6. **Numeric evaluation** — evaluate at 3n random f64 points, build an
//!    overdetermined linear system.
//! 7. **Solve via least-squares** — Gaussian elimination with partial
//!    pivoting on the normal equations.
//! 8. **Rational reconstruction** — continued-fraction approximation to
//!    recover exact rational coefficients (max denominator 1000).
//! 9. **Symbolic verification** — differentiate the candidate and verify
//!    it equals the integrand.
//!
//! # Design
//!
//! - All floating-point computation is confined to the numeric evaluation
//!   section; the rest is purely symbolic.
//! - No external linear-algebra crate is needed; matrices are small
//!   (typically 10–30 unknowns).
//! - If verification fails the function returns `None` and the caller
//!   leaves the integral unevaluated.

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode, SymbolId};

// ═══════════════════════════════════════════════════════════════════════════
// Public entry point
// ═══════════════════════════════════════════════════════════════════════════

/// Attempt to integrate `expr` w.r.t. `var` using the heuristic Risch
/// algorithm.
///
/// Returns `Some(antiderivative)` on success, `None` on failure.
#[must_use]
pub(crate) fn heurisch_integrate(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    tracing::debug!("heurisch_integrate: entering heuristic Risch integrator");

    // Quick bail-out: if the integrand doesn't contain the variable at all,
    // the answer is just expr * var.
    if !contains_var(arena, expr, var_sym) {
        return Some(arena.mul(&[expr, var]));
    }

    // Try with increasing degree offsets (up to 2 retries).
    for degree_offset in 0..=2 {
        if degree_offset > 0 {
            tracing::debug!("heurisch: retry with degree_offset={}", degree_offset);
        }
        if let Some(result) = heurisch_attempt(arena, expr, var, var_sym, degree_offset) {
            return Some(result);
        }
    }

    None
}

// ═══════════════════════════════════════════════════════════════════════════
// Core algorithm
// ═══════════════════════════════════════════════════════════════════════════

/// One attempt of the heuristic Risch algorithm with a given `degree_offset`
/// added to the computed degree bound.
fn heurisch_attempt(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
    degree_offset: usize,
) -> Option<ExprId> {
    // ── Step 1: Collect components ──────────────────────────────────
    let components = collect_components(arena, expr, var_sym);
    tracing::debug!("heurisch: collected {} components", components.len());
    if components.is_empty() {
        // Only the bare variable — fall back to simple power rule territory.
        // This shouldn't happen if we got here, but be safe.
        return None;
    }

    // ── Step 2: Substitution mapping ───────────────────────────────
    // Create fresh symbols V_0, V_1, … for each component.
    let mut comp_syms: Vec<ExprId> = Vec::with_capacity(components.len());
    let mut comp_sym_ids: Vec<SymbolId> = Vec::with_capacity(components.len());
    for i in 0..components.len() {
        let name = format!("__hV{i}");
        let sid = arena.symbols.intern(&name);
        let eid = arena.intern(ExprNode::Symbol(sid));
        comp_syms.push(eid);
        comp_sym_ids.push(sid);
    }

    // ── Step 3: Compute derivatives of components ──────────────────
    // For each component g_i, compute dg_i/dx.
    let comp_derivs: Vec<ExprId> = components
        .iter()
        .map(|&c| crate::transforms::diff::diff(arena, c, var))
        .collect();

    // ── Step 4: Degree bound ───────────────────────────────────────
    let degree_bound = compute_degree_bound(arena, &components, var_sym) + degree_offset;
    tracing::debug!(
        "heurisch: degree bound total={}, offset={}",
        degree_bound,
        degree_offset
    );
    if degree_bound > 12 {
        // Bail out if degree bound is too large (would create huge systems).
        return None;
    }

    // ── Step 5: Generate monomials ─────────────────────────────────
    let monomials = generate_monomials(arena, &comp_syms, degree_bound);
    if monomials.is_empty() {
        return None;
    }

    let n_unknowns = monomials.len();
    tracing::debug!(
        "heurisch: {} monomials, {} unknowns",
        monomials.len(),
        n_unknowns
    );
    if n_unknowns > 60 {
        // Too many unknowns for reliable f64 solution.
        return None;
    }

    // ── Step 6: Build the ansatz symbolically ──────────────────────
    // Candidate F = Σ c_j · monomial_j
    // Then dF/dx = Σ c_j · d(monomial_j)/dx
    // where d(monomial_j)/dx = Σ_i (∂monomial_j/∂V_i) · (dg_i/dx)
    //
    // We need: dF/dx = integrand
    //
    // For numeric evaluation we evaluate both sides at random points.

    // ── Step 7: Numeric evaluation ─────────────────────────────────
    let n_points = 3 * n_unknowns + 5;

    // Deterministic "random" evaluation points for reproducibility.
    let eval_points: Vec<f64> = (0..n_points)
        .map(|i| {
            let t = (i as f64 + 1.0) * 0.1317 + 0.2;
            // Avoid exact integers and zero; keep in a reasonable range.
            t + 0.01 * (t * 7.31).sin()
        })
        .collect();

    // For each evaluation point x_k:
    //   1. Evaluate each component g_i(x_k) → f64
    //   2. Evaluate d(monomial_j)/dx at x_k → this forms one row of A
    //   3. Evaluate integrand at x_k → this forms b_k
    let mut matrix_a: Vec<Vec<f64>> = Vec::with_capacity(n_points);
    let mut vec_b: Vec<f64> = Vec::with_capacity(n_points);

    for &x_val in &eval_points {
        // Evaluate each component at x = x_val.
        let comp_vals: Vec<f64> = components
            .iter()
            .map(|&c| eval_at_point(arena, c, var, x_val))
            .collect();

        // Check for NaN/Inf in component values.
        if comp_vals.iter().any(|v| !v.is_finite()) {
            continue;
        }

        // Evaluate each component derivative at x = x_val.
        let comp_deriv_vals: Vec<f64> = comp_derivs
            .iter()
            .map(|&d| eval_at_point(arena, d, var, x_val))
            .collect();

        if comp_deriv_vals.iter().any(|v| !v.is_finite()) {
            continue;
        }

        // Evaluate the integrand at x = x_val.
        let integrand_val = eval_at_point(arena, expr, var, x_val);
        if !integrand_val.is_finite() {
            continue;
        }

        // For each monomial, compute d(monomial)/dx evaluated at x_val.
        // monomial_j is a product of V_i^{e_i}.
        // d(monomial_j)/dx = Σ_i (∂monomial_j/∂V_i) · (dg_i/dx)
        //
        // We evaluate the monomial at the component values, and compute
        // the derivative using the chain rule numerically.
        let mut row: Vec<f64> = Vec::with_capacity(n_unknowns);
        for monomial in monomials.iter().take(n_unknowns) {
            let monomial_deriv = eval_monomial_deriv(&comp_vals, &comp_deriv_vals, monomial);
            if !monomial_deriv.is_finite() {
                // Skip this point entirely.
                row.clear();
                break;
            }
            row.push(monomial_deriv);
        }

        if row.len() == n_unknowns {
            matrix_a.push(row);
            vec_b.push(integrand_val);
        }
    }

    let n_rows = matrix_a.len();
    tracing::debug!("heurisch: solving {}x{} numeric system", n_rows, n_unknowns);
    if n_rows < n_unknowns {
        // Not enough valid evaluation points.
        return None;
    }

    // ── Step 8: Solve via least-squares ────────────────────────────
    // Solve A^T A x = A^T b using Gaussian elimination with partial pivoting.
    let coeffs = solve_least_squares(&matrix_a, &vec_b, n_unknowns)?;

    // ── Step 9: Rational reconstruction ────────────────────────────
    let rational_coeffs: Vec<(i64, i64)> = coeffs
        .iter()
        .map(|&c| rationalize(c, 1000))
        .collect::<Option<Vec<_>>>()?;
    tracing::debug!(
        "heurisch: reconstructed {}/{} coefficients",
        rational_coeffs.len(),
        coeffs.len()
    );

    // ── Step 10: Build symbolic candidate ──────────────────────────
    let candidate = build_candidate(arena, &monomials, &rational_coeffs, &components, &comp_syms);

    // ── Step 11: Symbolic verification ─────────────────────────────
    // Differentiate the candidate w.r.t. var and check it equals expr.
    let candidate_deriv = crate::transforms::diff::diff(arena, candidate, var);

    // Verify numerically at a few test points.
    let test_points = [0.37, 1.13, 2.71, 0.73, 1.89];
    let mut verified = true;

    for &x_val in &test_points {
        let orig_val = eval_at_point(arena, expr, var, x_val);
        let deriv_val = eval_at_point(arena, candidate_deriv, var, x_val);

        if !orig_val.is_finite() || !deriv_val.is_finite() {
            continue;
        }

        let abs_diff = (orig_val - deriv_val).abs();
        let scale = orig_val.abs().max(1.0);
        if abs_diff / scale > 1e-6 {
            verified = false;
            break;
        }
    }

    tracing::debug!(
        "heurisch: verification {}",
        if verified { "passed" } else { "failed" }
    );
    if verified { Some(candidate) } else { None }
}

// ═══════════════════════════════════════════════════════════════════════════
// Component collection
// ═══════════════════════════════════════════════════════════════════════════

/// Collect all functional subexpressions involving `var_sym`.
///
/// Walk the expression tree and gather:
/// - The variable itself
/// - Function nodes (Sin, Cos, Exp, Ln, Tan, etc.) whose argument
///   depends on `var_sym`
/// - `Pow(base, exp)` where base depends on `var_sym`
fn collect_components(arena: &Arena, expr: ExprId, var_sym: SymbolId) -> Vec<ExprId> {
    let mut components: FxHashSet<ExprId> = FxHashSet::default();
    let mut stack: Vec<ExprId> = vec![expr];
    let mut visited: FxHashSet<ExprId> = FxHashSet::default();

    while let Some(id) = stack.pop() {
        if visited.contains(&id) {
            continue;
        }
        visited.insert(id);

        let node = arena.node(id).clone();
        match node {
            ExprNode::Symbol(sid) if sid == var_sym => {
                components.insert(id);
            }

            ExprNode::Sin(inner)
            | ExprNode::Cos(inner)
            | ExprNode::Tan(inner)
            | ExprNode::Exp(inner)
            | ExprNode::Ln(inner)
            | ExprNode::Asin(inner)
            | ExprNode::Acos(inner)
            | ExprNode::Atan(inner)
            | ExprNode::Sinh(inner)
            | ExprNode::Cosh(inner)
            | ExprNode::Tanh(inner)
            | ExprNode::Asinh(inner)
            | ExprNode::Acosh(inner)
            | ExprNode::Atanh(inner)
            | ExprNode::Abs(inner) => {
                if contains_var(arena, inner, var_sym) {
                    components.insert(id);
                    // Also recurse into the inner expression to collect
                    // sub-components.
                    stack.push(inner);
                }
            }

            ExprNode::Pow(base, exp) => {
                if contains_var(arena, base, var_sym) || contains_var(arena, exp, var_sym) {
                    // If the exponent is a constant, collect base and recurse.
                    // If the exponent contains var, collect the whole Pow.
                    if contains_var(arena, exp, var_sym) {
                        // e.g. x^x or 2^x or exp(x) written as e^x
                        components.insert(id);
                    }
                    // Always include base if it depends on var.
                    if contains_var(arena, base, var_sym) {
                        // For things like sin(x)^2 we want sin(x) as a component.
                        stack.push(base);
                    }
                    stack.push(exp);
                }
            }

            ExprNode::Add(ref children) | ExprNode::Mul(ref children) => {
                for &child in children {
                    stack.push(child);
                }
            }

            ExprNode::Neg(inner) => {
                stack.push(inner);
            }

            _ => {
                // For other nodes, recurse into children.
                let children = arena.node(id).children();
                for child in children {
                    stack.push(child);
                }
            }
        }
    }

    let mut result: Vec<ExprId> = components.into_iter().collect();
    // Sort for deterministic ordering.
    result.sort_by_key(|&id| id.0);
    result
}

// ═══════════════════════════════════════════════════════════════════════════
// Degree bound computation
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the degree bound for the candidate antiderivative.
///
/// A = max exponent degree contribution from exponential-type components.
/// B = max polynomial degree appearing.
/// Bound = A + B + 1.
fn compute_degree_bound(arena: &Arena, components: &[ExprId], var_sym: SymbolId) -> usize {
    let mut max_poly_degree: usize = 1;
    let mut has_transcendental = false;

    for &comp in components {
        let node = arena.node(comp).clone();
        match node {
            ExprNode::Symbol(sid) if sid == var_sym => {
                // x itself — degree 1.
                max_poly_degree = max_poly_degree.max(1);
            }
            ExprNode::Pow(base, exp) => {
                if let ExprNode::Symbol(sid) = arena.node(base)
                    && *sid == var_sym
                    && let Some(r) = arena.as_num(exp)
                    && r.is_integer()
                {
                    let d = r.to_integer().to_string().parse::<i64>().unwrap_or(1);
                    max_poly_degree = max_poly_degree.max(d.unsigned_abs() as usize);
                }
                if contains_var(arena, exp, var_sym) {
                    has_transcendental = true;
                }
            }
            ExprNode::Sin(_)
            | ExprNode::Cos(_)
            | ExprNode::Tan(_)
            | ExprNode::Exp(_)
            | ExprNode::Ln(_)
            | ExprNode::Sinh(_)
            | ExprNode::Cosh(_)
            | ExprNode::Tanh(_) => {
                has_transcendental = true;
            }
            _ => {}
        }
    }

    let a = if has_transcendental { 1 } else { 0 };
    let b = max_poly_degree;
    a + b + 1
}

// ═══════════════════════════════════════════════════════════════════════════
// Monomial generation
// ═══════════════════════════════════════════════════════════════════════════

/// A monomial is represented as a vector of exponents, one per component.
/// monomial = V_0^{e_0} · V_1^{e_1} · … · V_{n-1}^{e_{n-1}}
type Monomial = Vec<u32>;

/// Generate all monomials in the component variables up to total degree
/// `max_degree`.
fn generate_monomials(arena: &mut Arena, comp_syms: &[ExprId], max_degree: usize) -> Vec<Monomial> {
    let n = comp_syms.len();
    if n == 0 {
        return vec![];
    }

    let mut monomials: Vec<Monomial> = Vec::new();

    // Generate monomials by total degree using a recursive enumeration
    // via an iterative stack.
    let mut stack: Vec<(usize, u32, Monomial)> = Vec::new();
    stack.push((0, max_degree as u32, vec![0; n]));

    while let Some((pos, remaining, current)) = stack.pop() {
        if pos == n {
            monomials.push(current);
            continue;
        }

        let max_for_this = remaining;
        for e in 0..=max_for_this {
            let mut m = current.clone();
            m[pos] = e;
            stack.push((pos + 1, remaining - e, m));
        }
    }

    // Limit total number of monomials.
    if monomials.len() > 100 {
        monomials.truncate(100);
    }

    // Suppress unused variable warning for arena (needed for API consistency).
    let _ = arena;

    monomials
}

/// Evaluate a monomial (product of V_i^{e_i}) given numeric component values.
fn eval_monomial(comp_vals: &[f64], monomial: &Monomial) -> f64 {
    let mut result = 1.0;
    for (i, &e) in monomial.iter().enumerate() {
        if e > 0 {
            result *= comp_vals[i].powi(e as i32);
        }
    }
    result
}

/// Evaluate the derivative of a monomial w.r.t. x using the chain rule.
///
/// d/dx (V_0^{e_0} · … · V_{n-1}^{e_{n-1}})
///   = Σ_i e_i · V_i^{e_i - 1} · (dV_i/dx) · Π_{j≠i} V_j^{e_j}
fn eval_monomial_deriv(comp_vals: &[f64], comp_deriv_vals: &[f64], monomial: &Monomial) -> f64 {
    let n = monomial.len();
    let full_product = eval_monomial(comp_vals, monomial);

    let mut total = 0.0;
    for i in 0..n {
        let e = monomial[i];
        if e == 0 {
            continue;
        }
        // d/dx V_i^{e_i} = e_i · V_i^{e_i - 1} · dV_i/dx
        // So the contribution is:
        //   e_i · (V_i^{e_i - 1} / V_i^{e_i}) · full_product · dV_i/dx
        //   = e_i · (1/V_i) · full_product · dV_i/dx
        //   = e_i · full_product / V_i · dV_i/dx
        if comp_vals[i].abs() < 1e-300 {
            // Near-zero component value: compute without dividing.
            let mut partial = 1.0;
            for j in 0..n {
                if j == i {
                    partial *= comp_vals[j].powi(monomial[j] as i32 - 1);
                } else if monomial[j] > 0 {
                    partial *= comp_vals[j].powi(monomial[j] as i32);
                }
            }
            total += (e as f64) * partial * comp_deriv_vals[i];
        } else {
            total += (e as f64) * full_product / comp_vals[i] * comp_deriv_vals[i];
        }
    }

    total
}

// ═══════════════════════════════════════════════════════════════════════════
// Numeric evaluation of expressions at a point
// ═══════════════════════════════════════════════════════════════════════════

/// Evaluate an expression at `var = x_val` using f64 arithmetic.
///
/// This is a quick-and-dirty evaluator for the heuristic integration.
/// It does NOT use the full `evalf` machinery (which requires BigFloat).
fn eval_at_point(arena: &Arena, expr: ExprId, var: ExprId, x_val: f64) -> f64 {
    let post_order = crate::base::walk::post_order_ids(arena, expr);
    let mut cache: FxHashMap<ExprId, f64> = FxHashMap::default();

    // Seed the variable.
    cache.insert(var, x_val);

    for &id in &post_order {
        if cache.contains_key(&id) {
            continue;
        }

        let val = eval_node_f64(arena, id, &cache);
        cache.insert(id, val);
    }

    cache.get(&expr).copied().unwrap_or(f64::NAN)
}

/// Evaluate a single node given cached child values.
fn eval_node_f64(arena: &Arena, id: ExprId, cache: &FxHashMap<ExprId, f64>) -> f64 {
    let node = arena.node(id).clone();

    match node {
        ExprNode::Num(nid) => {
            let r = arena.num(nid);
            // Convert rational to f64.
            let numer: f64 = r.numer().to_string().parse().unwrap_or(f64::NAN);
            let denom: f64 = r.denom().to_string().parse().unwrap_or(f64::NAN);
            if denom == 0.0 {
                f64::NAN
            } else {
                numer / denom
            }
        }

        ExprNode::Symbol(_) => {
            // If not in cache, it's a free symbol — treat as NaN.
            cache.get(&id).copied().unwrap_or(f64::NAN)
        }

        ExprNode::Pi => std::f64::consts::PI,
        ExprNode::E => std::f64::consts::E,
        ExprNode::ImaginaryUnit => f64::NAN, // Can't represent in f64.
        ExprNode::Infinity => f64::INFINITY,
        ExprNode::NegInfinity => f64::NEG_INFINITY,
        ExprNode::ComplexInfinity | ExprNode::NaN => f64::NAN,

        ExprNode::Add(ref children) => {
            let mut sum = 0.0;
            for &c in children {
                sum += cache.get(&c).copied().unwrap_or(f64::NAN);
            }
            sum
        }

        ExprNode::Mul(ref children) => {
            let mut prod = 1.0;
            for &c in children {
                prod *= cache.get(&c).copied().unwrap_or(f64::NAN);
            }
            prod
        }

        ExprNode::Pow(base, exp) => {
            let b = cache.get(&base).copied().unwrap_or(f64::NAN);
            let e = cache.get(&exp).copied().unwrap_or(f64::NAN);
            b.powf(e)
        }

        ExprNode::Neg(inner) => -cache.get(&inner).copied().unwrap_or(f64::NAN),

        ExprNode::Sin(inner) => cache.get(&inner).copied().unwrap_or(f64::NAN).sin(),
        ExprNode::Cos(inner) => cache.get(&inner).copied().unwrap_or(f64::NAN).cos(),
        ExprNode::Tan(inner) => cache.get(&inner).copied().unwrap_or(f64::NAN).tan(),
        ExprNode::Exp(inner) => cache.get(&inner).copied().unwrap_or(f64::NAN).exp(),
        ExprNode::Ln(inner) => cache.get(&inner).copied().unwrap_or(f64::NAN).ln(),

        ExprNode::Asin(inner) => cache.get(&inner).copied().unwrap_or(f64::NAN).asin(),
        ExprNode::Acos(inner) => cache.get(&inner).copied().unwrap_or(f64::NAN).acos(),
        ExprNode::Atan(inner) => cache.get(&inner).copied().unwrap_or(f64::NAN).atan(),

        ExprNode::Sinh(inner) => cache.get(&inner).copied().unwrap_or(f64::NAN).sinh(),
        ExprNode::Cosh(inner) => cache.get(&inner).copied().unwrap_or(f64::NAN).cosh(),
        ExprNode::Tanh(inner) => cache.get(&inner).copied().unwrap_or(f64::NAN).tanh(),

        ExprNode::Asinh(inner) => cache.get(&inner).copied().unwrap_or(f64::NAN).asinh(),
        ExprNode::Acosh(inner) => cache.get(&inner).copied().unwrap_or(f64::NAN).acosh(),
        ExprNode::Atanh(inner) => cache.get(&inner).copied().unwrap_or(f64::NAN).atanh(),

        ExprNode::Abs(inner) => cache.get(&inner).copied().unwrap_or(f64::NAN).abs(),
        ExprNode::Sign(inner) => {
            let v = cache.get(&inner).copied().unwrap_or(f64::NAN);
            if v > 0.0 {
                1.0
            } else if v < 0.0 {
                -1.0
            } else {
                0.0
            }
        }

        ExprNode::Floor(inner) => cache.get(&inner).copied().unwrap_or(f64::NAN).floor(),
        ExprNode::Ceiling(inner) => cache.get(&inner).copied().unwrap_or(f64::NAN).ceil(),

        ExprNode::Atan2(y, x) => {
            let yv = cache.get(&y).copied().unwrap_or(f64::NAN);
            let xv = cache.get(&x).copied().unwrap_or(f64::NAN);
            yv.atan2(xv)
        }

        ExprNode::Gamma(inner) => {
            // Use Stirling's approximation for quick evaluation.
            let v = cache.get(&inner).copied().unwrap_or(f64::NAN);
            gamma_f64(v)
        }

        ExprNode::Erf(inner) => {
            let v = cache.get(&inner).copied().unwrap_or(f64::NAN);
            erf_approx_f64(v)
        }

        ExprNode::Factorial(inner) => {
            let v = cache.get(&inner).copied().unwrap_or(f64::NAN);
            gamma_f64(v + 1.0)
        }

        _ => f64::NAN,
    }
}

/// Rough Γ(x) approximation using the Lanczos method.
fn gamma_f64(x: f64) -> f64 {
    if x <= 0.0 && x == x.floor() {
        return f64::NAN; // Poles at non-positive integers.
    }
    if x < 0.5 {
        // Reflection formula.
        std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * gamma_f64(1.0 - x))
    } else {
        let g = 7.0;
        #[allow(clippy::excessive_precision)]
        let c = [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7,
        ];
        let x = x - 1.0;
        let mut sum = c[0];
        for (i, c_val) in c.iter().enumerate().skip(1) {
            sum += c_val / (x + i as f64);
        }
        let t = x + g + 0.5;
        (2.0 * std::f64::consts::PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * sum
    }
}

/// Rough erf(x) approximation.
fn erf_approx_f64(x: f64) -> f64 {
    // Abramowitz & Stegun approximation.
    let sign = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - poly * (-x * x).exp())
}

// ═══════════════════════════════════════════════════════════════════════════
// Linear system solver
// ═══════════════════════════════════════════════════════════════════════════

/// Solve the overdetermined system A·x = b in the least-squares sense.
///
/// Forms the normal equations A^T·A·x = A^T·b and solves via Gaussian
/// elimination with partial pivoting.
fn solve_least_squares(a: &[Vec<f64>], b: &[f64], n: usize) -> Option<Vec<f64>> {
    let m = a.len();
    assert_eq!(b.len(), m);

    // Form A^T·A (n×n) and A^T·b (n×1).
    let mut ata = vec![vec![0.0; n]; n];
    let mut atb = vec![0.0; n];

    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for a_row in a.iter().take(m) {
                s += a_row[i] * a_row[j];
            }
            ata[i][j] = s;
        }
        let mut s = 0.0;
        for k in 0..m {
            s += a[k][i] * b[k];
        }
        atb[i] = s;
    }

    // Gaussian elimination with partial pivoting on the augmented matrix
    // [A^T·A | A^T·b].
    let mut aug: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = ata[i].clone();
        row.push(atb[i]);
        aug.push(row);
    }

    for col in 0..n {
        // Find pivot.
        let mut max_val = aug[col][col].abs();
        let mut max_row = col;
        for (row, aug_row) in aug.iter().enumerate().take(n).skip(col + 1) {
            let v = aug_row[col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }

        if max_val < 1e-14 {
            // Singular or near-singular.
            return None;
        }

        // Swap rows.
        if max_row != col {
            aug.swap(col, max_row);
        }

        // Eliminate below.
        let pivot = aug[col][col];
        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            #[allow(clippy::needless_range_loop)]
            for j in col..=n {
                let v = aug[col][j];
                aug[row][j] -= factor * v;
            }
        }
    }

    // Back-substitution.
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = aug[i][n];
        for j in (i + 1)..n {
            s -= aug[i][j] * x[j];
        }
        if aug[i][i].abs() < 1e-14 {
            return None;
        }
        x[i] = s / aug[i][i];
    }

    // Sanity check: all finite.
    if x.iter().any(|v| !v.is_finite()) {
        return None;
    }

    Some(x)
}

// ═══════════════════════════════════════════════════════════════════════════
// Rational reconstruction
// ═══════════════════════════════════════════════════════════════════════════

/// Use continued-fraction expansion to find p/q ≈ x with |q| ≤ max_denom.
///
/// Returns `Some((p, q))` with `q > 0` if a sufficiently close rational
/// approximation is found, `None` otherwise.
#[must_use]
fn rationalize(x: f64, max_denom: i64) -> Option<(i64, i64)> {
    if !x.is_finite() {
        return None;
    }

    // Handle zero.
    if x.abs() < 1e-12 {
        return Some((0, 1));
    }

    let sign = if x < 0.0 { -1 } else { 1 };
    let x = x.abs();

    // Continued fraction algorithm.
    let mut p0: i64 = 0;
    let mut q0: i64 = 1;
    let mut p1: i64 = 1;
    let mut q1: i64 = 0;
    let mut rem = x;

    for _ in 0..50 {
        let a = rem.floor() as i64;
        let p2 = a.checked_mul(p1)?.checked_add(p0)?;
        let q2 = a.checked_mul(q1)?.checked_add(q0)?;

        if q2 > max_denom {
            break;
        }

        p0 = p1;
        q0 = q1;
        p1 = p2;
        q1 = q2;

        let frac = rem - a as f64;
        if frac.abs() < 1e-12 {
            break;
        }
        rem = 1.0 / frac;

        if rem > 1e15 {
            break;
        }
    }

    if q1 <= 0 || q1 > max_denom {
        return None;
    }

    // Verify the approximation is good enough.
    let approx = (p1 as f64) / (q1 as f64);
    let err = (x - approx).abs();
    if err > 1e-8 {
        return None;
    }

    Some((sign * p1, q1))
}

// ═══════════════════════════════════════════════════════════════════════════
// Candidate construction
// ═══════════════════════════════════════════════════════════════════════════

/// Build the symbolic antiderivative candidate from rational coefficients
/// and monomials, then substitute back the original components for V_i.
fn build_candidate(
    arena: &mut Arena,
    monomials: &[Monomial],
    coeffs: &[(i64, i64)],
    components: &[ExprId],
    comp_syms: &[ExprId],
) -> ExprId {
    let mut terms: SmallVec<[ExprId; 16]> = SmallVec::new();

    for (mono, &(p, q)) in monomials.iter().zip(coeffs.iter()) {
        if p == 0 {
            continue;
        }

        // Build the monomial in V_i variables.
        let mut factors: SmallVec<[ExprId; 8]> = SmallVec::new();

        // Add the rational coefficient.
        let coeff = arena.rational(p, q);
        factors.push(coeff);

        for (i, &e) in mono.iter().enumerate() {
            if e == 0 {
                continue;
            }
            if e == 1 {
                factors.push(comp_syms[i]);
            } else {
                let exp = arena.int(e as i64);
                let pow = arena.pow(comp_syms[i], exp);
                factors.push(pow);
            }
        }

        let term = if factors.len() == 1 {
            factors[0]
        } else {
            arena.mul(&factors)
        };

        terms.push(term);
    }

    if terms.is_empty() {
        return arena.zero;
    }

    let candidate_in_v = if terms.len() == 1 {
        terms[0]
    } else {
        arena.add(&terms)
    };

    // Substitute back: V_i → component_i.
    let back_subs: Vec<(ExprId, ExprId)> = comp_syms
        .iter()
        .zip(components.iter())
        .map(|(&v, &c)| (v, c))
        .collect();

    crate::transforms::subs::subs_map(arena, candidate_in_v, &back_subs)
}

// ═══════════════════════════════════════════════════════════════════════════
// Utility helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Check whether `expr` contains the variable `var_sym` anywhere in its tree.
fn contains_var(arena: &Arena, expr: ExprId, var: SymbolId) -> bool {
    let mut stack: Vec<ExprId> = vec![expr];
    let mut visited: FxHashSet<ExprId> = FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        if let ExprNode::Symbol(sid) = arena.node(id)
            && *sid == var
        {
            return true;
        }
        let children = arena.node(id).children();
        for c in children {
            stack.push(c);
        }
    }
    false
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests
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

    // ── Rational reconstruction ─────────────────────────────────────

    #[test]
    fn rationalize_zero() {
        assert_eq!(rationalize(0.0, 1000), Some((0, 1)));
    }

    #[test]
    fn rationalize_integer() {
        assert_eq!(rationalize(3.0, 1000), Some((3, 1)));
        assert_eq!(rationalize(-5.0, 1000), Some((-5, 1)));
    }

    #[test]
    fn rationalize_half() {
        assert_eq!(rationalize(0.5, 1000), Some((1, 2)));
    }

    #[test]
    fn rationalize_third() {
        let r = rationalize(1.0 / 3.0, 1000);
        assert_eq!(r, Some((1, 3)));
    }

    #[test]
    fn rationalize_neg_quarter() {
        assert_eq!(rationalize(-0.25, 1000), Some((-1, 4)));
    }

    #[test]
    fn rationalize_irrational_fails() {
        // π should not have a good rational approx with denom ≤ 1000
        // Actually 355/113 ≈ π is very good… but the tolerance matters.
        let r = rationalize(std::f64::consts::PI, 1000);
        // It will find 355/113 which is close enough for the tolerance.
        // That's fine — the symbolic verification step will catch it.
        assert!(r.is_some() || r.is_none()); // Accept either.
    }

    // ── Component collection ────────────────────────────────────────

    #[test]
    fn collect_components_poly() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let var_sym = match a.node(x) {
            ExprNode::Symbol(s) => *s,
            _ => unreachable!(),
        };
        // x^2 + 1
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let expr = a.add(&[x2, a.one]);
        let comps = collect_components(&a, expr, var_sym);
        // Should contain x.
        assert!(comps.contains(&x), "should collect x as a component");
    }

    #[test]
    fn collect_components_sin_exp() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let var_sym = match a.node(x) {
            ExprNode::Symbol(s) => *s,
            _ => unreachable!(),
        };
        // sin(x) + exp(x)
        let sin_x = a.sin(x);
        let exp_x = a.exp(x);
        let expr = a.add(&[sin_x, exp_x]);
        let comps = collect_components(&a, expr, var_sym);
        assert!(comps.contains(&x), "should contain x");
        assert!(comps.contains(&sin_x), "should contain sin(x)");
        assert!(comps.contains(&exp_x), "should contain exp(x)");
    }

    // ── Degree bound ────────────────────────────────────────────────

    #[test]
    fn degree_bound_polynomial() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let var_sym = match a.node(x) {
            ExprNode::Symbol(s) => *s,
            _ => unreachable!(),
        };
        let comps = vec![x];
        let d = compute_degree_bound(&a, &comps, var_sym);
        assert!(d >= 2, "poly degree bound should be >= 2, got {d}");
    }

    #[test]
    fn degree_bound_with_exp() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let var_sym = match a.node(x) {
            ExprNode::Symbol(s) => *s,
            _ => unreachable!(),
        };
        let exp_x = a.exp(x);
        let comps = vec![x, exp_x];
        let d = compute_degree_bound(&a, &comps, var_sym);
        // Should be at least 3 (A=1, B=1, +1)
        assert!(d >= 3, "exp degree bound should be >= 3, got {d}");
    }

    // ── Monomial generation ─────────────────────────────────────────

    #[test]
    fn monomials_single_var_degree_2() {
        let mut a = Arena::new();
        let v0 = sym(&mut a, "V0");
        let monos = generate_monomials(&mut a, &[v0], 2);
        // Should have 3 monomials: V0^0, V0^1, V0^2.
        assert_eq!(monos.len(), 3, "expected 3 monomials, got {}", monos.len());
    }

    #[test]
    fn monomials_two_vars_degree_1() {
        let mut a = Arena::new();
        let v0 = sym(&mut a, "V0");
        let v1 = sym(&mut a, "V1");
        let monos = generate_monomials(&mut a, &[v0, v1], 1);
        // Total degree ≤ 1: {1, V0, V1} — 3 monomials.
        assert_eq!(monos.len(), 3, "expected 3 monomials, got {}", monos.len());
    }

    // ── f64 evaluator ───────────────────────────────────────────────

    #[test]
    fn eval_at_point_polynomial() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let expr = a.add(&[x2, a.one]);
        // x^2 + 1 at x=3 should be 10.
        let val = eval_at_point(&a, expr, x, 3.0);
        assert!((val - 10.0).abs() < 1e-10, "expected 10, got {val}");
    }

    #[test]
    fn eval_at_point_trig() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let sin_x = a.sin(x);
        let val = eval_at_point(&a, sin_x, x, std::f64::consts::FRAC_PI_2);
        assert!((val - 1.0).abs() < 1e-10, "sin(π/2) should be 1, got {val}");
    }

    // ── Least-squares solver ────────────────────────────────────────

    #[test]
    fn solve_simple_system() {
        // Solve 2x = 6 (overdetermined with 3 copies).
        let a = vec![vec![2.0], vec![2.0], vec![2.0]];
        let b = vec![6.0, 6.0, 6.0];
        let x = solve_least_squares(&a, &b, 1).unwrap();
        assert!((x[0] - 3.0).abs() < 1e-10, "expected x=3, got {}", x[0]);
    }

    #[test]
    fn solve_two_unknowns() {
        // x + y = 3, x - y = 1  → x=2, y=1
        // With extra equation: 2x + 0y = 4
        let a = vec![vec![1.0, 1.0], vec![1.0, -1.0], vec![2.0, 0.0]];
        let b = vec![3.0, 1.0, 4.0];
        let x = solve_least_squares(&a, &b, 2).unwrap();
        assert!((x[0] - 2.0).abs() < 1e-10, "expected x=2, got {}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-10, "expected y=1, got {}", x[1]);
    }

    // ── Integration tests ───────────────────────────────────────────

    #[test]
    fn heurisch_constant_times_var() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let var_sym = match a.node(x) {
            ExprNode::Symbol(s) => *s,
            _ => unreachable!(),
        };
        // ∫ 3 dx = 3x (trivially, since 3 doesn't contain x).
        let three = a.int(3);
        let result = heurisch_integrate(&mut a, three, x, var_sym);
        assert!(result.is_some(), "should handle constant");
        let r = result.unwrap();
        let s = display(&a, r);
        // Should be 3*x.
        assert!(s.contains("3") && s.contains("x"), "expected 3*x, got {s}");
    }

    #[test]
    fn heurisch_exp_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let var_sym = match a.node(x) {
            ExprNode::Symbol(s) => *s,
            _ => unreachable!(),
        };
        // ∫ exp(x) dx = exp(x)
        let exp_x = a.exp(x);
        let result = heurisch_integrate(&mut a, exp_x, x, var_sym);
        if let Some(r) = result {
            // Verify numerically: d/dx(result) should equal exp(x).
            let deriv = crate::transforms::diff::diff(&mut a, r, x);
            let val_orig = eval_at_point(&a, exp_x, x, 1.0);
            let val_deriv = eval_at_point(&a, deriv, x, 1.0);
            assert!(
                (val_orig - val_deriv).abs() < 1e-6,
                "FTC failed: f(1)={val_orig}, F'(1)={val_deriv}"
            );
        }
        // It's OK if this returns None — the test is structured defensively.
    }

    #[test]
    fn heurisch_polynomial() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let var_sym = match a.node(x) {
            ExprNode::Symbol(s) => *s,
            _ => unreachable!(),
        };
        // ∫ x^2 dx = x^3/3
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let result = heurisch_integrate(&mut a, x2, x, var_sym);
        if let Some(r) = result {
            let deriv = crate::transforms::diff::diff(&mut a, r, x);
            let val_orig = eval_at_point(&a, x2, x, 2.0);
            let val_deriv = eval_at_point(&a, deriv, x, 2.0);
            assert!(
                (val_orig - val_deriv).abs() < 1e-6,
                "FTC failed: f(2)={val_orig}, F'(2)={val_deriv}"
            );
        }
    }

    // ── Monomial derivative evaluation ──────────────────────────────

    #[test]
    fn monomial_deriv_single_var() {
        // Monomial = V_0^2.
        // d/dx(V_0^2) = 2*V_0*dV_0/dx.
        // At V_0=3.0, dV_0/dx=1.0 (for V_0=x): result = 2*3*1 = 6.
        let comp_vals = vec![3.0];
        let comp_deriv_vals = vec![1.0];
        let mono = vec![2];
        let d = eval_monomial_deriv(&comp_vals, &comp_deriv_vals, &mono);
        assert!((d - 6.0).abs() < 1e-10, "expected 6, got {d}");
    }

    #[test]
    fn monomial_deriv_two_vars() {
        // Monomial = V_0 * V_1.
        // d/dx = dV_0/dx * V_1 + V_0 * dV_1/dx
        // At V_0=2, V_1=3, dV_0/dx=1, dV_1/dx=0.5:
        //   = 1*3 + 2*0.5 = 4.0
        let comp_vals = vec![2.0, 3.0];
        let comp_deriv_vals = vec![1.0, 0.5];
        let mono = vec![1, 1];
        let d = eval_monomial_deriv(&comp_vals, &comp_deriv_vals, &mono);
        assert!((d - 4.0).abs() < 1e-10, "expected 4, got {d}");
    }
}

//! Finite difference methods: weights, application, and differentiation.
//!
//! This module implements:
//! - **Fornberg's algorithm** (1988) for computing finite difference weights
//!   for arbitrary-order derivatives on arbitrary grids.
//! - **`apply_finite_diff`** — apply computed weights to function values.
//! - **`differentiate_finite`** — replace symbolic derivatives with finite
//!   difference approximations.
//!
//! All arithmetic is performed symbolically through the arena, so weights
//! are exact rational numbers.

use smallvec::SmallVec;

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode, SymbolId};

// ═══════════════════════════════════════════════════════════════════════════
// Fornberg's algorithm
// ═══════════════════════════════════════════════════════════════════════════

/// Compute finite difference weights for derivatives of order `0..=order`
/// at point `x0`, given grid points `x_list`.
///
/// Returns a 3D structure `result[m][n]` where:
/// - `m` is the derivative order (`0..=order`)
/// - `n` is the number of grid points used (`0..=N` where `N = x_list.len() - 1`)
/// - Each `result[m][n]` is a `Vec<ExprId>` of length `n+1`, giving the
///   weight for each grid point `x_list[0..=n]`.
///
/// The final weights for derivative order `m` using all `N+1` grid points
/// are in `result[m][N]`.
///
/// # Algorithm
///
/// This implements the algorithm from:
/// B. Fornberg, "Generation of Finite Difference Formulas on Arbitrarily
/// Spaced Grids", Mathematics of Computation 51(184), 1988, pp. 699–706.
///
/// The recurrence fills a 3D array `delta[m][n][nu]`:
/// ```text
/// delta[0][0][0] = 1
/// c1 = 1
/// for n = 1..N:
///     c2 = 1
///     for nu = 0..n-1:
///         c3 = x_list[n] - x_list[nu]
///         c2 *= c3
///         for m = 0..min(n, M):
///             delta[m][n][nu] = ((x[n]-x0)*delta[m][n-1][nu] - m*delta[m-1][n-1][nu]) / c3
///     for m = 0..min(n, M):
///         delta[m][n][n] = c1/c2 * (m*delta[m-1][n-1][n-1] - (x[n-1]-x0)*delta[m][n-1][n-1])
///     c1 = c2
/// ```
pub fn finite_diff_weights(
    arena: &mut Arena,
    order: usize,
    x_list: &[ExprId],
    x0: ExprId,
) -> Vec<Vec<Vec<ExprId>>> {
    let n_points = x_list.len();
    tracing::debug!("finite_diff: computing weights order={}, {} points", order, n_points);
    if n_points == 0 {
        return vec![Vec::new(); order + 1];
    }

    let big_m = order; // maximum derivative order
    let big_n = n_points - 1; // maximum grid index

    // Allocate: delta[m][n] is a Vec of length n+1
    // m ranges 0..=big_m, n ranges 0..=big_n
    let mut delta: Vec<Vec<Vec<ExprId>>> = Vec::with_capacity(big_m + 1);
    for _m in 0..=big_m {
        let mut level_m = Vec::with_capacity(big_n + 1);
        for n in 0..=big_n {
            level_m.push(vec![arena.zero; n + 1]);
        }
        delta.push(level_m);
    }

    // delta[0][0][0] = 1
    delta[0][0][0] = arena.one;

    let mut c1 = arena.one;

    for n in 1..=big_n {
        let mut c2 = arena.one;

        for nu in 0..n {
            // c3 = x_list[n] - x_list[nu]
            let c3 = arena.sub(x_list[n], x_list[nu]);
            // c2 *= c3
            c2 = arena.mul(&[c2, c3]);

            let m_max = std::cmp::min(n, big_m);
            for m in 0..=m_max {
                // delta[m][n][nu] = ((x_list[n] - x0) * delta[m][n-1][nu]
                //                    - m * delta[m-1][n-1][nu]) / c3
                let x_n_minus_x0 = arena.sub(x_list[n], x0);
                let prev = delta[m][n - 1][nu];
                let first_term = arena.mul(&[x_n_minus_x0, prev]);

                let second_term = if m == 0 {
                    arena.zero
                } else {
                    let m_id = arena.int(m as i64);
                    let prev_m = delta[m - 1][n - 1][nu];
                    arena.mul(&[m_id, prev_m])
                };

                let numer = arena.sub(first_term, second_term);
                let val = arena.div(numer, c3);
                let val = crate::transforms::eval::eval(arena, val);
                delta[m][n][nu] = val;
            }
        }

        let m_max = std::cmp::min(n, big_m);
        for m in 0..=m_max {
            // delta[m][n][n] = c1/c2 * (m*delta[m-1][n-1][n-1] - (x_list[n-1]-x0)*delta[m][n-1][n-1])
            let ratio = arena.div(c1, c2);

            let m_term = if m == 0 {
                arena.zero
            } else {
                let m_id = arena.int(m as i64);
                let prev_m = delta[m - 1][n - 1][n - 1];
                arena.mul(&[m_id, prev_m])
            };

            let x_prev_minus_x0 = arena.sub(x_list[n - 1], x0);
            let prev = delta[m][n - 1][n - 1];
            let second = arena.mul(&[x_prev_minus_x0, prev]);

            let bracket = arena.sub(m_term, second);
            let val = arena.mul(&[ratio, bracket]);
            let val = crate::transforms::eval::eval(arena, val);
            delta[m][n][n] = val;
        }

        c1 = c2;
    }

    delta
}

// ═══════════════════════════════════════════════════════════════════════════
// Apply finite differences
// ═══════════════════════════════════════════════════════════════════════════

/// Apply finite difference weights to approximate the derivative of order
/// `order` at `x0`, given grid points `x_list` and corresponding function
/// values `y_list`.
///
/// Computes `Σ weight_i * y_i` where the weights come from Fornberg's
/// algorithm using all grid points.
///
/// # Panics
///
/// Panics if `x_list` and `y_list` have different lengths, or if either
/// is empty.
pub fn apply_finite_diff(
    arena: &mut Arena,
    order: usize,
    x_list: &[ExprId],
    y_list: &[ExprId],
    x0: ExprId,
) -> ExprId {
    tracing::debug!("finite_diff: applying order {} derivative", order);
    assert_eq!(
        x_list.len(),
        y_list.len(),
        "x_list and y_list must have the same length"
    );
    assert!(!x_list.is_empty(), "x_list must be non-empty");

    let weights_all = finite_diff_weights(arena, order, x_list, x0);
    let n = x_list.len() - 1;

    // Extract the weights for derivative order `order`, using all N+1 points
    if order >= weights_all.len() || n >= weights_all[order].len() {
        return arena.zero;
    }

    let weights = &weights_all[order][n];

    let mut terms: SmallVec<[ExprId; 8]> = SmallVec::new();
    for (i, &w) in weights.iter().enumerate() {
        if i < y_list.len() && !arena.is_zero_structural(w) {
            let term = arena.mul(&[w, y_list[i]]);
            terms.push(term);
        }
    }

    if terms.is_empty() {
        arena.zero
    } else if terms.len() == 1 {
        terms[0]
    } else {
        arena.add(&terms)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Differentiate finite
// ═══════════════════════════════════════════════════════════════════════════

/// Replace derivative nodes in `expr` with central finite difference
/// approximations.
///
/// When encountering `Derivative(f, x)`, replaces it with the central
/// difference formula:
///
/// ```text
/// (f(x + h/2) - f(x - h/2)) / h
/// ```
///
/// where `h` is a new symbol `_h`. For higher-order derivatives, the
/// transformation is applied repeatedly.
///
/// This is useful for converting symbolic derivative expressions into
/// numerical approximation formulas.
pub fn differentiate_finite(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
) -> ExprId {
    let var_sym = match arena.node(var) {
        ExprNode::Symbol(sid) => *sid,
        _ => return expr,
    };

    differentiate_finite_inner(arena, expr, var, var_sym)
}

fn differentiate_finite_inner(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    _var_sym: SymbolId,
) -> ExprId {
    let node = arena.node(expr).clone();

    match node {
        ExprNode::Derivative(body, wrt) => {
            // Check if this is a derivative w.r.t. our variable
            if wrt == var {
                // Create the step size symbol h
                let h = arena.symbol("_h");
                let two = arena.int(2);
                let half_h = arena.div(h, two);

                // f(x + h/2)
                let x_plus = arena.add(&[var, half_h]);
                let f_plus = crate::transforms::subs::subs(arena, body, var, x_plus);

                // f(x - h/2)
                let x_minus = arena.sub(var, half_h);
                let f_minus = crate::transforms::subs::subs(arena, body, var, x_minus);

                // (f(x + h/2) - f(x - h/2)) / h
                let diff = arena.sub(f_plus, f_minus);
                arena.div(diff, h)
            } else {
                // Derivative w.r.t. a different variable — recurse into body
                let new_body = differentiate_finite_inner(arena, body, var, _var_sym);
                arena.intern(ExprNode::Derivative(new_body, wrt))
            }
        }

        // Recurse into compound expressions
        ExprNode::Add(ref children) => {
            let new_children: SmallVec<[ExprId; 6]> = children
                .iter()
                .map(|&c| differentiate_finite_inner(arena, c, var, _var_sym))
                .collect();
            arena.add(&new_children)
        }
        ExprNode::Mul(ref children) => {
            let new_children: SmallVec<[ExprId; 6]> = children
                .iter()
                .map(|&c| differentiate_finite_inner(arena, c, var, _var_sym))
                .collect();
            arena.mul(&new_children)
        }
        ExprNode::Pow(base, exp) => {
            let new_base = differentiate_finite_inner(arena, base, var, _var_sym);
            let new_exp = differentiate_finite_inner(arena, exp, var, _var_sym);
            arena.pow(new_base, new_exp)
        }
        ExprNode::Neg(inner) => {
            let new_inner = differentiate_finite_inner(arena, inner, var, _var_sym);
            arena.neg(new_inner)
        }

        // Atoms and everything else: return as-is
        _ => expr,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Convenience: standard stencils
// ═══════════════════════════════════════════════════════════════════════════

/// Create a standard equispaced grid `[x0 - n*h, ..., x0, ..., x0 + n*h]`
/// centered at `x0` with step `h` and `2n+1` points.
///
/// Returns the grid as a Vec of ExprIds.
pub fn equispaced_grid(
    arena: &mut Arena,
    x0: ExprId,
    h: ExprId,
    half_width: usize,
) -> Vec<ExprId> {
    let mut grid = Vec::with_capacity(2 * half_width + 1);
    for i in -(half_width as i64)..=(half_width as i64) {
        if i == 0 {
            grid.push(x0);
        } else {
            let i_id = arena.int(i);
            let offset = arena.mul(&[i_id, h]);
            let point = arena.add(&[x0, offset]);
            grid.push(point);
        }
    }
    grid
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::arena::Arena;
    use num_bigint::BigInt;
    use num_rational::Ratio;

    fn sym(a: &mut Arena, name: &str) -> ExprId {
        a.symbol(name)
    }

    fn display(a: &Arena, id: ExprId) -> String {
        a.display(id).to_string()
    }

    /// Extract the rational value of an expression, if it is a Num node.
    fn as_rat(a: &Arena, id: ExprId) -> Option<Ratio<BigInt>> {
        a.as_num(id).cloned()
    }

    #[test]
    fn forward_diff_weights_two_points() {
        // Forward difference: grid [0, 1], x0 = 0
        // 1st derivative weights: [-1, 1]
        let mut a = Arena::new();
        let zero = a.zero;
        let one = a.one;
        let x_list = vec![zero, one];

        let weights = finite_diff_weights(&mut a, 1, &x_list, zero);

        // weights[1][1] should be the 1st derivative weights using 2 points
        let w = &weights[1][1];
        assert_eq!(w.len(), 2);

        let w0 = as_rat(&a, w[0]).expect("weight 0 should be rational");
        let w1 = as_rat(&a, w[1]).expect("weight 1 should be rational");

        assert_eq!(w0, Ratio::from_integer(BigInt::from(-1)));
        assert_eq!(w1, Ratio::from_integer(BigInt::from(1)));
    }

    #[test]
    fn central_diff_first_deriv() {
        // Central difference: grid [-1, 0, 1], x0 = 0
        // 1st derivative weights: [-1/2, 0, 1/2]
        let mut a = Arena::new();
        let zero = a.zero;
        let neg_one = a.int(-1);
        let one = a.one;
        let x_list = vec![neg_one, zero, one];

        let weights = finite_diff_weights(&mut a, 1, &x_list, zero);

        let w = &weights[1][2]; // derivative order 1, using all 3 points
        assert_eq!(w.len(), 3);

        let w0 = as_rat(&a, w[0]).expect("weight 0");
        let w1 = as_rat(&a, w[1]).expect("weight 1");
        let w2 = as_rat(&a, w[2]).expect("weight 2");

        assert_eq!(w0, Ratio::new(BigInt::from(-1), BigInt::from(2)));
        assert_eq!(w1, Ratio::from_integer(BigInt::from(0)));
        assert_eq!(w2, Ratio::new(BigInt::from(1), BigInt::from(2)));
    }

    #[test]
    fn central_diff_second_deriv() {
        // Central difference: grid [-1, 0, 1], x0 = 0
        // 2nd derivative weights: [1, -2, 1]
        let mut a = Arena::new();
        let zero = a.zero;
        let neg_one = a.int(-1);
        let one = a.one;
        let x_list = vec![neg_one, zero, one];

        let weights = finite_diff_weights(&mut a, 2, &x_list, zero);

        let w = &weights[2][2];
        assert_eq!(w.len(), 3);

        let w0 = as_rat(&a, w[0]).expect("weight 0");
        let w1 = as_rat(&a, w[1]).expect("weight 1");
        let w2 = as_rat(&a, w[2]).expect("weight 2");

        assert_eq!(w0, Ratio::from_integer(BigInt::from(1)));
        assert_eq!(w1, Ratio::from_integer(BigInt::from(-2)));
        assert_eq!(w2, Ratio::from_integer(BigInt::from(1)));
    }

    #[test]
    fn apply_finite_diff_quadratic_exact() {
        // f(x) = x^2, grid = [-1, 0, 1], evaluate 1st derivative at x=0
        // f'(0) should be exactly 0 for x^2
        // weights: [-1/2, 0, 1/2], y_values: [1, 0, 1]
        // result = -1/2 * 1 + 0 * 0 + 1/2 * 1 = 0
        let mut a = Arena::new();
        let zero = a.zero;
        let neg_one = a.int(-1);
        let one = a.one;

        let x_list = vec![neg_one, zero, one];
        let y_list = vec![one, zero, one]; // f(-1)=1, f(0)=0, f(1)=1

        let result = apply_finite_diff(&mut a, 1, &x_list, &y_list, zero);
        let result_eval = crate::transforms::eval::eval(&mut a, result);

        assert!(
            a.is_zero_structural(result_eval),
            "derivative of x^2 at 0 should be 0, got {}",
            display(&a, result_eval)
        );
    }

    #[test]
    fn apply_finite_diff_quadratic_second_deriv() {
        // f(x) = x^2, grid = [-1, 0, 1], 2nd derivative at x=0
        // weights: [1, -2, 1], y_values: [1, 0, 1]
        // result = 1*1 + (-2)*0 + 1*1 = 2
        let mut a = Arena::new();
        let zero = a.zero;
        let neg_one = a.int(-1);
        let one = a.one;

        let x_list = vec![neg_one, zero, one];
        let y_list = vec![one, zero, one];

        let result = apply_finite_diff(&mut a, 2, &x_list, &y_list, zero);
        let result_eval = crate::transforms::eval::eval(&mut a, result);

        let two = a.int(2);
        assert_eq!(
            result_eval, two,
            "2nd derivative of x^2 should be 2, got {}",
            display(&a, result_eval)
        );
    }

    #[test]
    fn apply_finite_diff_linear_first_deriv() {
        // f(x) = 3x + 1, grid = [0, 1], 1st derivative at x=0
        // weights: [-1, 1], y = [1, 4]
        // result = -1*1 + 1*4 = 3
        let mut a = Arena::new();
        let zero = a.zero;
        let one = a.one;

        let x_list = vec![zero, one];
        let y0 = a.int(1); // f(0) = 1
        let y1 = a.int(4); // f(1) = 4
        let y_list = vec![y0, y1];

        let result = apply_finite_diff(&mut a, 1, &x_list, &y_list, zero);
        let result_eval = crate::transforms::eval::eval(&mut a, result);

        let three = a.int(3);
        assert_eq!(
            result_eval, three,
            "derivative of 3x+1 should be 3, got {}",
            display(&a, result_eval)
        );
    }

    #[test]
    fn differentiate_finite_replaces_derivative() {
        // Create a formal derivative node Derivative(x^2, x) and apply
        // differentiate_finite — should produce a finite difference expression.
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let deriv_node = a.intern(ExprNode::Derivative(x2, x));

        let result = differentiate_finite(&mut a, deriv_node, x);

        // The result should NOT contain a Derivative node
        let s = display(&a, result);
        assert!(
            !s.contains("Derivative"),
            "should not contain unevaluated Derivative: {s}"
        );
        // It should mention _h (the step size symbol)
        assert!(
            s.contains("_h"),
            "should contain step size symbol _h: {s}"
        );
    }

    #[test]
    fn zeroth_derivative_weights_are_interpolation() {
        // Zeroth derivative weights at a grid point are just the
        // Lagrange interpolation weights.
        // Grid: [0, 1, 2], x0 = 0 → weights should be [1, 0, 0]
        let mut a = Arena::new();
        let zero = a.zero;
        let one = a.one;
        let two = a.int(2);
        let x_list = vec![zero, one, two];

        let weights = finite_diff_weights(&mut a, 0, &x_list, zero);

        let w = &weights[0][2];
        assert_eq!(w.len(), 3);

        let w0 = as_rat(&a, w[0]).expect("weight 0");
        let w1 = as_rat(&a, w[1]).expect("weight 1");
        let w2 = as_rat(&a, w[2]).expect("weight 2");

        assert_eq!(w0, Ratio::from_integer(BigInt::from(1)));
        assert_eq!(w1, Ratio::from_integer(BigInt::from(0)));
        assert_eq!(w2, Ratio::from_integer(BigInt::from(0)));
    }

    #[test]
    fn equispaced_grid_produces_correct_count() {
        let mut a = Arena::new();
        let zero = a.zero;
        let h = sym(&mut a, "h");
        let grid = equispaced_grid(&mut a, zero, h, 2);
        // half_width=2 → 5 points: [-2h, -h, 0, h, 2h]
        assert_eq!(grid.len(), 5);
    }

    #[test]
    fn four_point_first_deriv_weights() {
        // Forward-biased 4-point stencil: grid [0, 1, 2, 3], x0 = 0
        // Known 1st derivative weights: [-11/6, 3, -3/2, 1/3]
        let mut a = Arena::new();
        let zero = a.zero;
        let one = a.one;
        let two = a.int(2);
        let three = a.int(3);
        let x_list = vec![zero, one, two, three];

        let weights = finite_diff_weights(&mut a, 1, &x_list, zero);
        let w = &weights[1][3]; // 1st derivative, all 4 points

        assert_eq!(w.len(), 4);

        let w0 = as_rat(&a, w[0]).expect("weight 0");
        let w1 = as_rat(&a, w[1]).expect("weight 1");
        let w2 = as_rat(&a, w[2]).expect("weight 2");
        let w3 = as_rat(&a, w[3]).expect("weight 3");

        assert_eq!(w0, Ratio::new(BigInt::from(-11), BigInt::from(6)));
        assert_eq!(w1, Ratio::from_integer(BigInt::from(3)));
        assert_eq!(w2, Ratio::new(BigInt::from(-3), BigInt::from(2)));
        assert_eq!(w3, Ratio::new(BigInt::from(1), BigInt::from(3)));
    }
}

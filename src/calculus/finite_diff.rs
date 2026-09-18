//! Finite difference methods: weights, application, and differentiation.
//!
//! * [`finite_diff_weights`] — Fornberg weights for the `order`-th
//!   derivative on an arbitrary grid (exact rationals / exact expressions).
//! * [`finite_diff_weights_table`] — the full Fornberg table for all
//!   derivative orders `0..=order` and all grid prefixes.
//! * [`apply_finite_diff`] — `Σ wᵢ yᵢ` for given function values.
//! * [`equispaced_grid`] — `[x₀ − n·h, …, x₀, …, x₀ + n·h]`.
//! * [`Ex::differentiate_finite`](crate::api::expr::Ex::differentiate_finite)
//!   — finite-difference approximation of an expression's derivative.
//!
//! All arithmetic is symbolic and exact.
//!
//! ```
//! use symplex::prelude::*;
//! use symplex::finite_diff::finite_diff_weights;
//!
//! let ctx = Context::new();
//! let grid = [ctx.int(-1), ctx.int(0), ctx.int(1)];
//! let w = finite_diff_weights(2, &grid, &ctx.int(0));
//! let s: Vec<String> = w.iter().map(|e| e.to_string()).collect();
//! assert_eq!(s, ["1", "-2", "1"]);
//! ```

use crate::api::context::Context;
use crate::api::expr::Ex;
use crate::base::arena::Arena;
use crate::base::errors::SymplexError;
use crate::base::node::{ExprId, ExprNode};
use crate::base::walk;
use crate::transforms::eval;

// ═══════════════════════════════════════════════════════════════════════════
// Fornberg's algorithm (arena level)
// ═══════════════════════════════════════════════════════════════════════════

/// Fornberg's table `delta[m][n][ν]`: weight of grid point `ν` for the
/// `m`-th derivative using the first `n+1` grid points.
///
/// B. Fornberg, "Generation of Finite Difference Formulas on Arbitrarily
/// Spaced Grids", *Math. Comp.* 51 (1988), 699–706.
pub(crate) fn fornberg_table(
    arena: &mut Arena,
    order: usize,
    x_list: &[ExprId],
    x0: ExprId,
) -> Vec<Vec<Vec<ExprId>>> {
    let n_points = x_list.len();
    tracing::debug!(
        "finite_diff: computing weights order={}, {} points",
        order,
        n_points
    );
    if n_points == 0 {
        return vec![Vec::new(); order + 1];
    }

    let big_m = order;
    let big_n = n_points - 1;

    let mut delta: Vec<Vec<Vec<ExprId>>> = Vec::with_capacity(big_m + 1);
    for _m in 0..=big_m {
        let mut level_m = Vec::with_capacity(big_n + 1);
        for n in 0..=big_n {
            level_m.push(vec![arena.zero; n + 1]);
        }
        delta.push(level_m);
    }

    delta[0][0][0] = arena.one;
    let mut c1 = arena.one;

    for n in 1..=big_n {
        let mut c2 = arena.one;
        for nu in 0..n {
            let c3 = arena.sub(x_list[n], x_list[nu]);
            c2 = arena.mul(&[c2, c3]);
            let m_max = std::cmp::min(n, big_m);
            for m in 0..=m_max {
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
                delta[m][n][nu] = eval::eval(arena, val);
            }
        }
        let m_max = std::cmp::min(n, big_m);
        for m in 0..=m_max {
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
            delta[m][n][n] = eval::eval(arena, val);
        }
        c1 = c2;
    }

    delta
}

/// Weights for the `order`-th derivative at `x0` using all of `x_list`.
pub(crate) fn weights_arena(
    arena: &mut Arena,
    order: usize,
    x_list: &[ExprId],
    x0: ExprId,
) -> Vec<ExprId> {
    if x_list.is_empty() {
        return Vec::new();
    }
    let table = fornberg_table(arena, order, x_list, x0);
    let n = x_list.len() - 1;
    match table.get(order).and_then(|t| t.get(n)) {
        Some(w) => w.clone(),
        None => vec![arena.zero; x_list.len()],
    }
}

/// `Σ wᵢ yᵢ` (arena level).
pub(crate) fn apply_arena(
    arena: &mut Arena,
    order: usize,
    x_list: &[ExprId],
    y_list: &[ExprId],
    x0: ExprId,
) -> ExprId {
    let weights = weights_arena(arena, order, x_list, x0);
    let mut terms = Vec::with_capacity(weights.len());
    for (w, y) in weights.iter().zip(y_list) {
        if !arena.is_zero_structural(*w) {
            terms.push(arena.mul(&[*w, *y]));
        }
    }
    let s = match terms.len() {
        0 => arena.zero,
        1 => terms[0],
        _ => arena.add(&terms),
    };
    eval::eval(arena, s)
}

// ═══════════════════════════════════════════════════════════════════════════
// Public Ex-based API
// ═══════════════════════════════════════════════════════════════════════════

fn ctx_of(exprs: &[Ex], fallback: &Ex) -> Context {
    exprs.first().unwrap_or(fallback).context()
}

fn wrap(ctx: &Context, id: ExprId) -> Ex {
    Ex::from_raw_parts(ctx.id, std::sync::Arc::clone(&ctx.inner), id)
}

/// Finite difference weights for the `order`-th derivative at `x0` using
/// every point of `x_list` (Fornberg's algorithm, exact arithmetic).
///
/// Returns one weight per grid point.  The grid points may be symbolic
/// (e.g. `x − h, x, x + h`).
///
/// # Panics
///
/// Panics if the expressions come from different contexts.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::finite_diff::finite_diff_weights;
///
/// let ctx = Context::new();
/// let grid = [ctx.int(0), ctx.int(1), ctx.int(2), ctx.int(3)];
/// let w = finite_diff_weights(1, &grid, &ctx.int(0));
/// let s: Vec<String> = w.iter().map(|e| e.to_string()).collect();
/// assert_eq!(s, ["-11/6", "3", "-3/2", "1/3"]);
/// ```
#[must_use]
pub fn finite_diff_weights(order: usize, x_list: &[Ex], x0: &Ex) -> Vec<Ex> {
    let ctx = ctx_of(x_list, x0);
    let ids: Vec<ExprId> = x_list.iter().map(|x| x0.checked_id(x)).collect();
    let x0_id = x0.raw_id();
    let w = {
        let mut inner = ctx.inner.write();
        weights_arena(&mut inner.arena, order, &ids, x0_id)
    };
    w.into_iter().map(|id| wrap(&ctx, id)).collect()
}

/// The full Fornberg table `table[m][n]` — weights for derivative order
/// `m ∈ 0..=order` using the first `n+1` grid points (`n ∈ 0..x_list.len()`).
///
/// # Panics
///
/// Panics if the expressions come from different contexts.
#[must_use]
pub fn finite_diff_weights_table(order: usize, x_list: &[Ex], x0: &Ex) -> Vec<Vec<Vec<Ex>>> {
    let ctx = ctx_of(x_list, x0);
    let ids: Vec<ExprId> = x_list.iter().map(|x| x0.checked_id(x)).collect();
    let x0_id = x0.raw_id();
    let table = {
        let mut inner = ctx.inner.write();
        fornberg_table(&mut inner.arena, order, &ids, x0_id)
    };
    table
        .into_iter()
        .map(|level| {
            level
                .into_iter()
                .map(|w| w.into_iter().map(|id| wrap(&ctx, id)).collect())
                .collect()
        })
        .collect()
}

/// Approximate the `order`-th derivative at `x0` from grid points `x_list`
/// and function values `y_list`: `Σ wᵢ yᵢ` with Fornberg weights.
///
/// Returns [`SymplexError::InvalidArgument`] if the lists are empty or of
/// different lengths.
///
/// # Panics
///
/// Panics if the expressions come from different contexts.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::finite_diff::apply_finite_diff;
///
/// let ctx = Context::new();
/// // f(x) = x², grid −1, 0, 1 → f'' = 2 exactly
/// let xs = [ctx.int(-1), ctx.int(0), ctx.int(1)];
/// let ys = [ctx.int(1), ctx.int(0), ctx.int(1)];
/// let d2 = apply_finite_diff(2, &xs, &ys, &ctx.int(0)).unwrap();
/// assert_eq!(d2.to_string(), "2");
/// ```
pub fn apply_finite_diff(
    order: usize,
    x_list: &[Ex],
    y_list: &[Ex],
    x0: &Ex,
) -> Result<Ex, SymplexError> {
    if x_list.is_empty() || x_list.len() != y_list.len() {
        return Err(SymplexError::InvalidArgument {
            operation: "apply_finite_diff",
            reason: format!(
                "x_list ({}) and y_list ({}) must be non-empty and of equal length",
                x_list.len(),
                y_list.len()
            ),
        });
    }
    let ctx = ctx_of(x_list, x0);
    let xs: Vec<ExprId> = x_list.iter().map(|x| x0.checked_id(x)).collect();
    let ys: Vec<ExprId> = y_list.iter().map(|y| x0.checked_id(y)).collect();
    let x0_id = x0.raw_id();
    let id = {
        let mut inner = ctx.inner.write();
        apply_arena(&mut inner.arena, order, &xs, &ys, x0_id)
    };
    Ok(wrap(&ctx, id))
}

/// Equispaced grid `[center − n·h, …, center, …, center + n·h]` with
/// `2·half_width + 1` points.
///
/// # Panics
///
/// Panics if `center` and `h` come from different contexts.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::finite_diff::equispaced_grid;
///
/// let ctx = Context::new();
/// let x = ctx.symbol("x");
/// let h = ctx.symbol("h");
/// let grid = equispaced_grid(&x, &h, 1);
/// let s: Vec<String> = grid.iter().map(|e| e.to_string()).collect();
/// assert_eq!(s, ["-h + x", "x", "h + x"]);
/// ```
#[must_use]
pub fn equispaced_grid(center: &Ex, h: &Ex, half_width: usize) -> Vec<Ex> {
    let ctx = center.context();
    let h_id = center.checked_id(h);
    let c_id = center.raw_id();
    let ids = {
        let mut inner = ctx.inner.write();
        let arena = &mut inner.arena;
        let mut grid = Vec::with_capacity(2 * half_width + 1);
        for i in -(half_width as i64)..=(half_width as i64) {
            if i == 0 {
                grid.push(c_id);
            } else {
                let i_id = arena.int(i);
                let offset = arena.mul(&[i_id, h_id]);
                grid.push(arena.add(&[c_id, offset]));
            }
        }
        grid
    };
    ids.into_iter().map(|id| wrap(&ctx, id)).collect()
}

/// Backend for [`Ex::differentiate_finite`](crate::api::expr::Ex::differentiate_finite).
pub(crate) fn differentiate_finite(expr: &Ex, var: &Ex, points: &[Ex], order: usize) -> Ex {
    let var_id = expr.checked_id(var);
    let pts: Vec<ExprId> = points.iter().map(|p| expr.checked_id(p)).collect();
    let id = {
        let mut inner = expr.inner.write();
        differentiate_finite_arena(&mut inner.arena, expr.raw_id(), var_id, &pts, order)
    };
    expr.wrap(id)
}

/// Replace `Derivative(·, var)` chains by their finite differences, then
/// apply the `order`-th difference to the whole expression.
fn differentiate_finite_arena(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    points: &[ExprId],
    order: usize,
) -> ExprId {
    if points.is_empty() {
        return expr;
    }
    // 1. Replace formal derivative nodes bottom-up (explicit post-order).
    let ids = walk::post_order_ids(arena, expr);
    let mut cache: rustc_hash::FxHashMap<ExprId, ExprId> = rustc_hash::FxHashMap::default();
    for id in ids {
        let node = arena.node(id).clone();
        let new_id = if let ExprNode::Derivative(body, wrt) = node
            && wrt == var
        {
            // Collapse nested Derivative(Derivative(f, var), var) → order m.
            let mut m = 1usize;
            let mut f = body;
            while let ExprNode::Derivative(b2, w2) = arena.node(f).clone()
                && w2 == var
            {
                m += 1;
                f = b2;
            }
            let f = cache.get(&f).copied().unwrap_or(f);
            finite_difference_of(arena, f, var, points, m)
        } else {
            crate::base::walk::rebuild_with_cache(arena, id, &cache)
        };
        cache.insert(id, new_id);
    }
    let replaced = cache.get(&expr).copied().unwrap_or(expr);
    // 2. Apply the requested order to the whole expression.
    if order == 0 {
        replaced
    } else {
        finite_difference_of(arena, replaced, var, points, order)
    }
}

/// `Σ wᵢ f(var → pointsᵢ)` with Fornberg weights for derivative `order` at `var`.
fn finite_difference_of(
    arena: &mut Arena,
    f: ExprId,
    var: ExprId,
    points: &[ExprId],
    order: usize,
) -> ExprId {
    let ys: Vec<ExprId> = points
        .iter()
        .map(|&p| crate::transforms::subs::subs(arena, f, var, p))
        .collect();
    apply_arena(arena, order, points, &ys, var)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn strs(v: &[Ex]) -> Vec<String> {
        v.iter().map(|e| e.to_string()).collect()
    }

    #[test]
    fn forward_diff_weights_two_points() {
        let ctx = Context::new();
        let w = finite_diff_weights(1, &[ctx.int(0), ctx.int(1)], &ctx.int(0));
        assert_eq!(strs(&w), ["-1", "1"]);
    }

    #[test]
    fn central_diff_first_and_second() {
        let ctx = Context::new();
        let grid = [ctx.int(-1), ctx.int(0), ctx.int(1)];
        assert_eq!(
            strs(&finite_diff_weights(1, &grid, &ctx.int(0))),
            ["-1/2", "0", "1/2"]
        );
        assert_eq!(
            strs(&finite_diff_weights(2, &grid, &ctx.int(0))),
            ["1", "-2", "1"]
        );
    }

    #[test]
    fn zeroth_derivative_is_interpolation() {
        let ctx = Context::new();
        let grid = [ctx.int(0), ctx.int(1), ctx.int(2)];
        assert_eq!(
            strs(&finite_diff_weights(0, &grid, &ctx.int(0))),
            ["1", "0", "0"]
        );
    }

    #[test]
    fn four_point_forward() {
        let ctx = Context::new();
        let grid = [ctx.int(0), ctx.int(1), ctx.int(2), ctx.int(3)];
        assert_eq!(
            strs(&finite_diff_weights(1, &grid, &ctx.int(0))),
            ["-11/6", "3", "-3/2", "1/3"]
        );
    }

    #[test]
    fn symbolic_step_weights() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let h = ctx.symbol("h");
        let grid = equispaced_grid(&x, &h, 1);
        let w = finite_diff_weights(1, &grid, &x);
        // [-1/(2h), 0, 1/(2h)]
        assert_eq!(w[1].to_string(), "0");
        let sum = (&w[0] + &w[2]).simplify();
        assert_eq!(sum.to_string(), "0");
        let prod = (&w[2] * &h * 2).simplify();
        assert_eq!(prod.to_string(), "1");
    }

    #[test]
    fn apply_quadratic_second_derivative() {
        let ctx = Context::new();
        let xs = [ctx.int(-1), ctx.int(0), ctx.int(1)];
        let ys = [ctx.int(1), ctx.int(0), ctx.int(1)];
        let d2 = apply_finite_diff(2, &xs, &ys, &ctx.int(0)).unwrap();
        assert_eq!(d2.to_string(), "2");
        let d1 = apply_finite_diff(1, &xs, &ys, &ctx.int(0)).unwrap();
        assert_eq!(d1.to_string(), "0");
        assert!(apply_finite_diff(1, &xs, &ys[..2], &ctx.int(0)).is_err());
        assert!(apply_finite_diff(1, &[], &[], &ctx.int(0)).is_err());
    }

    #[test]
    fn table_shape() {
        let ctx = Context::new();
        let grid = [ctx.int(0), ctx.int(1), ctx.int(2)];
        let t = finite_diff_weights_table(2, &grid, &ctx.int(0));
        assert_eq!(t.len(), 3);
        assert_eq!(t[1].len(), 3);
        assert_eq!(t[1][2].len(), 3);
        assert_eq!(strs(&t[1][1]), ["-1", "1"]);
    }

    #[test]
    fn differentiate_finite_polynomial_exact() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let h = ctx.symbol("h");
        let grid = equispaced_grid(&x, &h, 1);
        // central difference of x²: exactly 2x
        let d = x.powi(2).differentiate_finite(&x, &grid, 1).expand();
        assert_eq!(d.to_string(), "2*x");
        // second difference of x³: exactly 6x
        let d2 = x.powi(3).differentiate_finite(&x, &grid, 2).expand();
        assert_eq!(d2.to_string(), "6*x");
    }

    #[test]
    fn differentiate_finite_replaces_derivative_nodes() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let h = ctx.symbol("h");
        let grid = equispaced_grid(&x, &h, 1);
        let e = x.sin().formal_diff(&x);
        let d = e.differentiate_finite(&x, &grid, 0);
        let s = d.to_string();
        assert!(!s.contains("Derivative"), "{s}");
        assert!(s.contains("sin(") && s.contains("h + x"), "{s}");
        // nested derivative → second difference
        let e2 = x.powi(3).formal_diff(&x).formal_diff(&x);
        let d2 = e2.differentiate_finite(&x, &grid, 0).expand();
        assert_eq!(d2.to_string(), "6*x");
        // non-derivative expressions pass through with order 0
        let same = x.powi(2).differentiate_finite(&x, &grid, 0);
        assert_eq!(same.to_string(), "x^2");
    }

    #[test]
    fn equispaced_grid_count() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let h = ctx.symbol("h");
        assert_eq!(equispaced_grid(&x, &h, 2).len(), 5);
        assert_eq!(equispaced_grid(&x, &h, 0).len(), 1);
    }
}

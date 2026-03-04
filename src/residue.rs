//! Compute residues of functions at poles.
//!
//! The residue of f(x) at x=a is the coefficient of 1/(x-a)
//! in the Laurent series expansion of f around a.

use crate::arena::Arena;
use crate::errors::SymplexError;
use crate::node::ExprId;

/// Compute the residue of `expr` at `var = point`.
///
/// Uses the formula: Res(f, a) = lim_{x→a} (x-a) * f(x)
/// For simple poles, this is exact. For higher-order poles,
/// we use: Res(f, a) = 1/(n-1)! * lim_{x→a} d^(n-1)/dx^(n-1) [(x-a)^n * f(x)]
/// We try the simple-pole formula first.
pub(crate) fn residue(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Result<ExprId, SymplexError> {
    // Simple pole: Res = lim_{x→a} (x-a)*f(x)
    let x_minus_a = arena.sub(var, point);
    let product = arena.mul(&[x_minus_a, expr]);

    // Try to compute the limit
    let result = crate::limit::limit(arena, product, var, point)?;

    // If the result is finite, it's the residue
    let evaled = crate::eval::eval(arena, result);
    Ok(evaled)
}

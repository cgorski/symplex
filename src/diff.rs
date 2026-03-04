//! Symbolic differentiation.
//!
//! This module implements [`diff`], which computes the derivative of an
//! expression with respect to a symbol.  All standard differentiation
//! rules are supported:
//!
//! - Linearity: `d/dx(a + b) = da/dx + db/dx`
//! - Product rule (n-ary): `d/dx(a·b·c) = a'bc + ab'c + abc'`
//! - Power rule with chain rule: `d/dx(f^n) = n·f^(n-1)·f'`
//! - General power: `d/dx(f^g) = f^g·(g'·ln(f) + g·f'/f)`
//! - Chain rule for all elementary functions (sin, cos, tan, exp, ln, sqrt)
//! - Constants (numeric, π, e, i, ∞, NaN) differentiate to zero
//!
//! # Design
//!
//! Differentiation is computed **iteratively** using a bottom-up
//! post-order traversal.  The derivative of each sub-expression is
//! computed and cached before its parent is processed, so by the time
//! we reach a composite node, all child derivatives are available in
//! the cache.  **No recursion occurs** — the algorithm uses an explicit
//! stack, matching our Principle 5 (no recursive tree walks).
//!
//! Results are constructed through the canonical `Arena` constructors
//! (`add`, `mul`, `pow`, `neg`, etc.), so all canonical-form invariants
//! (like-term collection, Number*Add distribution, etc.) are preserved.

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::arena::Arena;
use crate::node::{ExprId, ExprNode, SymbolId};

/// Differentiate `expr` with respect to the symbol identified by `var`.
///
/// `var` must be an `ExprId` pointing to a `Symbol` node.  If `var`
/// does not appear in `expr`, the result is `arena.zero`.
///
/// The result is fully canonicalized.
pub(crate) fn diff(arena: &mut Arena, expr: ExprId, var: ExprId) -> ExprId {
    // Extract the SymbolId of the variable we're differentiating with
    // respect to.  If `var` is not a symbol, everything is "constant"
    // w.r.t. it, so the derivative is zero.
    let var_sym = match arena.node(var) {
        ExprNode::Symbol(sid) => *sid,
        _ => return arena.zero,
    };

    // Phase 1: Compute a post-order traversal of the expression DAG.
    let post_order = crate::walk::post_order_ids(arena, expr);

    // Phase 2: For each node in post-order, compute its derivative
    // and store it in the cache.
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();

    for &id in &post_order {
        let deriv = diff_node(arena, id, var_sym, &cache);
        cache.insert(id, deriv);
    }

    cache.get(&expr).copied().unwrap_or(arena.zero)
}

/// Compute the derivative of a single node, assuming all children's
/// derivatives are already available in `cache`.
fn diff_node(
    arena: &mut Arena,
    id: ExprId,
    var: SymbolId,
    cache: &FxHashMap<ExprId, ExprId>,
) -> ExprId {
    let node = arena.node(id).clone();

    match node {
        // ── Atoms ──────────────────────────────────────────────────

        // d/dx(number) = 0
        ExprNode::Num(_) => arena.zero,

        // d/dx(x) = 1, d/dx(y) = 0
        ExprNode::Symbol(sid) => {
            if sid == var {
                arena.one
            } else {
                arena.zero
            }
        }

        // Constants → 0
        ExprNode::Pi
        | ExprNode::E
        | ExprNode::ImaginaryUnit
        | ExprNode::Infinity
        | ExprNode::NegInfinity
        | ExprNode::ComplexInfinity
        | ExprNode::NaN
        | ExprNode::BoolTrue
        | ExprNode::BoolFalse => arena.zero,

        // Boolean/relational/logic → 0 (not differentiable)
        ExprNode::Gt(..)
        | ExprNode::Ge(..)
        | ExprNode::Eq_(..)
        | ExprNode::Ne(..)
        | ExprNode::And(_)
        | ExprNode::Or(_)
        | ExprNode::Not(_) => arena.zero,

        // Piecewise: differentiate each value piece, keep conditions
        ExprNode::Piecewise(ref pairs) => {
            let pairs = pairs.clone();
            let mut new_pairs = SmallVec::new();
            for &(val, cond) in &pairs {
                let dval = get_deriv(cache, val, arena);
                new_pairs.push((dval, cond));
            }
            arena.intern(ExprNode::Piecewise(new_pairs))
        }

        // ── Add: linearity ─────────────────────────────────────────
        // d/dx(a + b + c) = da + db + dc
        ExprNode::Add(ref children) => {
            let derivs: SmallVec<[ExprId; 6]> = children
                .iter()
                .map(|&child| get_deriv(cache, child, arena))
                .collect();
            arena.add(&derivs)
        }

        // ── Mul: generalized product rule ──────────────────────────
        // d/dx(f₁·f₂·…·fₙ) = Σᵢ (f₁·…·fᵢ'·…·fₙ)
        ExprNode::Mul(ref children) => {
            let n = children.len();
            if n == 0 {
                return arena.zero;
            }

            let children = children.clone();
            let mut sum_terms: SmallVec<[ExprId; 6]> = SmallVec::new();

            for i in 0..n {
                let di = get_deriv(cache, children[i], arena);
                // Skip zero derivatives (common case: numeric coefficients).
                if arena.is_zero_structural(di) {
                    continue;
                }
                // Build the product: f₁ * … * fᵢ' * … * fₙ
                let mut factors: SmallVec<[ExprId; 6]> = SmallVec::new();
                for (j, &child) in children.iter().enumerate() {
                    if j == i {
                        factors.push(di);
                    } else {
                        factors.push(child);
                    }
                }
                let term = arena.mul(&factors);
                sum_terms.push(term);
            }

            if sum_terms.is_empty() {
                arena.zero
            } else {
                arena.add(&sum_terms)
            }
        }

        // ── Pow: power rule + chain rule ───────────────────────────
        // General: d/dx(f^g) = f^g * (g'·ln(f) + g·f'/f)
        // Special case (g constant): d/dx(f^n) = n·f^(n-1)·f'
        ExprNode::Pow(base, exp) => {
            let dbase = get_deriv(cache, base, arena);
            let dexp = get_deriv(cache, exp, arena);

            let base_is_const = arena.is_zero_structural(dbase);
            let exp_is_const = arena.is_zero_structural(dexp);

            if base_is_const && exp_is_const {
                // Both constant → derivative is 0.
                arena.zero
            } else if exp_is_const {
                // f^n where n is constant w.r.t. x:
                // d/dx = n * f^(n-1) * f'
                let n_minus_1 = arena.sub(exp, arena.one);
                let pow_part = arena.pow(base, n_minus_1);
                arena.mul(&[exp, pow_part, dbase])
            } else if base_is_const {
                // a^g where a is constant w.r.t. x:
                // d/dx = a^g * ln(a) * g'
                let ln_base = arena.ln(base);
                let pow_part = arena.pow(base, exp);
                arena.mul(&[pow_part, ln_base, dexp])
            } else {
                // General case: f^g
                // d/dx = f^g * (g'·ln(f) + g·f'/f)
                let pow_part = arena.pow(base, exp);
                let ln_f = arena.ln(base);
                let term1 = arena.mul(&[dexp, ln_f]);
                let neg_one = arena.neg_one;
                let f_inv = arena.pow(base, neg_one);
                let term2 = arena.mul(&[exp, dbase, f_inv]);
                let inner = arena.add(&[term1, term2]);
                arena.mul(&[pow_part, inner])
            }
        }

        // ── Neg: d/dx(-f) = -f' ───────────────────────────────────
        ExprNode::Neg(inner) => {
            let di = get_deriv(cache, inner, arena);
            arena.neg(di)
        }

        // ── Sin: d/dx(sin(f)) = cos(f) · f' ───────────────────────
        ExprNode::Sin(inner) => {
            let di = get_deriv(cache, inner, arena);
            if arena.is_zero_structural(di) {
                return arena.zero;
            }
            let cos_f = arena.cos(inner);
            arena.mul(&[cos_f, di])
        }

        // ── Cos: d/dx(cos(f)) = -sin(f) · f' ─────────────────────
        ExprNode::Cos(inner) => {
            let di = get_deriv(cache, inner, arena);
            if arena.is_zero_structural(di) {
                return arena.zero;
            }
            let sin_f = arena.sin(inner);
            let neg_sin = arena.neg(sin_f);
            arena.mul(&[neg_sin, di])
        }

        // ── Tan: d/dx(tan(f)) = (1 + tan²(f)) · f' ───────────────
        // Equivalently: sec²(f) · f'
        ExprNode::Tan(inner) => {
            let di = get_deriv(cache, inner, arena);
            if arena.is_zero_structural(di) {
                return arena.zero;
            }
            let tan_f = arena.tan(inner);
            let two = arena.int(2);
            let tan_sq = arena.pow(tan_f, two);
            let one = arena.one;
            let one_plus_tan_sq = arena.add(&[one, tan_sq]);
            arena.mul(&[one_plus_tan_sq, di])
        }

        // ── Exp: d/dx(exp(f)) = exp(f) · f' ───────────────────────
        ExprNode::Exp(inner) => {
            let di = get_deriv(cache, inner, arena);
            if arena.is_zero_structural(di) {
                return arena.zero;
            }
            let exp_f = arena.exp(inner);
            arena.mul(&[exp_f, di])
        }

        // ── Ln: d/dx(ln(f)) = f' / f ──────────────────────────────
        ExprNode::Ln(inner) => {
            let di = get_deriv(cache, inner, arena);
            if arena.is_zero_structural(di) {
                return arena.zero;
            }
            arena.div(di, inner)
        }

        // d/dx(|f|) = sign(f) * f'
        ExprNode::Abs(inner) => {
            let inner_diff = cache.get(&inner).copied().unwrap_or(arena.zero);
            let sign_f = arena.sign(inner);
            arena.mul(&[sign_f, inner_diff])
        }

        // d/dx(sign(f)) = 0 (piecewise, but zero almost everywhere)
        ExprNode::Sign(_) => arena.zero,

        // d/dx(asin(f)) = f' / sqrt(1 - f^2)
        ExprNode::Asin(inner) => {
            let df = get_deriv(cache, inner, arena);
            if arena.is_zero_structural(df) {
                return arena.zero;
            }
            let one = arena.one;
            let two = arena.int(2);
            let f_sq = arena.pow(inner, two);
            let one_minus_f_sq = arena.sub(one, f_sq);
            let sqrt_denom = arena.sqrt(one_minus_f_sq);
            arena.div(df, sqrt_denom)
        }

        // d/dx(acos(f)) = -f' / sqrt(1 - f^2)
        ExprNode::Acos(inner) => {
            let df = get_deriv(cache, inner, arena);
            if arena.is_zero_structural(df) {
                return arena.zero;
            }
            let one = arena.one;
            let two = arena.int(2);
            let f_sq = arena.pow(inner, two);
            let one_minus_f_sq = arena.sub(one, f_sq);
            let sqrt_denom = arena.sqrt(one_minus_f_sq);
            let frac = arena.div(df, sqrt_denom);
            arena.neg(frac)
        }

        // d/dx(atan(f)) = f' / (1 + f^2)
        ExprNode::Atan(inner) => {
            let df = get_deriv(cache, inner, arena);
            if arena.is_zero_structural(df) {
                return arena.zero;
            }
            let one = arena.one;
            let two = arena.int(2);
            let f_sq = arena.pow(inner, two);
            let one_plus_f_sq = arena.add(&[one, f_sq]);
            arena.div(df, one_plus_f_sq)
        }

        // d/dvar(atan2(y, x)) = (x·dy - y·dx) / (x² + y²)
        ExprNode::Atan2(y_id, x_id) => {
            let dy = get_deriv(cache, y_id, arena);
            let dx = get_deriv(cache, x_id, arena);
            let both_zero = arena.is_zero_structural(dy) && arena.is_zero_structural(dx);
            if both_zero {
                return arena.zero;
            }
            let two = arena.int(2);
            let x_sq = arena.pow(x_id, two);
            let y_sq = arena.pow(y_id, two);
            let denom = arena.add(&[x_sq, y_sq]);
            let x_dy = arena.mul(&[x_id, dy]);
            let y_dx = arena.mul(&[y_id, dx]);
            let numer = arena.sub(x_dy, y_dx);
            arena.div(numer, denom)
        }

        // d/dx(sinh(f)) = cosh(f) * f'
        ExprNode::Sinh(inner) => {
            let df = get_deriv(cache, inner, arena);
            if arena.is_zero_structural(df) {
                return arena.zero;
            }
            let cosh_f = arena.intern(ExprNode::Cosh(inner));
            arena.mul(&[cosh_f, df])
        }

        // d/dx(cosh(f)) = sinh(f) * f'
        ExprNode::Cosh(inner) => {
            let df = get_deriv(cache, inner, arena);
            if arena.is_zero_structural(df) {
                return arena.zero;
            }
            let sinh_f = arena.intern(ExprNode::Sinh(inner));
            arena.mul(&[sinh_f, df])
        }

        // d/dx(tanh(f)) = (1 - tanh^2(f)) * f'
        ExprNode::Tanh(inner) => {
            let df = get_deriv(cache, inner, arena);
            if arena.is_zero_structural(df) {
                return arena.zero;
            }
            let tanh_f = arena.intern(ExprNode::Tanh(inner));
            let two = arena.int(2);
            let tanh_sq = arena.pow(tanh_f, two);
            let one = arena.one;
            let one_minus_tanh_sq = arena.sub(one, tanh_sq);
            arena.mul(&[one_minus_tanh_sq, df])
        }

        // d/dx(asinh(f)) = f' / sqrt(f^2 + 1)
        ExprNode::Asinh(inner) => {
            let df = get_deriv(cache, inner, arena);
            if arena.is_zero_structural(df) {
                return arena.zero;
            }
            let one = arena.one;
            let two = arena.int(2);
            let f_sq = arena.pow(inner, two);
            let f_sq_plus_1 = arena.add(&[f_sq, one]);
            let sqrt_denom = arena.sqrt(f_sq_plus_1);
            arena.div(df, sqrt_denom)
        }

        // d/dx(acosh(f)) = f' / sqrt(f^2 - 1)
        ExprNode::Acosh(inner) => {
            let df = get_deriv(cache, inner, arena);
            if arena.is_zero_structural(df) {
                return arena.zero;
            }
            let one = arena.one;
            let two = arena.int(2);
            let f_sq = arena.pow(inner, two);
            let f_sq_minus_1 = arena.sub(f_sq, one);
            let sqrt_denom = arena.sqrt(f_sq_minus_1);
            arena.div(df, sqrt_denom)
        }

        // d/dx(atanh(f)) = f' / (1 - f^2)
        ExprNode::Atanh(inner) => {
            let df = get_deriv(cache, inner, arena);
            if arena.is_zero_structural(df) {
                return arena.zero;
            }
            let one = arena.one;
            let two = arena.int(2);
            let f_sq = arena.pow(inner, two);
            let one_minus_f_sq = arena.sub(one, f_sq);
            arena.div(df, one_minus_f_sq)
        }

        // Factorial: d/dx(n!) — leave as unevaluated derivative
        // (factorial is typically of integer-valued expressions)
        ExprNode::Factorial(_) => {
            let v = var_expr(arena, var);
            arena.intern(ExprNode::Derivative(id, v))
        }

        // Binomial: leave as unevaluated derivative
        ExprNode::Binomial(_, _) => {
            let v = var_expr(arena, var);
            arena.intern(ExprNode::Derivative(id, v))
        }

        // ── Apply (user-defined function): leave unevaluated ───────
        ExprNode::Apply(_, _) => {
            let v = var_expr(arena, var);
            arena.intern(ExprNode::Derivative(id, v))
        }

        // ── Derivative: leave as higher-order derivative ───────────
        ExprNode::Derivative(_, _) => {
            let v = var_expr(arena, var);
            arena.intern(ExprNode::Derivative(id, v))
        }

        // ── Integral: fundamental theorem of calculus ───────────────
        // d/dx(∫ f dx) = f when the integration variable matches the
        // differentiation variable.
        ExprNode::Integral(body, int_var) => {
            if let ExprNode::Symbol(int_sym) = arena.node(int_var)
                && *int_sym == var
            {
                return body;
            }
            // Different variable — leave as unevaluated derivative.
            let v = var_expr(arena, var);
            arena.intern(ExprNode::Derivative(id, v))
        }

        // ── Floor/Ceiling: piecewise constant → derivative is 0 ────
        ExprNode::Floor(_) | ExprNode::Ceiling(_) => arena.zero,

        // ── Min/Max: complex piecewise derivative → leave unevaluated
        ExprNode::Min(_) | ExprNode::Max(_) => {
            let v = var_expr(arena, var);
            arena.intern(ExprNode::Derivative(id, v))
        }

        // ── Sum: linearity — d/dx Sum(f, k, a, b) = Sum(d/dx f, k, a, b)
        // assuming the summation variable k is not x.
        ExprNode::Sum(body, sum_var, lo, hi) => {
            if let ExprNode::Symbol(sum_sym) = arena.node(sum_var)
                && *sum_sym == var
            {
                // Differentiating w.r.t. the summation variable itself — leave unevaluated.
                let v = var_expr(arena, var);
                return arena.intern(ExprNode::Derivative(id, v));
            }
            let dbody = get_deriv(cache, body, arena);
            arena.intern(ExprNode::Sum(dbody, sum_var, lo, hi))
        }

        // ── Product_: leave as unevaluated derivative ──────────────
        ExprNode::Product_(_, _, _, _) => {
            let v = var_expr(arena, var);
            arena.intern(ExprNode::Derivative(id, v))
        }
    }
}

/// Look up the derivative of `id` from the cache.
/// Returns `arena.zero` if not found (shouldn't happen in correct usage).
#[inline]
fn get_deriv(cache: &FxHashMap<ExprId, ExprId>, id: ExprId, arena: &Arena) -> ExprId {
    cache.get(&id).copied().unwrap_or(arena.zero)
}

/// Reconstruct the Symbol ExprId for the variable.
fn var_expr(arena: &mut Arena, var: SymbolId) -> ExprId {
    arena.intern(ExprNode::Symbol(var))
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::Arena;

    fn sym(a: &mut Arena, name: &str) -> ExprId {
        a.symbol(name)
    }

    fn display(a: &Arena, id: ExprId) -> String {
        a.display(id).to_string()
    }

    // ── Constants ───────────────────────────────────────────────────

    #[test]
    fn diff_constant_is_zero() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let five = a.int(5);
        assert_eq!(diff(&mut a, five, x), a.zero);
    }

    #[test]
    fn diff_pi_is_zero() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let pi = a.pi;
        let result = diff(&mut a, pi, x);
        assert_eq!(result, a.zero);
    }

    #[test]
    fn diff_other_symbol_is_zero() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        assert_eq!(diff(&mut a, y, x), a.zero);
    }

    // ── d/dx(x) = 1 ────────────────────────────────────────────────

    #[test]
    fn diff_x_is_one() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        assert_eq!(diff(&mut a, x, x), a.one);
    }

    // ── Linearity: Add ──────────────────────────────────────────────

    #[test]
    fn diff_add() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let three = a.int(3);
        // d/dx(x + 3) = 1
        let expr = a.add(&[x, three]);
        let result = diff(&mut a, expr, x);
        assert_eq!(result, a.one);
    }

    #[test]
    fn diff_add_two_xs() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // d/dx(x + x) = d/dx(2x) = 2
        let expr = a.add(&[x, x]);
        let result = diff(&mut a, expr, x);
        let two = a.int(2);
        assert_eq!(result, two);
    }

    // ── Product rule ────────────────────────────────────────────────

    #[test]
    fn diff_mul_constant_times_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let three = a.int(3);
        // d/dx(3*x) = 3
        let expr = a.mul(&[three, x]);
        let result = diff(&mut a, expr, x);
        assert_eq!(display(&a, result), "3");
    }

    #[test]
    fn diff_mul_x_times_y() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        // d/dx(x*y) = y
        let expr = a.mul(&[x, y]);
        let result = diff(&mut a, expr, x);
        assert_eq!(result, y);
    }

    #[test]
    fn diff_mul_x_times_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // d/dx(x*x) = d/dx(x^2) = 2*x
        // But x*x canonicalizes to x^2, and diff of x^2 uses power rule.
        let expr = a.mul(&[x, x]);
        let result = diff(&mut a, expr, x);
        assert_eq!(display(&a, result), "2*x");
    }

    #[test]
    fn diff_product_three_factors() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let z = sym(&mut a, "z");
        // d/dx(x*y*z) = y*z  (only x depends on x)
        let expr = a.mul(&[x, y, z]);
        let result = diff(&mut a, expr, x);
        assert_eq!(display(&a, result), "y*z");
    }

    // ── Power rule ──────────────────────────────────────────────────

    #[test]
    fn diff_x_squared() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let expr = a.pow(x, two);
        let result = diff(&mut a, expr, x);
        assert_eq!(display(&a, result), "2*x");
    }

    #[test]
    fn diff_x_cubed() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let three = a.int(3);
        let expr = a.pow(x, three);
        let result = diff(&mut a, expr, x);
        assert_eq!(display(&a, result), "3*x^2");
    }

    #[test]
    fn diff_x_to_the_one() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x^1 canonicalizes to x, so diff gives 1.
        let one = a.one;
        let expr = a.pow(x, one);
        assert_eq!(expr, x); // canonical: x^1 = x
        let result = diff(&mut a, expr, x);
        assert_eq!(result, a.one);
    }

    #[test]
    fn diff_constant_power() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let three = a.int(3);
        // d/dx(2^3) = 0 (constant)
        let expr = a.pow(two, three);
        let result = diff(&mut a, expr, x);
        assert_eq!(result, a.zero);
    }

    // ── Chain rule with power ───────────────────────────────────────

    #[test]
    fn diff_sin_x_squared() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        // d/dx(sin(x)^2) = 2*sin(x)*cos(x)
        let sin_x = a.sin(x);
        let expr = a.pow(sin_x, two);
        let result = diff(&mut a, expr, x);
        let s = display(&a, result);
        assert!(s.contains("2"), "should contain 2, got: {s}");
        assert!(s.contains("sin"), "should contain sin, got: {s}");
        assert!(s.contains("cos"), "should contain cos, got: {s}");
    }

    // ── Neg ─────────────────────────────────────────────────────────

    #[test]
    fn diff_neg_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.neg(x);
        let result = diff(&mut a, expr, x);
        assert_eq!(result, a.neg_one);
    }

    // ── Trigonometric functions ──────────────────────────────────────

    #[test]
    fn diff_sin_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.sin(x);
        let result = diff(&mut a, expr, x);
        assert_eq!(display(&a, result), "cos(x)");
    }

    #[test]
    fn diff_cos_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.cos(x);
        let result = diff(&mut a, expr, x);
        assert_eq!(display(&a, result), "-sin(x)");
    }

    #[test]
    fn diff_tan_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.tan(x);
        let result = diff(&mut a, expr, x);
        let s = display(&a, result);
        // Should be 1 + tan(x)^2
        assert!(s.contains("tan"), "should contain tan, got: {s}");
    }

    #[test]
    fn diff_sin_chain_rule() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let x_sq = a.pow(x, two);
        // d/dx(sin(x^2)) = cos(x^2) * 2*x = 2*x*cos(x^2)
        let expr = a.sin(x_sq);
        let result = diff(&mut a, expr, x);
        let s = display(&a, result);
        assert!(s.contains("cos"), "should contain cos, got: {s}");
        assert!(s.contains("2"), "should contain 2, got: {s}");
        assert!(s.contains("x"), "should contain x, got: {s}");
    }

    // ── Exponential and logarithm ───────────────────────────────────

    #[test]
    fn diff_exp_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.exp(x);
        let result = diff(&mut a, expr, x);
        assert_eq!(display(&a, result), "exp(x)");
    }

    #[test]
    fn diff_ln_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.ln(x);
        let result = diff(&mut a, expr, x);
        assert_eq!(display(&a, result), "1/x");
    }

    #[test]
    fn diff_exp_chain_rule() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let two_x = a.mul(&[two, x]);
        // d/dx(exp(2*x)) = 2*exp(2*x)
        let expr = a.exp(two_x);
        let result = diff(&mut a, expr, x);
        let s = display(&a, result);
        assert!(s.contains("exp"), "should contain exp, got: {s}");
        assert!(s.contains("2"), "should contain 2, got: {s}");
    }

    // ── Sqrt ────────────────────────────────────────────────────────

    #[test]
    fn diff_sqrt_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.sqrt(x);
        let result = diff(&mut a, expr, x);
        let s = display(&a, result);
        // Should be 1/(2*sqrt(x)) = (1/2) * sqrt(x)^(-1)
        assert!(s.contains("sqrt") || s.contains("1/2"), "got: {s}");
    }

    // ── Polynomial differentiation ──────────────────────────────────

    #[test]
    fn diff_polynomial() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let three = a.int(3);

        // d/dx(x^3 + 2*x^2 + x + 5)
        let x3 = a.pow(x, three);
        let x2 = a.pow(x, two);
        let two_x2 = a.mul(&[two, x2]);
        let five = a.int(5);
        let expr = a.add(&[x3, two_x2, x, five]);
        let result = diff(&mut a, expr, x);
        // = 3*x^2 + 4*x + 1
        let s = display(&a, result);
        assert!(s.contains("3*x^2"), "should contain 3*x^2, got: {s}");
        assert!(s.contains("4*x"), "should contain 4*x, got: {s}");
        assert!(s.contains('1'), "should contain 1, got: {s}");
    }

    // ── Higher-order derivatives ─────────────────────────────────────

    #[test]
    fn diff_second_derivative() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let three = a.int(3);
        // d²/dx²(x^3) = d/dx(3*x^2) = 6*x
        let expr = a.pow(x, three);
        let first = diff(&mut a, expr, x);
        let second = diff(&mut a, first, x);
        assert_eq!(display(&a, second), "6*x");
    }

    // ── Deep expression (stack safety) ──────────────────────────────

    #[test]
    fn diff_deep_expression_no_stack_overflow() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");

        // Build sin(sin(sin(...sin(x)...))) 100 levels deep.
        let mut expr = x;
        for _ in 0..100 {
            expr = a.sin(expr);
        }

        // Should not stack-overflow.
        let result = diff(&mut a, expr, x);
        assert_ne!(
            result, a.zero,
            "derivative of sin^100(x) should not be zero"
        );
    }

    // ── Product rule: x * sin(x) ────────────────────────────────────

    #[test]
    fn diff_x_times_sin_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // d/dx(x * sin(x)) = sin(x) + x*cos(x)
        let sin_x = a.sin(x);
        let expr = a.mul(&[x, sin_x]);
        let result = diff(&mut a, expr, x);
        let s = display(&a, result);
        assert!(s.contains("sin"), "should contain sin, got: {s}");
        assert!(s.contains("cos"), "should contain cos, got: {s}");
    }

    // ── Fundamental theorem of calculus ──────────────────────────────

    #[test]
    fn diff_integral_fundamental_theorem() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // d/dx(∫ x^2 dx) = x^2
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let integral = a.intern(crate::node::ExprNode::Integral(x2, x));
        let result = diff(&mut a, integral, x);
        assert_eq!(display(&a, result), "x^2");
    }

    #[test]
    fn diff_integral_different_var_stays_unevaluated() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        // d/dx(∫ y dy) stays as Derivative(Integral(y, y), x)
        let integral = a.intern(crate::node::ExprNode::Integral(y, y));
        let result = diff(&mut a, integral, x);
        assert_eq!(display(&a, result), "Derivative(Integral(y, y), x)");
    }
}

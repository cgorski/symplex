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

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode, SymbolId};

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
    let post_order = crate::base::walk::post_order_ids(arena, expr);

    // Phase 2: For each node in post-order, compute its derivative
    // and store it in the cache.
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();

    for &id in &post_order {
        let deriv = diff_node(arena, id, var_sym, &cache);
        cache.insert(id, deriv);
    }

    cache.get(&expr).copied().unwrap_or(arena.zero)
}

/// Differentiate `expr` with respect to `var`, treating symbols in `deps`
/// as dependent on `var`.
///
/// For any symbol `y` in `deps`, d/d(var)(y) returns `Derivative(y, var)`
/// instead of zero.  All existing chain/product/sum rules automatically
/// propagate these formal derivatives correctly.
///
/// Callers that don't need dependency awareness should use [`diff`] which
/// passes an empty dependency set.
pub(crate) fn diff_with_deps(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    deps: &FxHashSet<ExprId>,
) -> ExprId {
    let var_sym = match arena.node(var) {
        ExprNode::Symbol(sid) => *sid,
        _ => return arena.zero,
    };

    let post_order = crate::base::walk::post_order_ids(arena, expr);
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();

    // Pre-seed the cache: for dependent symbols, their derivative is
    // a formal Derivative node rather than zero.
    for &dep_id in deps {
        let formal = arena.intern(ExprNode::Derivative(dep_id, var));
        cache.insert(dep_id, formal);
    }

    for &id in &post_order {
        if cache.contains_key(&id) {
            continue; // Already seeded (a dependent symbol)
        }
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
        | ExprNode::EulerGamma
        | ExprNode::Catalan
        | ExprNode::GoldenRatio
        | ExprNode::PhysicalConstant(_, _)
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

        // Set-valued nodes → 0 (not differentiable)
        ExprNode::EmptySet
        | ExprNode::UniversalSet
        | ExprNode::Interval(..)
        | ExprNode::FiniteSet(_)
        | ExprNode::SetUnion(_)
        | ExprNode::SetIntersection(_)
        | ExprNode::SetComplement(..) => arena.zero,

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

        // d/dx(H(f)) = δ(f) · f'
        ExprNode::Heaviside(inner) => {
            let df = get_deriv(cache, inner, arena);
            if arena.is_zero_structural(df) {
                return arena.zero;
            }
            let delta = arena.intern(ExprNode::DiracDelta(inner));
            arena.mul(&[delta, df])
        }

        // d/dx(δ(f)) = δ'(f) · f'  — we can't represent δ', so leave unevaluated
        ExprNode::DiracDelta(_) => {
            let v = var_expr(arena, var);
            arena.intern(ExprNode::Derivative(id, v))
        }

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

        // ── Gamma: d/dx(Γ(f)) = Γ(f) · ψ(f) · f' ────────────────
        ExprNode::Gamma(inner) => {
            let df = get_deriv(cache, inner, arena);
            if arena.is_zero_structural(df) {
                return arena.zero;
            }
            let gamma_f = arena.gamma(inner);
            let digamma_f = arena.digamma(inner);
            arena.mul(&[gamma_f, digamma_f, df])
        }

        // ── LogGamma: d/dx(lnΓ(f)) = ψ(f) · f' ──────────────────
        ExprNode::LogGamma(inner) => {
            let df = get_deriv(cache, inner, arena);
            if arena.is_zero_structural(df) {
                return arena.zero;
            }
            let digamma_f = arena.digamma(inner);
            arena.mul(&[digamma_f, df])
        }

        // ── Digamma: d/dx(ψ(f)) = ψ₁(f) · f'  (trigamma = Polygamma(1, f))
        ExprNode::Digamma(inner) => {
            let df = get_deriv(cache, inner, arena);
            if arena.is_zero_structural(df) {
                return arena.zero;
            }
            let one = arena.one;
            let trigamma = arena.polygamma(one, inner);
            arena.mul(&[trigamma, df])
        }

        // ── Polygamma: d/dx(ψ⁽ⁿ⁾(f)) = ψ⁽ⁿ⁺¹⁾(f) · f'  (for constant n)
        ExprNode::Polygamma(n, inner) => {
            let dn = get_deriv(cache, n, arena);
            let df = get_deriv(cache, inner, arena);
            if !arena.is_zero_structural(dn) {
                // Differentiating w.r.t. the order is not elementary.
                let v = var_expr(arena, var);
                return arena.intern(ExprNode::Derivative(id, v));
            }
            if arena.is_zero_structural(df) {
                return arena.zero;
            }
            let one = arena.one;
            let n_plus_1 = arena.add(&[n, one]);
            let next = arena.polygamma(n_plus_1, inner);
            arena.mul(&[next, df])
        }

        // ── Complex analysis ────────────────────────────────────────────
        //
        // re/im/conjugate/arg are not holomorphic, so d/dx only makes sense
        // as a derivative along the real axis.  When the differentiation
        // variable is provably real:
        //   d/dx re(f) = re(f'),  d/dx im(f) = im(f'),
        //   d/dx conj(f) = conj(f'),  d/dx arg(f) = im(f'/f).
        // Otherwise the derivative stays formal.
        ExprNode::Re(inner) => {
            let df = get_deriv(cache, inner, arena);
            if arena.is_zero_structural(df) {
                return arena.zero;
            }
            if !var_is_real(arena, var) {
                let v = var_expr(arena, var);
                return arena.intern(ExprNode::Derivative(id, v));
            }
            arena.re(df)
        }
        ExprNode::Im(inner) => {
            let df = get_deriv(cache, inner, arena);
            if arena.is_zero_structural(df) {
                return arena.zero;
            }
            if !var_is_real(arena, var) {
                let v = var_expr(arena, var);
                return arena.intern(ExprNode::Derivative(id, v));
            }
            arena.im(df)
        }
        ExprNode::Conjugate(inner) => {
            let df = get_deriv(cache, inner, arena);
            if arena.is_zero_structural(df) {
                return arena.zero;
            }
            if !var_is_real(arena, var) {
                let v = var_expr(arena, var);
                return arena.intern(ExprNode::Derivative(id, v));
            }
            arena.conjugate(df)
        }
        ExprNode::Arg(inner) => {
            let df = get_deriv(cache, inner, arena);
            if arena.is_zero_structural(df) {
                return arena.zero;
            }
            if !var_is_real(arena, var) {
                let v = var_expr(arena, var);
                return arena.intern(ExprNode::Derivative(id, v));
            }
            let ratio = arena.div(df, inner);
            arena.im(ratio)
        }

        // ── Si: d/dx Si(f) = sin(f)/f · f' ─────────────────────────────
        ExprNode::Si(inner) => {
            let df = get_deriv(cache, inner, arena);
            if arena.is_zero_structural(df) {
                return arena.zero;
            }
            let sin_f = arena.sin(inner);
            let ratio = arena.div(sin_f, inner);
            arena.mul(&[ratio, df])
        }

        // ── Ci: d/dx Ci(f) = cos(f)/f · f' ─────────────────────────────
        ExprNode::Ci(inner) => {
            let df = get_deriv(cache, inner, arena);
            if arena.is_zero_structural(df) {
                return arena.zero;
            }
            let cos_f = arena.cos(inner);
            let ratio = arena.div(cos_f, inner);
            arena.mul(&[ratio, df])
        }

        // ── Ei: d/dx Ei(f) = e^f/f · f' ───────────────────────────────
        ExprNode::Ei(inner) => {
            let df = get_deriv(cache, inner, arena);
            if arena.is_zero_structural(df) {
                return arena.zero;
            }
            let exp_f = arena.exp(inner);
            let ratio = arena.div(exp_f, inner);
            arena.mul(&[ratio, df])
        }

        // ── li: d/dx li(f) = f' / ln(f) ────────────────────────────────
        ExprNode::Li(inner) => {
            let df = get_deriv(cache, inner, arena);
            if arena.is_zero_structural(df) {
                return arena.zero;
            }
            let ln_f = arena.ln(inner);
            arena.div(df, ln_f)
        }

        // ── ζ: no elementary derivative → formal (0 if argument is constant)
        ExprNode::Zeta(inner) => {
            let df = get_deriv(cache, inner, arena);
            if arena.is_zero_structural(df) {
                return arena.zero;
            }
            let v = var_expr(arena, var);
            arena.intern(ExprNode::Derivative(id, v))
        }

        // ── Kronecker delta: piecewise constant → 0
        ExprNode::KroneckerDelta(_, _) => arena.zero,

        // ── Erf: d/dx(erf(f)) = 2/√π · exp(-f²) · f' ────────────
        ExprNode::Erf(inner) => {
            let df = get_deriv(cache, inner, arena);
            if arena.is_zero_structural(df) {
                return arena.zero;
            }
            let two = arena.int(2);
            let pi = arena.pi;
            let sqrt_pi = arena.sqrt(pi);
            let coeff = arena.div(two, sqrt_pi);
            let two2 = arena.int(2);
            let f_sq = arena.pow(inner, two2);
            let neg_f_sq = arena.neg(f_sq);
            let exp_neg_f_sq = arena.exp(neg_f_sq);
            arena.mul(&[coeff, exp_neg_f_sq, df])
        }

        // ── Erfc: d/dx(erfc(f)) = -2/√π · exp(-f²) · f' ─────────
        ExprNode::Erfc(inner) => {
            let df = get_deriv(cache, inner, arena);
            if arena.is_zero_structural(df) {
                return arena.zero;
            }
            let two = arena.int(2);
            let pi = arena.pi;
            let sqrt_pi = arena.sqrt(pi);
            let coeff = arena.div(two, sqrt_pi);
            let neg_coeff = arena.neg(coeff);
            let two2 = arena.int(2);
            let f_sq = arena.pow(inner, two2);
            let neg_f_sq = arena.neg(f_sq);
            let exp_neg_f_sq = arena.exp(neg_f_sq);
            arena.mul(&[neg_coeff, exp_neg_f_sq, df])
        }

        // ── LambertW: d/dx(W(f)) = W(f) / (f · (1 + W(f))) · f' ──
        ExprNode::LambertW(inner) => {
            let df = get_deriv(cache, inner, arena);
            if arena.is_zero_structural(df) {
                return arena.zero;
            }
            let w_f = arena.lambertw(inner);
            let one = arena.one;
            let one_plus_w = arena.add(&[one, w_f]);
            let denom = arena.mul(&[inner, one_plus_w]);
            let frac = arena.div(w_f, denom);
            arena.mul(&[frac, df])
        }

        // ── Beta: leave as unevaluated derivative ──────────────────
        ExprNode::Beta(_, _) => {
            let v = var_expr(arena, var);
            arena.intern(ExprNode::Derivative(id, v))
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

        // ── Apply (user-defined function): chain rule ──────────────
        // d/dx(f(u₁,…,uₙ)) = Σᵢ (∂f/∂uᵢ) · (duᵢ/dx)
        ExprNode::Apply(func_sym, ref args) => {
            let args_clone = args.clone();

            // Library special functions with known derivative rules
            // (Bessel functions, orthogonal polynomials).
            if let Some(result) = diff_known_apply(arena, func_sym, &args_clone, cache) {
                return result;
            }

            let mut terms: SmallVec<[ExprId; 4]> = SmallVec::new();

            for &arg in &args_clone {
                let d_arg = get_deriv(cache, arg, arena);
                if arena.is_zero_structural(d_arg) {
                    continue;
                }
                // ∂f/∂(arg_i) — stays as formal Derivative since f is unknown
                let partial = arena.intern(ExprNode::Derivative(id, arg));
                terms.push(arena.mul(&[partial, d_arg]));
            }

            if terms.is_empty() {
                // All argument derivatives are zero — f applied to
                // constants is itself a constant.
                arena.zero
            } else if terms.len() == 1 {
                terms[0]
            } else {
                arena.add(&terms)
            }
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

        // ── RootSum: differentiate the body, keep polynomial and sumvar ──
        // d/dx RootSum(q, body(t,x), t) = RootSum(q, d/dx body(t,x), t)
        //
        // The derivative of a sum over roots is the sum of derivatives.
        // The polynomial q and bound variable t are unchanged; only the
        // body (which depends on x) is differentiated.
        ExprNode::RootSum(poly, body, sumvar) => {
            if let ExprNode::Symbol(sum_sym) = arena.node(sumvar)
                && *sum_sym == var
            {
                // Differentiating w.r.t. the bound variable — leave formal.
                let v = var_expr(arena, var);
                arena.intern(ExprNode::Derivative(id, v))
            } else {
                let dbody = get_deriv(cache, body, arena);
                tracing::trace!("diff: RootSum — differentiating body w.r.t. outer variable");
                arena.intern(ExprNode::RootSum(poly, dbody, sumvar))
            }
        }

        // ── Formal / unevaluated nodes: leave as unevaluated derivative ──
        ExprNode::Limit(_, _, _)
        | ExprNode::Series(_, _, _, _)
        | ExprNode::LaplaceTransform(_, _, _)
        | ExprNode::InverseLaplaceTransform(_, _, _)
        | ExprNode::Residue(_, _, _)
        | ExprNode::RootOf(_, _)
        | ExprNode::DSolve(_, _, _)
        | ExprNode::ConditionSet(_, _) => {
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

/// Is the differentiation variable provably real?
///
/// Used by the `re`/`im`/`conjugate`/`arg` rules, which are only valid
/// for derivatives along the real axis.
fn var_is_real(arena: &mut Arena, var: SymbolId) -> bool {
    let v = var_expr(arena, var);
    crate::base::complex::is_real(arena, v) == Some(true)
}

/// Derivative rules for library `Apply` functions whose derivative has a
/// closed form in terms of the same family (chain rule applied):
///
/// * Bessel: `Jᵥ' = (Jᵥ₋₁ − Jᵥ₊₁)/2`, `Yᵥ'` likewise,
///   `Iᵥ' = (Iᵥ₋₁ + Iᵥ₊₁)/2`, `Kᵥ' = −(Kᵥ₋₁ + Kᵥ₊₁)/2`
/// * Legendre: `Pₙ' = n (x Pₙ − Pₙ₋₁) / (x² − 1)`
/// * Chebyshev: `Tₙ' = n Uₙ₋₁`, `Uₙ' = ((n+1) Tₙ₊₁ − x Uₙ) / (x² − 1)`
/// * Hermite: `Hₙ' = 2n Hₙ₋₁`
/// * Laguerre: `Lₙ' = (n Lₙ − n Lₙ₋₁) / x`
///
/// Returns `None` when the function is not one of these, when the
/// order/degree parameter depends on the variable, or when the arity is
/// unexpected — the caller then falls back to a formal derivative.
fn diff_known_apply(
    arena: &mut Arena,
    func_sym: SymbolId,
    args: &[ExprId],
    cache: &FxHashMap<ExprId, ExprId>,
) -> Option<ExprId> {
    use crate::base::arena::{
        FN_BESSELI, FN_BESSELJ, FN_BESSELK, FN_BESSELY, FN_CHEBYSHEV_T, FN_CHEBYSHEV_U, FN_HERMITE,
        FN_LAGUERRE, FN_LEGENDRE,
    };

    if args.len() != 2 {
        return None;
    }
    let name = arena.symbol_name(func_sym).to_owned();
    let (param, x) = (args[0], args[1]);

    // The order / degree must be constant w.r.t. the variable.
    let dparam = get_deriv(cache, param, arena);
    if !arena.is_zero_structural(dparam) {
        return None;
    }
    let dx = get_deriv(cache, x, arena);
    if arena.is_zero_structural(dx) {
        return Some(arena.zero);
    }

    let one = arena.one;
    let two = arena.int(2);
    let half = arena.rational(1, 2);
    let p_minus_1 = arena.sub(param, one);
    let p_plus_1 = arena.add(&[param, one]);

    let outer = match name.as_str() {
        FN_BESSELJ => {
            let a = arena.besselj(p_minus_1, x);
            let b = arena.besselj(p_plus_1, x);
            let d = arena.sub(a, b);
            arena.mul(&[half, d])
        }
        FN_BESSELY => {
            let a = arena.bessely(p_minus_1, x);
            let b = arena.bessely(p_plus_1, x);
            let d = arena.sub(a, b);
            arena.mul(&[half, d])
        }
        FN_BESSELI => {
            let a = arena.besseli(p_minus_1, x);
            let b = arena.besseli(p_plus_1, x);
            let s = arena.add(&[a, b]);
            arena.mul(&[half, s])
        }
        FN_BESSELK => {
            let a = arena.besselk(p_minus_1, x);
            let b = arena.besselk(p_plus_1, x);
            let s = arena.add(&[a, b]);
            let neg_half = arena.rational(-1, 2);
            arena.mul(&[neg_half, s])
        }
        FN_LEGENDRE => {
            // Pₙ' = n (x Pₙ − Pₙ₋₁) / (x² − 1)
            let pn = arena.legendre(param, x);
            let pn_1 = arena.legendre(p_minus_1, x);
            let x_pn = arena.mul(&[x, pn]);
            let numer_inner = arena.sub(x_pn, pn_1);
            let numer = arena.mul(&[param, numer_inner]);
            let x2 = arena.pow(x, two);
            let denom = arena.sub(x2, one);
            arena.div(numer, denom)
        }
        FN_CHEBYSHEV_T => {
            // Tₙ' = n Uₙ₋₁
            let u = arena.chebyshev_u(p_minus_1, x);
            arena.mul(&[param, u])
        }
        FN_CHEBYSHEV_U => {
            // Uₙ' = ((n+1) Tₙ₊₁ − x Uₙ) / (x² − 1)
            let t = arena.chebyshev_t(p_plus_1, x);
            let un = arena.chebyshev_u(param, x);
            let a = arena.mul(&[p_plus_1, t]);
            let b = arena.mul(&[x, un]);
            let numer = arena.sub(a, b);
            let x2 = arena.pow(x, two);
            let denom = arena.sub(x2, one);
            arena.div(numer, denom)
        }
        FN_HERMITE => {
            // Hₙ' = 2n Hₙ₋₁
            let h = arena.hermite(p_minus_1, x);
            arena.mul(&[two, param, h])
        }
        FN_LAGUERRE => {
            // Lₙ' = (n Lₙ − n Lₙ₋₁) / x
            let ln_ = arena.laguerre(param, x);
            let ln_1 = arena.laguerre(p_minus_1, x);
            let d = arena.sub(ln_, ln_1);
            let numer = arena.mul(&[param, d]);
            arena.div(numer, x)
        }
        _ => return None,
    };

    Some(arena.mul(&[outer, dx]))
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
        let integral = a.intern(crate::base::node::ExprNode::Integral(x2, x));
        let result = diff(&mut a, integral, x);
        assert_eq!(display(&a, result), "x^2");
    }

    #[test]
    fn diff_integral_different_var_stays_unevaluated() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        // d/dx(∫ y dy) stays as Derivative(Integral(y, y), x)
        let integral = a.intern(crate::base::node::ExprNode::Integral(y, y));
        let result = diff(&mut a, integral, x);
        assert_eq!(display(&a, result), "Derivative(Integral(y, y), x)");
    }

    // ── Apply chain rule (Wave C) ───────────────────────────────────

    #[test]
    fn diff_apply_constant_arg_is_zero() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let three = a.int(3);
        let f_sid = a.symbols.intern("f");
        let apply = a.intern(ExprNode::Apply(f_sid, smallvec::smallvec![three]));
        let result = diff(&mut a, apply, x);
        assert_eq!(result, a.zero, "d/dx(f(3)) should be 0");
    }

    #[test]
    fn diff_apply_identity_is_formal_derivative() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let f_sid = a.symbols.intern("f");
        let apply = a.intern(ExprNode::Apply(f_sid, smallvec::smallvec![x]));
        let result = diff(&mut a, apply, x);
        let s = display(&a, result);
        // d/dx(f(x)) = Derivative(f(x), x) · 1 = Derivative(f(x), x)
        assert!(
            s.contains("Derivative") && s.contains("f(x)"),
            "d/dx(f(x)) should be Derivative(f(x), x), got: {s}"
        );
    }

    #[test]
    fn diff_apply_chain_rule_x_squared() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let x_sq = a.pow(x, two);
        let f_sid = a.symbols.intern("f");
        let apply = a.intern(ExprNode::Apply(f_sid, smallvec::smallvec![x_sq]));
        let result = diff(&mut a, apply, x);
        let s = display(&a, result);
        // Should contain 2, x, and Derivative
        assert!(
            s.contains('2') && s.contains('x') && s.contains("Derivative"),
            "d/dx(f(x²)) should contain 2, x, Derivative, got: {s}"
        );
    }

    #[test]
    fn diff_apply_chain_sin_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let sin_x = a.sin(x);
        let f_sid = a.symbols.intern("f");
        let apply = a.intern(ExprNode::Apply(f_sid, smallvec::smallvec![sin_x]));
        let result = diff(&mut a, apply, x);
        let s = display(&a, result);
        assert!(
            s.contains("cos"),
            "d/dx(f(sin(x))) should contain cos(x), got: {s}"
        );
    }

    #[test]
    fn diff_apply_other_symbol_is_zero() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let f_sid = a.symbols.intern("f");
        let apply = a.intern(ExprNode::Apply(f_sid, smallvec::smallvec![y]));
        let result = diff(&mut a, apply, x);
        assert_eq!(result, a.zero, "d/dx(f(y)) should be 0");
    }

    #[test]
    fn diff_known_apply_fibonacci_constant() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let ten = a.int(10);
        let fib = a.fibonacci(ten);
        let result = diff(&mut a, fib, x);
        assert_eq!(result, a.zero, "d/dx(fibonacci(10)) should be 0");
    }

    #[test]
    fn diff_known_apply_fibonacci_variable() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let fib = a.fibonacci(x);
        let result = diff(&mut a, fib, x);
        let s = display(&a, result);
        assert!(
            s.contains("Derivative") && s.contains("fibonacci"),
            "d/dx(fibonacci(x)) should be formal Derivative, got: {s}"
        );
    }

    #[test]
    fn diff_apply_multiarg_chain_rule() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let x_sq = a.pow(x, two);
        let f_sid = a.symbols.intern("f");
        // f(x, x²)
        let apply = a.intern(ExprNode::Apply(f_sid, smallvec::smallvec![x, x_sq]));
        let result = diff(&mut a, apply, x);
        let s = display(&a, result);
        assert!(
            s.contains("Derivative"),
            "d/dx(f(x, x²)) should contain Derivative terms, got: {s}"
        );
    }

    #[test]
    fn diff_apply_all_constant_args_is_zero() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let five = a.int(5);
        let seven = a.int(7);
        let g_sid = a.symbols.intern("g");
        let apply = a.intern(ExprNode::Apply(g_sid, smallvec::smallvec![five, seven]));
        let result = diff(&mut a, apply, x);
        assert_eq!(result, a.zero, "d/dx(g(5,7)) should be 0");
    }

    #[test]
    fn diff_apply_no_args_is_zero() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let h_sid = a.symbols.intern("h");
        let apply = a.intern(ExprNode::Apply(h_sid, smallvec::smallvec![]));
        let result = diff(&mut a, apply, x);
        assert_eq!(result, a.zero, "d/dx(h()) should be 0");
    }

    // ── Dependency-aware differentiation (Wave D) ───────────────────

    #[test]
    fn diff_with_deps_y_depends_on_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let mut deps = FxHashSet::default();
        deps.insert(y);
        let result = diff_with_deps(&mut a, y, x, &deps);
        let s = display(&a, result);
        assert!(
            s.contains("Derivative") && s.contains('y') && s.contains('x'),
            "d/dx(y) with deps={{y}} should be Derivative(y, x), got: {s}"
        );
    }

    #[test]
    fn diff_with_deps_y_squared() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let two = a.int(2);
        let y_sq = a.pow(y, two);
        let mut deps = FxHashSet::default();
        deps.insert(y);
        let result = diff_with_deps(&mut a, y_sq, x, &deps);
        let s = display(&a, result);
        assert!(
            s.contains('2') && s.contains('y') && s.contains("Derivative"),
            "d/dx(y²) with deps={{y}} should be 2*y*Derivative(y,x), got: {s}"
        );
    }

    #[test]
    fn diff_with_deps_implicit() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let two = a.int(2);
        let x_sq = a.pow(x, two);
        let y_sq = a.pow(y, two);
        let expr = a.add(&[x_sq, y_sq]);
        let mut deps = FxHashSet::default();
        deps.insert(y);
        let result = diff_with_deps(&mut a, expr, x, &deps);
        let s = display(&a, result);
        assert!(
            s.contains('2') && s.contains('x') && s.contains("Derivative"),
            "d/dx(x²+y²) with deps={{y}} should have 2*x and Derivative, got: {s}"
        );
    }

    #[test]
    fn diff_with_deps_sin_y() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let sin_y = a.sin(y);
        let mut deps = FxHashSet::default();
        deps.insert(y);
        let result = diff_with_deps(&mut a, sin_y, x, &deps);
        let s = display(&a, result);
        assert!(
            s.contains("cos") && s.contains("Derivative"),
            "d/dx(sin(y)) with deps={{y}} should contain cos and Derivative, got: {s}"
        );
    }

    #[test]
    fn diff_with_deps_product_xy() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let xy = a.mul(&[x, y]);
        let mut deps = FxHashSet::default();
        deps.insert(y);
        let result = diff_with_deps(&mut a, xy, x, &deps);
        let s = display(&a, result);
        assert!(
            s.contains('y') && s.contains("Derivative"),
            "d/dx(x*y) with deps={{y}} should contain y and Derivative, got: {s}"
        );
    }

    #[test]
    fn diff_with_deps_empty_is_zero() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let deps = FxHashSet::default();
        let result = diff_with_deps(&mut a, y, x, &deps);
        assert_eq!(result, a.zero, "d/dx(y) with empty deps should be 0");
    }

    #[test]
    fn diff_with_deps_matches_plain_diff() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let x_sq = a.pow(x, two);
        let sin_x2 = a.sin(x_sq);
        let deps = FxHashSet::default();
        let r1 = diff_with_deps(&mut a, sin_x2, x, &deps);
        let r2 = diff(&mut a, sin_x2, x);
        assert_eq!(r1, r2, "diff_with_deps with empty deps should match diff");
    }

    #[test]
    fn diff_with_deps_var_not_symbol_returns_zero() {
        let mut a = Arena::new();
        let three = a.int(3);
        let y = sym(&mut a, "y");
        let deps = FxHashSet::default();
        // var is a number, not a symbol — should return zero
        let result = diff_with_deps(&mut a, y, three, &deps);
        assert_eq!(result, a.zero);
    }

    // ── 0.2 nodes ────────────────────────────────────────────────────────────

    fn real_sym(a: &mut Arena, name: &str) -> ExprId {
        use crate::base::assumptions::{Assumptions, Props};
        let id = a.symbol(name);
        if let ExprNode::Symbol(sid) = a.node(id).clone() {
            let mut asm = Assumptions::default();
            asm.assert_true(Props::REAL);
            a.set_symbol_assumptions(sid, asm);
        }
        id
    }

    #[test]
    fn diff_named_constants_are_zero() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        for c in [a.euler_gamma, a.catalan, a.golden_ratio] {
            assert_eq!(diff(&mut a, c, x), a.zero);
        }
    }

    /// Differentiate and render.
    fn dd(a: &mut Arena, f: ExprId, x: ExprId) -> String {
        let d = diff(a, f, x);
        display(a, d)
    }

    #[test]
    fn diff_special_functions() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let si = a.si(x);
        assert_eq!(dd(&mut a, si, x), "sin(x)/x");
        let ci = a.ci(x);
        assert_eq!(dd(&mut a, ci, x), "cos(x)/x");
        let ei = a.ei(x);
        assert_eq!(dd(&mut a, ei, x), "exp(x)/x");
        let li = a.li(x);
        assert_eq!(dd(&mut a, li, x), "1/ln(x)");
        let dg = a.digamma(x);
        assert_eq!(dd(&mut a, dg, x), "polygamma(1, x)");
        let two = a.int(2);
        let pg = a.polygamma(two, x);
        assert_eq!(dd(&mut a, pg, x), "polygamma(3, x)");
        let z = a.zeta(x);
        let dz = diff(&mut a, z, x);
        assert!(matches!(a.node(dz), ExprNode::Derivative(_, _)));
        let y = sym(&mut a, "y");
        let kd = a.kronecker_delta(x, y);
        assert_eq!(diff(&mut a, kd, x), a.zero);
        // chain rule through Si(x²)
        let x2 = a.pow(x, two);
        let si_x2 = a.si(x2);
        assert_eq!(dd(&mut a, si_x2, x), "2*sin(x^2)/x");
    }

    #[test]
    fn diff_complex_nodes_require_real_variable() {
        let mut a = Arena::new();
        let t = real_sym(&mut a, "t");
        let z = sym(&mut a, "z");
        let two = a.int(2);
        let t2 = a.pow(t, two);
        let f = a.mul(&[z, t2]);
        let re_f = a.re(f);
        assert_eq!(dd(&mut a, re_f, t), "2*t*re(z)");
        let im_f = a.im(f);
        assert_eq!(dd(&mut a, im_f, t), "2*t*im(z)");
        let cf = a.conjugate(f);
        assert_eq!(dd(&mut a, cf, t), "2*t*conjugate(z)");
        // arg(z·e^t)' = im(1) = 0
        let et = a.exp(t);
        let g = a.mul(&[z, et]);
        let ag = a.arg(g);
        assert_eq!(diff(&mut a, ag, t), a.zero);
        // Non-real variable → formal derivative.
        let x = sym(&mut a, "x");
        let re_x = a.re(x);
        let d = diff(&mut a, re_x, x);
        assert!(matches!(a.node(d), ExprNode::Derivative(_, _)));
    }

    #[test]
    fn diff_bessel_and_orthogonal_apply() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let nu = sym(&mut a, "nu");
        let j = a.besselj(nu, x);
        let dj = diff(&mut a, j, x);
        let half = a.rational(1, 2);
        let nm1 = a.sub(nu, a.one);
        let np1 = a.add(&[nu, a.one]);
        let jm = a.besselj(nm1, x);
        let jp = a.besselj(np1, x);
        let diffj = a.sub(jm, jp);
        let expected = a.mul(&[half, diffj]);
        assert_eq!(dj, expected);
        let k = a.besselk(nu, x);
        let dk = diff(&mut a, k, x);
        let km = a.besselk(nm1, x);
        let kp = a.besselk(np1, x);
        let sumk = a.add(&[km, kp]);
        let neg_half = a.rational(-1, 2);
        let expected = a.mul(&[neg_half, sumk]);
        assert_eq!(dk, expected);
        let n = sym(&mut a, "n");
        let t = a.chebyshev_t(n, x);
        assert_eq!(dd(&mut a, t, x), "n*chebyshev_u(n - 1, x)");
        let h = a.hermite(n, x);
        assert_eq!(dd(&mut a, h, x), "2*n*hermite(n - 1, x)");
        // Order depending on the variable falls back to a formal derivative.
        let jx = a.besselj(x, x);
        let d = diff(&mut a, jx, x);
        assert!(crate::base::walk::has_unevaluated(&a, d));
    }
}

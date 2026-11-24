//! ODE solution verification (part of [`ode`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// ODE solution verification
// ═══════════════════════════════════════════════════════════════════════════

/// Check whether a solution satisfies an ODE.
///
/// Substitutes the solution for `func`, differentiates as needed,
/// and checks if the ODE expression evaluates to zero: by expansion, the
/// full simplifier, and three normal forms (radical bases factored out at
/// their lowest exponent; a rational function of `sin`, `cos` over a common
/// denominator with `sin² = 1 − cos²`; a rational function after
/// `e^{W(z)} = z/W(z)` for every branch of Lambert W).  Every route is a chain of
/// identities, so `true` is a proof; `false` means no route found zero (not
/// necessarily a wrong solution).  There is no numerical fallback.
///
/// The ODE is given as `ode_expr = 0`.
pub fn checkodesol(
    arena: &mut Arena,
    ode_expr: ExprId,
    solution: ExprId,
    func: ExprId,
    var: ExprId,
) -> bool {
    // Derivative nodes y, y', y'', … up to the highest order present.
    let chain = derivative_chain(arena, func, var);
    let max_order = (1..chain.len())
        .rev()
        .find(|&k| expr_contains(arena, ode_expr, chain[k]))
        .unwrap_or(0);

    // Derivatives of the solution.
    let mut sol_derivs = vec![solution];
    for k in 1..=max_order {
        let d = crate::transforms::diff::diff(arena, sol_derivs[k - 1], var);
        sol_derivs.push(d);
    }

    // Substitute the highest derivative first (more specific), down to func.
    let mut result = ode_expr;
    for k in (0..=max_order).rev() {
        result = crate::transforms::subs::subs(arena, result, chain[k], sol_derivs[k]);
    }

    // Evaluate and simplify
    result = crate::transforms::eval::eval(arena, result);
    result = crate::transforms::expand::expand(arena, result);
    result = crate::transforms::eval::eval(arena, result);

    if result == arena.zero {
        return true;
    }

    // Try another round of expand + eval for stubborn expressions
    result = crate::transforms::expand::expand(arena, result);
    result = crate::transforms::eval::eval(arena, result);
    if result == arena.zero {
        return true;
    }

    // The full simplifier.
    let simplified = crate::simplify::simplify_engine::unified_simplify(
        arena,
        result,
        &crate::simplify::simplify_engine::SimplifyOpts::default(),
    );
    if arena.is_zero_structural(simplified.expr) {
        return true;
    }

    // Normal forms the simplifier misses; each step is an identity or a
    // multiplication by a non-zero factor, so a zero here is a proof.
    radical_normal_form_is_zero(arena, result)
        || trig_normal_form_is_zero(arena, result)
        || lambert_normal_form_is_zero(arena, result)
}

/// The argument `z` of a Lambert W value `W_k(z)` (any branch), or `None`.
fn lambert_argument(arena: &Arena, w: ExprId) -> Option<ExprId> {
    match arena.node(w) {
        ExprNode::LambertW(z) => Some(*z),
        ExprNode::Lib(crate::base::libfn::LibFn::LambertW, args) => match args[..] {
            [z] => Some(z),
            [z, k] if arena.as_num(k).is_some_and(|q| q.is_integer()) => Some(z),
            _ => None,
        },
        _ => None,
    }
}

/// `e` with every `e^{k·W(z) + s}` (`k` an integer, `W` any branch of
/// Lambert W) written `(z/W(z))^k·e^s`; `None` when there is no such
/// exponential.  Every branch solves `W·e^W = z`, so `e^{W(z)} = z/W(z)`
/// wherever `W(z) ≠ 0`, that is off `z = 0` (only `W₀(0)` is 0), and an
/// integer power of it is single-valued.
fn exp_lambert_to_ratio(arena: &mut Arena, e: ExprId) -> Option<ExprId> {
    // Integer exponents beyond this are left alone (no blow-up of powers).
    const MAX_POWER: i64 = 8;
    let post = crate::base::walk::post_order_ids(arena, e);
    let mut cache: rustc_hash::FxHashMap<ExprId, ExprId> = rustc_hash::FxHashMap::default();
    let mut changed = false;
    for &id in &post {
        let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
        let new = match *arena.node(rebuilt) {
            ExprNode::Exp(arg) => {
                let terms: Vec<ExprId> = match arena.node(arg) {
                    ExprNode::Add(c) => c.to_vec(),
                    _ => vec![arg],
                };
                let mut factors = Vec::new();
                let mut rest = Vec::new();
                for t in terms {
                    let (k, term) = arena.as_coeff_term(t);
                    let power = k
                        .is_integer()
                        .then(|| num_traits::ToPrimitive::to_i64(&k.to_integer()))
                        .flatten()
                        .filter(|p| p.abs() <= MAX_POWER);
                    match (lambert_argument(arena, term), power) {
                        (Some(z), Some(p)) => {
                            let ratio = arena.div(z, term);
                            let pe = arena.int(p);
                            factors.push(arena.pow(ratio, pe));
                        }
                        _ => rest.push(t),
                    }
                }
                if factors.is_empty() {
                    rebuilt
                } else {
                    changed = true;
                    if !rest.is_empty() {
                        let r = arena.add(&rest);
                        factors.push(arena.exp(r));
                    }
                    arena.mul(&factors)
                }
            }
            _ => rebuilt,
        };
        cache.insert(id, new);
    }
    if !changed {
        return None;
    }
    let out = cache.get(&e).copied().unwrap_or(e);
    Some(crate::transforms::eval::eval(arena, out))
}

/// Is the residual `r` zero once `e^{W(z)}` is written `z/W(z)`
/// ([`exp_lambert_to_ratio`]), as a rational function or with radicals
/// factored out?  The residual is then zero wherever `z ≠ 0`, a dense set,
/// hence everywhere it is defined.  Before, the correct solution
/// `C1·e^{W(2x/C1)} + x` of `y′ = (3y − x)/(x + y)` was rejected: its
/// residual `2·e^W·W/(z·(1 + W)) + 1 − (3y − x)/(x + y)` is zero only
/// through `e^W = z/W` (it gives `2/(1 + W) + 1 − (3 + W)/(1 + W)`), so
/// `dsolve` fell back to the implicit solution.
fn lambert_normal_form_is_zero(arena: &mut Arena, r: ExprId) -> bool {
    let Some(s) = exp_lambert_to_ratio(arena, r) else {
        return false;
    };
    arena.is_zero_structural(s)
        || trig_normal_form_is_zero(arena, s)
        || radical_normal_form_is_zero(arena, s)
}

/// Most terms or radical bases [`radical_normal_form_is_zero`] handles.
const MAX_RADICAL_TERMS: usize = 64;

/// A term of an expanded residual: its other factors, and its radicals as
/// (normalised base, exponent).
type TermRadicals = (Vec<ExprId>, Vec<(ExprId, Q)>);

/// `expand`, `eval`, then combine powers of equal bases (`e^{−2x}·e^{2x} =
/// 1`, which canonical products keep apart).
pub(super) fn expand_combine(arena: &mut Arena, e: ExprId) -> ExprId {
    let e = crate::transforms::expand::expand(arena, e);
    let e = crate::transforms::eval::eval(arena, e);
    let e = crate::simplify::powsimp::powsimp(arena, e);
    crate::transforms::eval::eval(arena, e)
}

/// Is the residual `r` zero once every radical base `B` (a power `B^q`
/// with a non-integer rational `q`) is factored out at its lowest exponent?
///
/// The expanded residual `Σ tᵢ·Π B_j^{q_ij}` is multiplied by
/// `Π B_j^{−min_i q_ij}`: `B^a·B^b = B^{a+b}` holds for principal powers
/// (`B^a = e^{a·Log B}`, `B ≠ 0`), the remaining exponents are non-negative
/// integers, and the product with a non-zero factor is zero exactly when
/// the residual is.  Before, the correct solution
/// `((C1 + e^{−2x})·e^{2x})^{−1/2}` of the Bernoulli equation `y′ + y = y³`
/// was rejected: the residual `−C1e^{2x}B^{−3/2} + B^{−1/2} − B^{−3/2}`
/// is zero only through `B^{−1/2} = B·B^{−3/2}`.
fn radical_normal_form_is_zero(arena: &mut Arena, r: ExprId) -> bool {
    let r = expand_combine(arena, r);
    let terms: Vec<ExprId> = match arena.node(r) {
        ExprNode::Add(cs) => cs.to_vec(),
        _ => vec![r],
    };
    if terms.len() > MAX_RADICAL_TERMS {
        return false;
    }
    // Per term: the other factors and the radicals (normalised base, exponent).
    let mut split: Vec<TermRadicals> = Vec::with_capacity(terms.len());
    let mut bases: Vec<ExprId> = Vec::new();
    for &t in &terms {
        let factors: Vec<ExprId> = match arena.node(t) {
            ExprNode::Mul(fs) => fs.to_vec(),
            _ => vec![t],
        };
        let mut rest = Vec::new();
        let mut rads = Vec::new();
        for f in factors {
            match arena.node(f).clone() {
                ExprNode::Pow(b, e)
                    if arena.as_num(e).is_some_and(|q| !q.is_integer())
                        && !matches!(arena.node(b), ExprNode::Num(_)) =>
                {
                    let q = arena.as_num(e).cloned().unwrap_or_default();
                    let nb = expand_combine(arena, b);
                    if !bases.contains(&nb) {
                        bases.push(nb);
                    }
                    rads.push((nb, q));
                }
                _ => rest.push(f),
            }
        }
        split.push((rest, rads));
    }
    if bases.is_empty() || bases.len() > MAX_RADICAL_TERMS {
        return false;
    }
    let exponent = |rads: &[(ExprId, Q)], b: ExprId| -> Q {
        rads.iter()
            .filter(|(nb, _)| *nb == b)
            .map(|(_, q)| q.clone())
            .sum()
    };
    let mut minima = Vec::with_capacity(bases.len());
    for &b in &bases {
        let mut m: Option<Q> = None;
        for (_, rads) in &split {
            let e = exponent(rads, b);
            m = Some(match m {
                Some(v) if v <= e => v,
                _ => e,
            });
        }
        minima.push(m.unwrap_or_default());
    }
    let mut new_terms = Vec::with_capacity(split.len());
    for (rest, rads) in &split {
        let mut fs = rest.clone();
        for (&b, m) in bases.iter().zip(&minima) {
            let k = exponent(rads, b) - m;
            if !k.is_integer() {
                return false;
            }
            if num_traits::Zero::is_zero(&k) {
                continue;
            }
            let Some(ki) = num_traits::ToPrimitive::to_i64(&k.to_integer()) else {
                return false;
            };
            if ki > 8 {
                return false;
            }
            let ke = arena.int(ki);
            fs.push(arena.pow(b, ke));
        }
        new_terms.push(arena.mul(&fs));
    }
    let s = arena.add(&new_terms);
    let s = expand_combine(arena, s);
    arena.is_zero_structural(s) || trig_normal_form_is_zero(arena, s)
}

/// Is the residual `r` zero as a rational function (of `sin` and `cos`,
/// among other things)?
///
/// Over a common denominator the numerator must expand to zero — which also
/// covers rational functions of the parameters that the simplifier leaves
/// apart (`eˣ·(a + 1)/(a·(1/a + 1)) − eˣ`, the residual of the particular
/// solution of `y″ + a·y = eˣ`).  For trigonometric residuals:
/// `tan u → sin u/cos u`, the sum over a common denominator, and in the
/// expanded numerator `sin²u → 1 − cos²u` until no square of a sine is
/// left: a polynomial in `sin u`, `cos u` vanishes on the circle exactly
/// when this reduction gives zero.  Before, the variation-of-parameters
/// solution `−cos x·ln|sec x + tan x| + 2 sin x` of `y″ + y = tan x` was
/// rejected: its residual is zero only through `sin² + cos² = 1` over the
/// denominator `(sec x + tan x)²`.
fn trig_normal_form_is_zero(arena: &mut Arena, r: ExprId) -> bool {
    let post = crate::base::walk::post_order_ids(arena, r);
    let mut cache: rustc_hash::FxHashMap<ExprId, ExprId> = rustc_hash::FxHashMap::default();
    for &id in &post {
        let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
        let new = match arena.node(rebuilt).clone() {
            ExprNode::Tan(u) => {
                let s = arena.sin(u);
                let c = arena.cos(u);
                arena.div(s, c)
            }
            _ => rebuilt,
        };
        cache.insert(id, new);
    }
    let t = cache.get(&r).copied().unwrap_or(r);
    let t = crate::transforms::eval::eval(arena, t);
    let t = crate::poly::polybridge::together(arena, t);
    let (num, _) = crate::poly::polybridge::as_numer_denom(arena, t);
    let mut num = expand_combine(arena, num);
    for _ in 0..6 {
        if arena.is_zero_structural(num) {
            return true;
        }
        let post = crate::base::walk::post_order_ids(arena, num);
        let mut cache: rustc_hash::FxHashMap<ExprId, ExprId> = rustc_hash::FxHashMap::default();
        let mut changed = false;
        for &id in &post {
            let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
            let new = match arena.node(rebuilt).clone() {
                ExprNode::Pow(b, e) => {
                    let k = arena.as_num(e).and_then(|q| {
                        q.is_integer()
                            .then(|| num_traits::ToPrimitive::to_i64(&q.to_integer()))
                            .flatten()
                    });
                    match (arena.node(b).clone(), k) {
                        (ExprNode::Sin(u), Some(k)) if (2..=64).contains(&k) => {
                            changed = true;
                            let c = arena.cos(u);
                            let two = arena.int(2);
                            let c2 = arena.pow(c, two);
                            let one = arena.one;
                            let one_minus = arena.sub(one, c2);
                            let half = arena.int(k / 2);
                            let p = arena.pow(one_minus, half);
                            if k % 2 == 1 { arena.mul(&[b, p]) } else { p }
                        }
                        _ => rebuilt,
                    }
                }
                _ => rebuilt,
            };
            cache.insert(id, new);
        }
        if !changed {
            return false;
        }
        let n = cache.get(&num).copied().unwrap_or(num);
        num = expand_combine(arena, n);
    }
    arena.is_zero_structural(num)
}

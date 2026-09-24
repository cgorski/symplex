//! Symbolic Fourier transform and inverse Fourier transform (table + rules).
//!
//! The public entry points are the `Ex` methods `fourier_transform`,
//! `inverse_fourier_transform`, `fourier_transform_with` and
//! `inverse_fourier_transform_with`; the convention is selected with
//! [`FourierConvention`].
//!
//! # Conventions
//!
//! | [`FourierConvention`] | Forward `F(ω) =`                        | Inverse `f(t) =`                       |
//! |-----------------------|-----------------------------------------|----------------------------------------|
//! | `NonUnitaryAngular`   | `∫ f(t) e^{−iωt} dt`                    | `(1/2π) ∫ F(ω) e^{iωt} dω`             |
//! | `UnitaryAngular`      | `(1/√(2π)) ∫ f(t) e^{−iωt} dt`          | `(1/√(2π)) ∫ F(ω) e^{iωt} dω`          |
//! | `Ordinary`            | `∫ f(t) e^{−2πiνt} dt`                  | `∫ F(ν) e^{2πiνt} dν`                  |
//!
//! Everything is computed in the non-unitary angular convention and then
//! converted: `F_unitary = F/√(2π)` and `F_ordinary(ν) = F(2πν)`.
//!
//! # Table (non-unitary angular)
//!
//! | `f(t)`                         | `F(ω)`                                   | condition   |
//! |--------------------------------|------------------------------------------|-------------|
//! | `δ(t)`                         | `1`                                      |             |
//! | `1`                            | `2π δ(ω)`                                |             |
//! | `H(t)`                         | `π δ(ω) + 1/(iω)`                        |             |
//! | `sign(t)`                      | `2/(iω)`                                 |             |
//! | `1/t`                          | `−iπ sign(ω)`                            |             |
//! | `\|t\|`                        | `−2/ω²`                                  |             |
//! | `H(t+a) − H(t−a)` (rect)       | `2 sin(aω)/ω`                            |             |
//! | `e^{−a\|t\|}`                  | `2a/(a² + ω²)`                           | `a > 0`     |
//! | `e^{−at²}`                     | `√(π/a) e^{−ω²/(4a)}`                    | `a > 0`     |
//! | `tⁿ e^{−at} H(t)`              | `n!/(iω + a)^{n+1}`                      | `a > 0`     |
//! | `tⁿ e^{at} H(−t)`              | `(−1)ⁿ n!/(a − iω)^{n+1}`                | `a > 0`     |
//! | `cos(ω₀t)`, `sin(ω₀t)`         | `π[δ(ω−ω₀) + δ(ω+ω₀)]`, `iπ[δ(ω+ω₀) − δ(ω−ω₀)]` | |
//! | `sin(at)/t` (sinc)             | `π [H(ω+a) − H(ω−a)]`                    | `a > 0`     |
//!
//! Rules: linearity, time shift `f(t−t₀) → e^{−iωt₀}F(ω)`, modulation
//! `e^{iω₀t} f(t) → F(ω−ω₀)`, scaling `f(at) → F(ω/a)/|a|`, derivative
//! `f'(t) → iω F(ω)`, and multiplication by `t`: `t f(t) → i F'(ω)`.
//! `Piecewise` inputs with linear conditions in `t` are converted to
//! Heaviside form first.
//!
//! Symbols other than the transform variables are treated as **real**
//! parameters; conditions such as `a > 0` are checked through the
//! assumption system (`Assumption::Positive`) and an unprovable condition is
//! an error rather than a guess.
//!
//! Results containing `δ(ω − c)·g(ω)` are simplified to `δ(ω − c)·g(c)`.

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{Signed, ToPrimitive, Zero};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::base::arena::Arena;
use crate::base::errors::SymplexError;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::Q;

/// Maximum nesting of rule applications (shift → scale → shift …).
const MAX_RULE_DEPTH: usize = 6;

/// Normalisation convention of the Fourier transform pair.
///
/// See the [module documentation](self) for the defining integrals.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::fourier_transform::FourierConvention;
///
/// let ctx = Context::new();
/// let t = ctx.symbol("t");
/// let w = ctx.symbol("w");
/// let f = (-t.abs()).exp();
/// // Non-unitary angular (default): 2/(1 + w²)
/// let fa = f.fourier_transform(&t, &w).unwrap();
/// assert_eq!(format!("{fa}"), "2/(w^2 + 1)");
/// // Unitary angular: √(2/π)/(1 + w²)
/// let fu = f
///     .fourier_transform_with(&t, &w, FourierConvention::UnitaryAngular)
///     .unwrap();
/// let ratio = (&fu / &fa).subs(&w, &ctx.int(1)).eval_f64().unwrap();
/// assert!((ratio - 1.0 / (2.0 * std::f64::consts::PI).sqrt()).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FourierConvention {
    /// `F(ω) = ∫ f(t) e^{−iωt} dt`, `f(t) = (1/2π) ∫ F(ω) e^{iωt} dω`.
    /// The convention of most engineering and physics texts; the default.
    #[default]
    NonUnitaryAngular,
    /// `F(ω) = (1/√(2π)) ∫ f(t) e^{−iωt} dt` with the symmetric inverse.
    UnitaryAngular,
    /// `F(ν) = ∫ f(t) e^{−2πiνt} dt`, `f(t) = ∫ F(ν) e^{2πiνt} dν`
    /// (ordinary frequency `ν` in hertz).
    Ordinary,
}

fn fail(operation: &'static str, reason: impl Into<String>) -> SymplexError {
    SymplexError::ComputationFailed {
        operation,
        reason: reason.into(),
    }
}

fn require_symbol(
    arena: &Arena,
    id: ExprId,
    operation: &'static str,
    what: &str,
) -> Result<(), SymplexError> {
    if matches!(arena.node(id), ExprNode::Symbol(_)) {
        Ok(())
    } else {
        Err(SymplexError::InvalidArgument {
            operation,
            reason: format!("{what} must be a symbol, got {}", arena.display(id)),
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Public (crate) entry points
// ═══════════════════════════════════════════════════════════════════════════

/// Fourier transform of `expr` (a function of `t`) as a function of
/// `omega`, in the non-unitary angular convention.
pub(crate) fn fourier_transform(
    arena: &mut Arena,
    expr: ExprId,
    t: ExprId,
    omega: ExprId,
) -> Result<ExprId, SymplexError> {
    fourier_transform_with(arena, expr, t, omega, FourierConvention::NonUnitaryAngular)
}

/// Fourier transform of `expr` in the given convention.
pub(crate) fn fourier_transform_with(
    arena: &mut Arena,
    expr: ExprId,
    t: ExprId,
    omega: ExprId,
    convention: FourierConvention,
) -> Result<ExprId, SymplexError> {
    require_symbol(arena, t, "fourier_transform", "the time variable")?;
    require_symbol(arena, omega, "fourier_transform", "the frequency variable")?;
    if t == omega {
        return Err(SymplexError::InvalidArgument {
            operation: "fourier_transform",
            reason: "time and frequency variables must be distinct".into(),
        });
    }
    // `Piecewise` must be eliminated *before* any `eval`: `eval` currently
    // selects the first branch whose condition is literally `true` even when
    // an earlier condition is undecided (`Piecewise(1 if |t| < 1, 0 if true)`
    // evaluates to `0`).
    let expr = eliminate_piecewise(arena, expr, t, "fourier_transform")?;
    let f = forward(arena, expr, t, omega, 0)?;
    let f = finish(arena, f, omega);
    let f = match convention {
        FourierConvention::NonUnitaryAngular => f,
        FourierConvention::UnitaryAngular => {
            let inv_root = inv_sqrt_two_pi(arena);
            let g = arena.mul(&[inv_root, f]);
            crate::simplify::powsimp::powsimp(arena, g)
        }
        FourierConvention::Ordinary => {
            let two_pi = two_pi(arena);
            let two_pi_nu = arena.mul(&[two_pi, omega]);
            let g = crate::transforms::subs::subs(arena, f, omega, two_pi_nu);
            finish(arena, g, omega)
        }
    };
    validate(arena, f, t, "fourier_transform")
}

/// Inverse Fourier transform of `expr` (a function of `omega`) as a
/// function of `t`, in the non-unitary angular convention.
pub(crate) fn inverse_fourier_transform(
    arena: &mut Arena,
    expr: ExprId,
    omega: ExprId,
    t: ExprId,
) -> Result<ExprId, SymplexError> {
    inverse_fourier_transform_with(arena, expr, omega, t, FourierConvention::NonUnitaryAngular)
}

/// Inverse Fourier transform of `expr` in the given convention.
pub(crate) fn inverse_fourier_transform_with(
    arena: &mut Arena,
    expr: ExprId,
    omega: ExprId,
    t: ExprId,
    convention: FourierConvention,
) -> Result<ExprId, SymplexError> {
    require_symbol(
        arena,
        omega,
        "inverse_fourier_transform",
        "the frequency variable",
    )?;
    require_symbol(arena, t, "inverse_fourier_transform", "the time variable")?;
    if t == omega {
        return Err(SymplexError::InvalidArgument {
            operation: "inverse_fourier_transform",
            reason: "time and frequency variables must be distinct".into(),
        });
    }
    let g = match convention {
        FourierConvention::NonUnitaryAngular => expr,
        FourierConvention::UnitaryAngular => {
            let root = sqrt_two_pi(arena);
            let g = arena.mul(&[root, expr]);
            crate::simplify::powsimp::powsimp(arena, g)
        }
        FourierConvention::Ordinary => {
            let two_pi = two_pi(arena);
            let nu_over = arena.div(omega, two_pi);
            let g = crate::transforms::subs::subs(arena, expr, omega, nu_over);
            normalize_deltas(arena, g, omega)
        }
    };
    let g = eliminate_piecewise(arena, g, omega, "inverse_fourier_transform")?;
    let g = crate::transforms::eval::eval(arena, g);
    let f = inverse(arena, g, omega, t, 0)?;
    let f = finish(arena, f, t);
    validate(arena, f, omega, "inverse_fourier_transform")
}

/// A transform result must be free of the source variable and of
/// unevaluated nodes (e.g. an unevaluated derivative of `δ`).
fn validate(
    arena: &Arena,
    f: ExprId,
    src: ExprId,
    op: &'static str,
) -> Result<ExprId, SymplexError> {
    if crate::base::walk::contains(arena, f, src) {
        return Err(fail(op, "result still depends on the source variable"));
    }
    if crate::base::walk::has_unevaluated(arena, f) {
        return Err(fail(
            op,
            "result would require a distribution that cannot be represented (e.g. δ′)",
        ));
    }
    Ok(f)
}

/// Final clean-up shared by both directions: sift deltas, rewrite
/// `e^{±iθ}` pairs into trigonometric form, normalise `δ(a·v + b)`.
fn finish(arena: &mut Arena, f: ExprId, var: ExprId) -> ExprId {
    let f = crate::transforms::eval::eval(arena, f);
    let f = crate::transforms::expand::expand(arena, f);
    let f = normalize_deltas(arena, f, var);
    let f = sift_deltas(arena, f, var);
    let f = euler_pairs_to_trig(arena, f);
    let f = crate::transforms::eval::eval(arena, f);
    let f = crate::transforms::expand::expand(arena, f);
    let f = crate::simplify::powsimp::powsimp(arena, f);
    crate::transforms::eval::eval(arena, f)
}

/// `√(a²) → a` for a provably positive `a`, otherwise `√(e)`.
fn positive_sqrt(arena: &mut Arena, e: ExprId) -> ExprId {
    if let ExprNode::Pow(base, exp) = arena.node(e).clone()
        && exp == arena.int(2)
        && param_sign(arena, base) == Some(1)
    {
        return base;
    }
    let r = arena.sqrt(e);
    crate::transforms::eval::eval(arena, r)
}

/// `cos θ + i sin θ` with `θ` sign-normalised so that `cos(−2t)` becomes
/// `cos(2t)` and `sin(−2t)` becomes `−sin(2t)`.
fn euler(arena: &mut Arena, theta: ExprId) -> ExprId {
    let i = arena.i_unit();
    let negative = match arena.node(theta).clone() {
        ExprNode::Neg(_) => true,
        ExprNode::Mul(ch) => ch
            .iter()
            .any(|&c| arena.as_num(c).is_some_and(|r| r.is_negative())),
        _ => arena.as_num(theta).is_some_and(|r| r.is_negative()),
    };
    let (th, sign) = if negative {
        let n = arena.neg(theta);
        (crate::transforms::eval::eval(arena, n), arena.neg_one())
    } else {
        (theta, arena.one())
    };
    let c = arena.cos(th);
    let s = arena.sin(th);
    let is = arena.mul(&[sign, i, s]);
    arena.add(&[c, is])
}

fn two_pi(arena: &mut Arena) -> ExprId {
    let two = arena.int(2);
    let pi = arena.pi();
    arena.mul(&[two, pi])
}

/// `√(2π)` as `√2·√π` (so that it cancels against [`inv_sqrt_two_pi`]).
fn sqrt_two_pi(arena: &mut Arena) -> ExprId {
    let two = arena.int(2);
    let pi = arena.pi();
    let r2 = arena.sqrt(two);
    let rpi = arena.sqrt(pi);
    arena.mul(&[r2, rpi])
}

/// `1/√(2π)` as `2^{-1/2}·π^{-1/2}`.
fn inv_sqrt_two_pi(arena: &mut Arena) -> ExprId {
    let two = arena.int(2);
    let pi = arena.pi();
    let neg_half = arena.rational(-1, 2);
    let r2 = arena.pow(two, neg_half);
    let rpi = arena.pow(pi, neg_half);
    arena.mul(&[r2, rpi])
}

fn i_times(arena: &mut Arena, e: ExprId) -> ExprId {
    let i = arena.i_unit();
    arena.mul(&[i, e])
}

// ═══════════════════════════════════════════════════════════════════════════
// Linear-argument analysis
// ═══════════════════════════════════════════════════════════════════════════

/// Decompose `expr = a·v + b` with `a`, `b` free of `v`. Returns `None` if
/// `expr` is not linear in `v`.
fn linear_in(arena: &mut Arena, expr: ExprId, v: ExprId) -> Option<(ExprId, ExprId)> {
    if expr == v {
        return Some((arena.one(), arena.zero()));
    }
    if !crate::base::walk::contains(arena, expr, v) {
        return Some((arena.zero(), expr));
    }
    match arena.node(expr).clone() {
        ExprNode::Neg(inner) => {
            let (a, b) = linear_in(arena, inner, v)?;
            Some((arena.neg(a), arena.neg(b)))
        }
        ExprNode::Mul(children) => {
            let mut coeff: SmallVec<[ExprId; 4]> = SmallVec::new();
            let mut seen_v = false;
            for &c in &children {
                if c == v {
                    if seen_v {
                        return None;
                    }
                    seen_v = true;
                } else if crate::base::walk::contains(arena, c, v) {
                    return None;
                } else {
                    coeff.push(c);
                }
            }
            if !seen_v {
                return None;
            }
            let a = if coeff.is_empty() {
                arena.one()
            } else {
                arena.mul(&coeff)
            };
            Some((a, arena.zero()))
        }
        ExprNode::Add(children) => {
            let mut a_terms: SmallVec<[ExprId; 4]> = SmallVec::new();
            let mut b_terms: SmallVec<[ExprId; 4]> = SmallVec::new();
            for &c in &children {
                let (a, b) = linear_in(arena, c, v)?;
                if !arena.is_zero_structural(a) {
                    a_terms.push(a);
                }
                if !arena.is_zero_structural(b) {
                    b_terms.push(b);
                }
            }
            let a = arena.add(&a_terms);
            let b = arena.add(&b_terms);
            Some((a, b))
        }
        _ => None,
    }
}

/// Split a constant `c` into `(real part, imaginary part)` under the
/// convention that all symbols are real: `c = re + i·im` structurally.
fn split_real_imag(arena: &mut Arena, c: ExprId) -> (ExprId, ExprId) {
    let i = arena.i_unit();
    if c == i {
        return (arena.zero(), arena.one());
    }
    match arena.node(c).clone() {
        ExprNode::Mul(children) if children.contains(&i) => {
            let rest: SmallVec<[ExprId; 4]> =
                children.iter().copied().filter(|&x| x != i).collect();
            if rest
                .iter()
                .any(|&r| crate::base::walk::contains(arena, r, i))
            {
                // i·(something containing i): fall through to the generic split.
            } else {
                let im = arena.mul(&rest);
                return (arena.zero(), im);
            }
        }
        ExprNode::Neg(inner) => {
            let (re, im) = split_real_imag(arena, inner);
            return (arena.neg(re), arena.neg(im));
        }
        ExprNode::Add(children) => {
            let mut re: SmallVec<[ExprId; 4]> = SmallVec::new();
            let mut im: SmallVec<[ExprId; 4]> = SmallVec::new();
            for &ch in &children {
                let (r, m) = split_real_imag(arena, ch);
                re.push(r);
                im.push(m);
            }
            let re = arena.add(&re);
            let im = arena.add(&im);
            return (re, im);
        }
        _ => {}
    }
    if crate::base::walk::contains(arena, c, i) {
        let (re, im) = crate::base::complex::as_real_imag(arena, c);
        (re, im)
    } else {
        (c, arena.zero())
    }
}

/// Sign of a real parameter via numbers and assumptions.
fn param_sign(arena: &mut Arena, e: ExprId) -> Option<i32> {
    crate::calculus::limit::const_sign(arena, e)
}

/// Error for a parameter whose sign is needed but unknown.
fn need_sign(op: &'static str, arena: &Arena, e: ExprId, cond: &str) -> SymplexError {
    fail(
        op,
        format!(
            "requires {cond} (declare the sign of {} with Assumption::Positive / Assumption::Negative)",
            arena.display(e)
        ),
    )
}

/// `n!` as an expression.
fn factorial_expr(arena: &mut Arena, n: u64) -> ExprId {
    let mut result = BigInt::from(1u64);
    for i in 2..=n {
        result *= BigInt::from(i);
    }
    let nid = arena.intern_num(Ratio::from_integer(result));
    arena.intern(ExprNode::Num(nid))
}

/// Non-negative integer power of `v` in a factor, if it is one.
fn power_of(arena: &Arena, factor: ExprId, v: ExprId) -> Option<u64> {
    if factor == v {
        return Some(1);
    }
    if let ExprNode::Pow(base, exp) = arena.node(factor)
        && *base == v
        && let Some(r) = arena.as_num(*exp)
        && r.is_integer()
        && r.is_positive()
    {
        return r.to_integer().to_u64();
    }
    None
}

/// Split a `Mul` (or single factor) into `(constant factors, v-dependent factors)`.
fn split_factors(arena: &Arena, expr: ExprId, v: ExprId) -> (Vec<ExprId>, Vec<ExprId>) {
    let children: Vec<ExprId> = match arena.node(expr) {
        ExprNode::Mul(ch) => ch.iter().copied().collect(),
        _ => vec![expr],
    };
    let mut consts = Vec::new();
    let mut deps = Vec::new();
    for c in children {
        if crate::base::walk::contains(arena, c, v) {
            deps.push(c);
        } else {
            consts.push(c);
        }
    }
    (consts, deps)
}

// ═══════════════════════════════════════════════════════════════════════════
// Result clean-up: δ normalisation, sifting, Euler pairs
// ═══════════════════════════════════════════════════════════════════════════

/// `δ(a·v + b) → δ(v + b/a)/|a|` for every delta with a linear argument in `v`.
fn normalize_deltas(arena: &mut Arena, f: ExprId, v: ExprId) -> ExprId {
    let post = crate::base::walk::post_order_ids(arena, f);
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();
    for &id in &post {
        if !crate::base::walk::contains(arena, id, v) {
            cache.insert(id, id);
            continue;
        }
        let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
        let new = match arena.node(rebuilt).clone() {
            ExprNode::DiracDelta(arg) => match linear_in(arena, arg, v) {
                Some((a, b)) if a != arena.one() && !arena.is_zero_structural(a) => {
                    let shift = arena.div(b, a);
                    let inner = arena.add(&[v, shift]);
                    let delta = arena.dirac_delta(inner);
                    let abs_a = match param_sign(arena, a) {
                        Some(s) if s > 0 => a,
                        Some(s) if s < 0 => arena.neg(a),
                        _ => arena.abs(a),
                    };
                    arena.div(delta, abs_a)
                }
                _ => rebuilt,
            },
            _ => rebuilt,
        };
        cache.insert(id, new);
    }
    cache.get(&f).copied().unwrap_or(f)
}

/// In every product containing `δ(v − c)`, evaluate the other factors at
/// `v = c` (the sifting property). Works on the top-level sum after
/// expansion.
fn sift_deltas(arena: &mut Arena, f: ExprId, v: ExprId) -> ExprId {
    let terms: Vec<ExprId> = match arena.node(f) {
        ExprNode::Add(ch) => ch.iter().copied().collect(),
        _ => vec![f],
    };
    let mut out = Vec::with_capacity(terms.len());
    for term in terms {
        out.push(sift_term(arena, term, v));
    }
    if out.len() == 1 {
        out[0]
    } else {
        arena.add(&out)
    }
}

fn sift_term(arena: &mut Arena, term: ExprId, v: ExprId) -> ExprId {
    let (neg, body) = match arena.node(term) {
        ExprNode::Neg(inner) => (true, *inner),
        _ => (false, term),
    };
    let children: Vec<ExprId> = match arena.node(body) {
        ExprNode::Mul(ch) => ch.iter().copied().collect(),
        _ => return term,
    };
    // Find a δ(v − c).
    let mut point = None;
    let mut delta_idx = None;
    for (idx, &c) in children.iter().enumerate() {
        if let ExprNode::DiracDelta(arg) = arena.node(c).clone()
            && let Some((a, b)) = linear_in(arena, arg, v)
            && a == arena.one()
        {
            point = Some(arena.neg(b));
            delta_idx = Some(idx);
            break;
        }
    }
    let (Some(point), Some(delta_idx)) = (point, delta_idx) else {
        return term;
    };
    let mut new_children = Vec::with_capacity(children.len());
    for (idx, &c) in children.iter().enumerate() {
        if idx == delta_idx || !crate::base::walk::contains(arena, c, v) {
            new_children.push(c);
        } else {
            let at = crate::transforms::subs::subs(arena, c, v, point);
            let at = crate::transforms::eval::eval(arena, at);
            new_children.push(at);
        }
    }
    let rebuilt = arena.mul(&new_children);
    if neg { arena.neg(rebuilt) } else { rebuilt }
}

/// When both `e^{iθ}` and `e^{−iθ}` occur, rewrite every `e^{iθ}` as
/// `cos θ + i sin θ` so that the pair combines into `2cos θ` / `2i sin θ`.
fn euler_pairs_to_trig(arena: &mut Arena, f: ExprId) -> ExprId {
    let post = crate::base::walk::post_order_ids(arena, f);
    // Collect imaginary exponents θ (from exp(i·θ)).
    let mut thetas: Vec<ExprId> = Vec::new();
    for &id in &post {
        if let ExprNode::Exp(arg) = arena.node(id).clone() {
            let (re, im) = split_real_imag(arena, arg);
            if arena.is_zero_structural(re) && !arena.is_zero_structural(im) {
                thetas.push(crate::transforms::eval::eval(arena, im));
            }
        }
    }
    let has_pair = thetas.iter().any(|&th| {
        let neg = arena.neg(th);
        let neg = crate::transforms::eval::eval(arena, neg);
        thetas.contains(&neg)
    });
    if !has_pair {
        return f;
    }
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();
    for &id in &post {
        let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
        let new = if let ExprNode::Exp(arg) = arena.node(rebuilt).clone() {
            let (re, im) = split_real_imag(arena, arg);
            if arena.is_zero_structural(re) && !arena.is_zero_structural(im) {
                euler(arena, im)
            } else {
                rebuilt
            }
        } else {
            rebuilt
        };
        cache.insert(id, new);
    }
    cache.get(&f).copied().unwrap_or(f)
}

// ═══════════════════════════════════════════════════════════════════════════
// Piecewise → Heaviside
// ═══════════════════════════════════════════════════════════════════════════

/// Indicator of a boolean condition in `v` as a product of Heaviside steps,
/// for conditions of the form `v ⋚ c`, `|v| < c`, `|v| > c`, `And(...)`,
/// `Or(...)` (disjoint assumed), `Not(...)`, `true`.
fn condition_indicator(arena: &mut Arena, cond: ExprId, v: ExprId) -> Option<ExprId> {
    match arena.node(cond).clone() {
        ExprNode::BoolTrue => Some(arena.one()),
        ExprNode::BoolFalse => Some(arena.zero()),
        // `a < b` is stored as `Gt(b, a)`, so these two arms cover all
        // strict and non-strict inequalities.
        ExprNode::Gt(a, b) | ExprNode::Ge(a, b) => {
            let diff = arena.sub(a, b);
            positive_indicator(arena, diff, v)
        }
        ExprNode::And(children) => {
            let mut factors = Vec::with_capacity(children.len());
            for c in children {
                factors.push(condition_indicator(arena, c, v)?);
            }
            Some(arena.mul(&factors))
        }
        ExprNode::Or(children) => {
            // 1 − Π(1 − I_k)
            let one = arena.one();
            let mut factors = Vec::with_capacity(children.len());
            for c in children {
                let ind = condition_indicator(arena, c, v)?;
                factors.push(arena.sub(one, ind));
            }
            let prod = arena.mul(&factors);
            Some(arena.sub(one, prod))
        }
        ExprNode::Not(inner) => {
            let ind = condition_indicator(arena, inner, v)?;
            let one = arena.one();
            Some(arena.sub(one, ind))
        }
        _ => None,
    }
}

/// Indicator of `expr > 0` where `expr` is `a·v + b` or `c − |v|` / `|v| − c`.
fn positive_indicator(arena: &mut Arena, expr: ExprId, v: ExprId) -> Option<ExprId> {
    if let Some((a, b)) = linear_in(arena, expr, v) {
        if arena.is_zero_structural(a) {
            // Constant condition (e.g. the trailing `true` written as `1 > 0`).
            return match param_sign(arena, b) {
                Some(s) if s > 0 => Some(arena.one()),
                Some(_) => Some(arena.zero()),
                None => None,
            };
        }
        let lin = expr_or_lin(arena, a, b, v);
        return Some(arena.heaviside(lin));
    }
    // c − |v| > 0  ⇔  −c < v < c ;  |v| − c > 0 ⇔ v > c or v < −c
    let (consts, deps) = match arena.node(expr).clone() {
        ExprNode::Add(ch) => {
            let mut consts = Vec::new();
            let mut deps = Vec::new();
            for c in ch {
                if crate::base::walk::contains(arena, c, v) {
                    deps.push(c);
                } else {
                    consts.push(c);
                }
            }
            (consts, deps)
        }
        _ => return None,
    };
    if deps.len() != 1 {
        return None;
    }
    let c = arena.add(&consts);
    // dep = k·|v| with a constant k (−|v| is canonicalised as Mul(-1, |v|)).
    let (k, inner) = match arena.node(deps[0]).clone() {
        ExprNode::Abs(inner) => (arena.one(), inner),
        ExprNode::Neg(n) => match arena.node(n).clone() {
            ExprNode::Abs(inner) => (arena.neg_one(), inner),
            _ => return None,
        },
        ExprNode::Mul(ch) => {
            let mut abs_inner = None;
            let mut rest: SmallVec<[ExprId; 4]> = SmallVec::new();
            for &f in &ch {
                if let ExprNode::Abs(inner) = arena.node(f).clone()
                    && abs_inner.is_none()
                {
                    abs_inner = Some(inner);
                } else if crate::base::walk::contains(arena, f, v) {
                    return None;
                } else {
                    rest.push(f);
                }
            }
            (arena.mul(&rest), abs_inner?)
        }
        _ => return None,
    };
    if inner != v {
        return None;
    }
    let ks = param_sign(arena, k)?;
    if ks == 0 {
        return None;
    }
    // k|v| + c > 0  ⇔  |v| < −c/k (k < 0)   or   |v| > −c/k (k > 0)
    let thr = arena.div(c, k);
    let thr = arena.neg(thr);
    let thr = crate::transforms::eval::eval(arena, thr);
    let vpc = arena.add(&[v, thr]);
    let cmv = arena.sub(thr, v);
    let h1 = arena.heaviside(vpc);
    let h2 = arena.heaviside(cmv);
    let inside = arena.mul(&[h1, h2]);
    if ks < 0 {
        Some(inside)
    } else {
        let one = arena.one();
        Some(arena.sub(one, inside))
    }
}

fn expr_or_lin(arena: &mut Arena, a: ExprId, b: ExprId, v: ExprId) -> ExprId {
    let av = arena.mul(&[a, v]);
    arena.add(&[av, b])
}

/// Replace every `Piecewise` node whose conditions involve `v` by its
/// Heaviside form (see [`piecewise_to_heaviside`]).
fn eliminate_piecewise(
    arena: &mut Arena,
    expr: ExprId,
    v: ExprId,
    op: &'static str,
) -> Result<ExprId, SymplexError> {
    let post = crate::base::walk::post_order_ids(arena, expr);
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();
    for &id in &post {
        let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
        let new = match arena.node(rebuilt).clone() {
            ExprNode::Piecewise(pairs) if crate::base::walk::contains(arena, rebuilt, v) => {
                piecewise_to_heaviside(arena, &pairs, v).ok_or_else(|| {
                    fail(
                        op,
                        format!(
                            "piecewise conditions must be linear inequalities in {}",
                            arena.display(v)
                        ),
                    )
                })?
            }
            _ => rebuilt,
        };
        cache.insert(id, new);
    }
    Ok(cache.get(&expr).copied().unwrap_or(expr))
}

/// Convert `Piecewise` (sequential semantics) into a sum of
/// `value·indicator` terms using Heaviside steps.
fn piecewise_to_heaviside(
    arena: &mut Arena,
    pairs: &[(ExprId, ExprId)],
    v: ExprId,
) -> Option<ExprId> {
    let one = arena.one();
    let mut remaining = one;
    let mut terms = Vec::with_capacity(pairs.len());
    for &(val, cond) in pairs {
        let ind = condition_indicator(arena, cond, v)?;
        let eff = arena.mul(&[remaining, ind]);
        terms.push(arena.mul(&[val, eff]));
        // Complement: `1 − H(u)` is written `H(−u)` so that the following
        // branches stay single-step products (`e^{−t}·H(t)` rather than
        // `e^{−t} − e^{−t}H(−t)`, which is not transformable term by term).
        let not_ind = match arena.node(ind).clone() {
            ExprNode::Heaviside(arg) => {
                let neg = arena.neg(arg);
                arena.heaviside(neg)
            }
            _ => arena.sub(one, ind),
        };
        remaining = arena.mul(&[remaining, not_ind]);
    }
    let sum = arena.add(&terms);
    let sum = crate::transforms::expand::expand(arena, sum);
    Some(simplify_heaviside_products(arena, sum, v))
}

/// `H(v − a)·H(v − b) → H(v − max(a,b))`, `H(v − a)·H(b − v) → 0` when
/// `a ≥ b`, and `H(v − a)·H(b − v) → H(v − a) − H(v − b)` when `a < b`,
/// for numeric `a`, `b`.
fn simplify_heaviside_products(arena: &mut Arena, f: ExprId, v: ExprId) -> ExprId {
    let terms: Vec<ExprId> = match arena.node(f) {
        ExprNode::Add(ch) => ch.iter().copied().collect(),
        _ => vec![f],
    };
    let mut out = Vec::with_capacity(terms.len());
    for term in terms {
        out.push(simplify_heaviside_term(arena, term, v));
    }
    let r = arena.add(&out);
    crate::transforms::eval::eval(arena, r)
}

/// Classify `H(arg)` with `arg = a·v + b`, numeric `a ≠ 0`: returns
/// `(rising, threshold)` meaning `H(v − θ)` (rising) or `H(θ − v)`.
fn heaviside_kind(arena: &mut Arena, arg: ExprId, v: ExprId) -> Option<(bool, Q)> {
    let (a, b) = linear_in(arena, arg, v)?;
    let a = arena.as_num(a).cloned()?;
    let b = arena.as_num(b).cloned()?;
    if a.is_zero() {
        return None;
    }
    // a·v + b > 0 ⇔ v > −b/a (a > 0) or v < −b/a (a < 0)
    let theta = -b / a.clone();
    Some((a.is_positive(), theta))
}

fn simplify_heaviside_term(arena: &mut Arena, term: ExprId, v: ExprId) -> ExprId {
    let children: Vec<ExprId> = match arena.node(term) {
        ExprNode::Mul(ch) => ch.iter().copied().collect(),
        _ => return term,
    };
    let mut lower: Option<Q> = None; // v > lower
    let mut upper: Option<Q> = None; // v < upper
    let mut others = Vec::new();
    let mut n_heaviside = 0;
    for &c in &children {
        // `H(u)^k` (k a positive integer) is `H(u)`.
        let c = match arena.node(c).clone() {
            ExprNode::Pow(base, e)
                if matches!(arena.node(base), ExprNode::Heaviside(_))
                    && arena
                        .as_num(e)
                        .is_some_and(|r| r.is_integer() && r.is_positive()) =>
            {
                n_heaviside += 1; // counts as a simplifiable product
                base
            }
            _ => c,
        };
        if let ExprNode::Heaviside(arg) = arena.node(c).clone()
            && let Some((rising, theta)) = heaviside_kind(arena, arg, v)
        {
            n_heaviside += 1;
            if rising {
                lower = Some(match lower {
                    Some(l) if l > theta => l,
                    _ => theta,
                });
            } else {
                upper = Some(match upper {
                    Some(u) if u < theta => u,
                    _ => theta,
                });
            }
        } else {
            others.push(c);
        }
    }
    if n_heaviside < 2 {
        return term;
    }
    let num = |arena: &mut Arena, r: Q| {
        let nid = arena.intern_num(r);
        arena.intern(ExprNode::Num(nid))
    };
    let window = match (lower, upper) {
        (Some(l), Some(u)) => {
            if l >= u {
                return arena.zero();
            }
            let l_id = num(arena, l);
            let u_id = num(arena, u);
            let vl = arena.sub(v, l_id);
            let vu = arena.sub(v, u_id);
            let hl = arena.heaviside(vl);
            let hu = arena.heaviside(vu);
            arena.sub(hl, hu)
        }
        (Some(l), None) => {
            let l_id = num(arena, l);
            let vl = arena.sub(v, l_id);
            arena.heaviside(vl)
        }
        (None, Some(u)) => {
            let u_id = num(arena, u);
            let uv = arena.sub(u_id, v);
            arena.heaviside(uv)
        }
        (None, None) => return term,
    };
    others.push(window);
    let r = arena.mul(&others);
    crate::transforms::expand::expand(arena, r)
}

// ═══════════════════════════════════════════════════════════════════════════
// Forward transform
// ═══════════════════════════════════════════════════════════════════════════

fn forward(
    arena: &mut Arena,
    expr: ExprId,
    t: ExprId,
    w: ExprId,
    depth: usize,
) -> Result<ExprId, SymplexError> {
    const OP: &str = "fourier_transform";
    if depth > MAX_RULE_DEPTH {
        return Err(fail(OP, "rule nesting too deep"));
    }
    let expr = crate::transforms::eval::eval(arena, expr);
    let node = arena.node(expr).clone();

    // ── Linearity ──
    if let ExprNode::Add(children) = &node {
        let kids = children.clone();
        let mut terms = Vec::with_capacity(kids.len());
        for child in kids {
            terms.push(forward(arena, child, t, w, depth)?);
        }
        return Ok(arena.add(&terms));
    }
    if let ExprNode::Neg(inner) = node {
        let f = forward(arena, inner, t, w, depth)?;
        return Ok(arena.neg(f));
    }

    // ── Constant: c → 2π c δ(ω) ──
    if !crate::base::walk::contains(arena, expr, t) {
        let tp = two_pi(arena);
        let d = arena.dirac_delta(w);
        return Ok(arena.mul(&[expr, tp, d]));
    }

    // ── Constant factors ──
    let (consts, deps) = split_factors(arena, expr, t);
    if !consts.is_empty() {
        let body = arena.mul(&deps);
        let f = forward(arena, body, t, w, depth)?;
        let mut all = consts;
        all.push(f);
        return Ok(arena.mul(&all));
    }

    // ── Unary table entries ──
    if let Some(r) = forward_unary(arena, expr, &node, t, w)? {
        return Ok(r);
    }

    // ── Products ──
    if let ExprNode::Mul(children) = &node {
        let kids: Vec<ExprId> = children.iter().copied().collect();
        if let Some(r) = forward_product(arena, &kids, t, w, depth)? {
            return Ok(r);
        }
    }

    // ── Derivative rule: f'(t) → iω F(ω) ──
    if let ExprNode::Derivative(body, var) = node
        && var == t
    {
        let f = forward(arena, body, t, w, depth + 1)?;
        let iw = i_times(arena, w);
        return Ok(arena.mul(&[iw, f]));
    }

    // ── Shift / scaling rules ──
    if let Some(r) = try_shift_scale(arena, expr, t, w, depth, Dir::Forward)? {
        return Ok(r);
    }

    Err(fail(
        OP,
        format!("no transform rule applies to {}", arena.display(expr)),
    ))
}

/// Table entries for a single function node.
fn forward_unary(
    arena: &mut Arena,
    expr: ExprId,
    node: &ExprNode,
    t: ExprId,
    w: ExprId,
) -> Result<Option<ExprId>, SymplexError> {
    const OP: &str = "fourier_transform";
    let one = arena.one();
    let zero = arena.zero();
    let pi = arena.pi();
    let i = arena.i_unit();
    match node.clone() {
        // δ(a t + b) → e^{iωb/a}/|a|
        ExprNode::DiracDelta(arg) => {
            let Some((a, b)) = linear_in(arena, arg, t) else {
                return Ok(None);
            };
            if arena.is_zero_structural(a) {
                return Ok(None);
            }
            let abs_a = match param_sign(arena, a) {
                Some(s) if s > 0 => a,
                Some(s) if s < 0 => arena.neg(a),
                Some(_) => return Ok(None),
                None => arena.abs(a),
            };
            let shift = arena.div(b, a);
            let phase = arena.mul(&[i, w, shift]);
            let e = arena.exp(phase);
            Ok(Some(arena.div(e, abs_a)))
        }
        // H(t) → πδ(ω) + 1/(iω);  H(−t) → πδ(ω) − 1/(iω)
        ExprNode::Heaviside(arg) if arg == t => {
            let d = arena.dirac_delta(w);
            let pd = arena.mul(&[pi, d]);
            let iw = i_times(arena, w);
            let r = arena.div(one, iw);
            Ok(Some(arena.add(&[pd, r])))
        }
        ExprNode::Heaviside(arg) => {
            let Some((a, b)) = linear_in(arena, arg, t) else {
                return Ok(None);
            };
            if a == arena.neg_one() && arena.is_zero_structural(b) {
                let d = arena.dirac_delta(w);
                let pd = arena.mul(&[pi, d]);
                let iw = i_times(arena, w);
                let r = arena.div(one, iw);
                return Ok(Some(arena.sub(pd, r)));
            }
            Ok(None) // shift / scale rules
        }
        // sign(t) → 2/(iω)
        ExprNode::Sign(arg) if arg == t => {
            let two = arena.int(2);
            let iw = i_times(arena, w);
            Ok(Some(arena.div(two, iw)))
        }
        // |t| → −2/ω²
        ExprNode::Abs(arg) if arg == t => {
            let two = arena.int(2);
            let w2 = arena.pow(w, two);
            let r = arena.div(two, w2);
            Ok(Some(arena.neg(r)))
        }
        // 1/t → −iπ sign(ω)
        ExprNode::Pow(base, e) if base == t && e == arena.neg_one() => {
            let s = arena.sign(w);
            let ips = arena.mul(&[i, pi, s]);
            Ok(Some(arena.neg(ips)))
        }
        // sin(ω₀ t) → iπ[δ(ω+ω₀) − δ(ω−ω₀)],  cos(ω₀ t) → π[δ(ω−ω₀) + δ(ω+ω₀)]
        ExprNode::Sin(arg) | ExprNode::Cos(arg) => {
            let Some((w0, b)) = linear_in(arena, arg, t) else {
                return Ok(None);
            };
            if !arena.is_zero_structural(b) {
                return Ok(None); // handled by the shift rule
            }
            let wp = arena.add(&[w, w0]);
            let wm = arena.sub(w, w0);
            let dp = arena.dirac_delta(wp);
            let dm = arena.dirac_delta(wm);
            if matches!(node, ExprNode::Sin(_)) {
                let diff = arena.sub(dp, dm);
                Ok(Some(arena.mul(&[i, pi, diff])))
            } else {
                let sum = arena.add(&[dm, dp]);
                Ok(Some(arena.mul(&[pi, sum])))
            }
        }
        ExprNode::Exp(arg) => {
            // e^{−a|t|} → 2a/(a² + ω²)
            if let Some((a, inner)) = neg_coeff_times_abs(arena, arg)
                && inner == t
            {
                return match param_sign(arena, a) {
                    Some(s) if s > 0 => {
                        let two = arena.int(2);
                        let num = arena.mul(&[two, a]);
                        let a2 = arena.pow(a, two);
                        let w2 = arena.pow(w, two);
                        let den = arena.add(&[a2, w2]);
                        Ok(Some(arena.div(num, den)))
                    }
                    Some(_) => Err(fail(
                        OP,
                        "exp(a|t|) with a ≥ 0 is not Fourier transformable",
                    )),
                    None => Err(need_sign(OP, arena, a, "a > 0 in exp(-a|t|)")),
                };
            }
            // Gaussian e^{−a t²} → √(π/a) e^{−ω²/(4a)}
            if let Some(a) = neg_coeff_times_square(arena, arg, t) {
                return match param_sign(arena, a) {
                    Some(s) if s > 0 => {
                        let ratio = arena.div(pi, a);
                        let root = arena.sqrt(ratio);
                        let two = arena.int(2);
                        let four = arena.int(4);
                        let w2 = arena.pow(w, two);
                        let four_a = arena.mul(&[four, a]);
                        let q = arena.div(w2, four_a);
                        let nq = arena.neg(q);
                        let e = arena.exp(nq);
                        Ok(Some(arena.mul(&[root, e])))
                    }
                    Some(_) => Err(fail(
                        OP,
                        "exp(a t²) with a ≥ 0 is not Fourier transformable",
                    )),
                    None => Err(need_sign(OP, arena, a, "a > 0 in exp(-a t²)")),
                };
            }
            // e^{(iω₀) t + b} → e^b 2π δ(ω − ω₀);  real exponentials need a step.
            if let Some((a, b)) = linear_in(arena, arg, t) {
                let (re, im) = split_real_imag(arena, a);
                if arena.is_zero_structural(re) {
                    let tp = two_pi(arena);
                    let eb = arena.exp(b);
                    let wm = arena.sub(w, im);
                    let d = arena.dirac_delta(wm);
                    return Ok(Some(arena.mul(&[eb, tp, d])));
                }
                return Err(fail(
                    OP,
                    format!(
                        "{} grows without bound; multiply by H(t) or H(-t)",
                        arena.display(expr)
                    ),
                ));
            }
            Ok(None)
        }
        _ => {
            let _ = zero;
            Ok(None)
        }
    }
}

/// `arg = −a·|inner|` → `Some((a, inner))`.
fn neg_coeff_times_abs(arena: &mut Arena, arg: ExprId) -> Option<(ExprId, ExprId)> {
    let (coeff, inner) = match arena.node(arg).clone() {
        ExprNode::Neg(x) => match arena.node(x).clone() {
            ExprNode::Abs(inner) => (arena.one(), inner),
            ExprNode::Mul(ch) => {
                let mut abs_inner = None;
                let mut rest: SmallVec<[ExprId; 4]> = SmallVec::new();
                for &c in &ch {
                    if let ExprNode::Abs(inner) = arena.node(c).clone()
                        && abs_inner.is_none()
                    {
                        abs_inner = Some(inner);
                    } else {
                        rest.push(c);
                    }
                }
                (arena.mul(&rest), abs_inner?)
            }
            _ => return None,
        },
        ExprNode::Mul(ch) => {
            let mut abs_inner = None;
            let mut rest: SmallVec<[ExprId; 4]> = SmallVec::new();
            for &c in &ch {
                if let ExprNode::Abs(inner) = arena.node(c).clone()
                    && abs_inner.is_none()
                {
                    abs_inner = Some(inner);
                } else {
                    rest.push(c);
                }
            }
            let coeff = arena.mul(&rest);
            (arena.neg(coeff), abs_inner?)
        }
        _ => return None,
    };
    let coeff = crate::transforms::eval::eval(arena, coeff);
    Some((coeff, inner))
}

/// `arg = −a·t²` → `Some(a)`.
fn neg_coeff_times_square(arena: &mut Arena, arg: ExprId, t: ExprId) -> Option<ExprId> {
    let two = arena.int(2);
    let t2 = arena.pow(t, two);
    let (coeff, sq) = match arena.node(arg).clone() {
        ExprNode::Neg(x) => {
            if x == t2 {
                (arena.one(), t2)
            } else if let ExprNode::Mul(ch) = arena.node(x).clone() {
                let rest: SmallVec<[ExprId; 4]> = ch.iter().copied().filter(|&c| c != t2).collect();
                if rest.len() + 1 != ch.len() {
                    return None;
                }
                (arena.mul(&rest), t2)
            } else {
                return None;
            }
        }
        ExprNode::Mul(ch) => {
            let rest: SmallVec<[ExprId; 4]> = ch.iter().copied().filter(|&c| c != t2).collect();
            if rest.len() + 1 != ch.len() {
                return None;
            }
            let c = arena.mul(&rest);
            (arena.neg(c), t2)
        }
        _ => return None,
    };
    let _ = sq;
    if crate::base::walk::contains(arena, coeff, t) {
        return None;
    }
    Some(crate::transforms::eval::eval(arena, coeff))
}

/// Products: `tⁿ e^{at} H(±t)`, `sin(at)/t`, modulation `e^{iω₀t} g(t)`,
/// step windows `H(·)H(·)`, and `tⁿ g(t) → iⁿ dⁿG/dωⁿ`.
fn forward_product(
    arena: &mut Arena,
    kids: &[ExprId],
    t: ExprId,
    w: ExprId,
    depth: usize,
) -> Result<Option<ExprId>, SymplexError> {
    const OP: &str = "fourier_transform";
    let i = arena.i_unit();

    // Classify factors.
    let mut t_power: u64 = 0;
    let mut steps: Vec<(ExprId, ExprId)> = Vec::new(); // (a, b) of H(a t + b)
    let mut exp_lin: Vec<(ExprId, ExprId)> = Vec::new(); // (a, b) of exp(a t + b)
    let mut inv_t = false;
    let mut others: Vec<ExprId> = Vec::new();
    for &k in kids {
        if let Some(n) = power_of(arena, k, t) {
            t_power += n;
            continue;
        }
        match arena.node(k).clone() {
            ExprNode::Pow(base, e) if base == t && e == arena.neg_one() => inv_t = true,
            ExprNode::Heaviside(arg) => {
                if let Some(lin) = linear_in(arena, arg, t) {
                    steps.push(lin);
                } else {
                    others.push(k);
                }
            }
            ExprNode::Exp(arg) => {
                if let Some(lin) = linear_in(arena, arg, t) {
                    exp_lin.push(lin);
                } else {
                    others.push(k);
                }
            }
            _ => others.push(k),
        }
    }

    // ── Two steps forming a window: H(t − lo)·H(hi − t) → H(t−lo) − H(t−hi) ──
    if steps.len() >= 2 {
        let mut prod: Vec<ExprId> = Vec::new();
        for (a, b) in &steps {
            let arg = expr_or_lin(arena, *a, *b, t);
            prod.push(arena.heaviside(arg));
        }
        let hp = arena.mul(&prod);
        let simplified = simplify_heaviside_products(arena, hp, t);
        if simplified != hp {
            let mut rest = others.clone();
            for (a, b) in &exp_lin {
                let arg = expr_or_lin(arena, *a, *b, t);
                rest.push(arena.exp(arg));
            }
            if inv_t {
                rest.push(arena.pow(t, arena.neg_one()));
            }
            if t_power > 0 {
                let n = arena.int(t_power as i64);
                rest.push(arena.pow(t, n));
            }
            rest.push(simplified);
            let e = arena.mul(&rest);
            let e = crate::transforms::expand::expand(arena, e);
            return forward(arena, e, t, w, depth + 1).map(Some);
        }
    }

    // ── Modulation: pull out e^{iω₀ t} (imaginary part of every exponential rate) ──
    let mut real_exp: Option<(ExprId, ExprId)> = None;
    let mut modulation: Option<ExprId> = None;
    let mut const_factor: Vec<ExprId> = Vec::new();
    for (a, b) in &exp_lin {
        let (re, im) = split_real_imag(arena, *a);
        if !arena.is_zero_structural(im) {
            modulation = Some(match modulation {
                Some(m) => arena.add(&[m, im]),
                None => im,
            });
            const_factor.push(arena.exp(*b));
            if !arena.is_zero_structural(re) {
                if real_exp.is_some() {
                    return Err(fail(OP, "cannot combine several real exponentials"));
                }
                real_exp = Some((re, arena.zero()));
            }
        } else if real_exp.is_none() {
            real_exp = Some((*a, *b));
        } else {
            return Err(fail(OP, "cannot combine several real exponentials"));
        }
    }
    if let Some(w0) = modulation {
        // Rebuild g(t) without the modulating factors.
        let mut g_factors = others.clone();
        if let Some((a, b)) = real_exp {
            let arg = expr_or_lin(arena, a, b, t);
            g_factors.push(arena.exp(arg));
        }
        for (a, b) in &steps {
            let arg = expr_or_lin(arena, *a, *b, t);
            g_factors.push(arena.heaviside(arg));
        }
        if inv_t {
            g_factors.push(arena.pow(t, arena.neg_one()));
        }
        if t_power > 0 {
            let n = arena.int(t_power as i64);
            g_factors.push(arena.pow(t, n));
        }
        let g = if g_factors.is_empty() {
            arena.one()
        } else {
            arena.mul(&g_factors)
        };
        let gf = forward(arena, g, t, w, depth + 1)?;
        let shifted_w = arena.sub(w, w0);
        let shifted = crate::transforms::subs::subs(arena, gf, w, shifted_w);
        const_factor.push(shifted);
        return Ok(Some(arena.mul(&const_factor)));
    }

    // ── cos(ω₀t)·g(t) → [G(ω−ω₀) + G(ω+ω₀)]/2,  sin(ω₀t)·g(t) → [G(ω−ω₀) − G(ω+ω₀)]/(2i) ──
    // (not for sinc, which has its own entry)
    if let Some((idx, is_sin, arg)) =
        others
            .iter()
            .enumerate()
            .find_map(|(k, &o)| match *arena.node(o) {
                ExprNode::Sin(a) => Some((k, true, a)),
                ExprNode::Cos(a) => Some((k, false, a)),
                _ => None,
            })
        && !(inv_t && others.len() == 1 && t_power == 0 && steps.is_empty() && exp_lin.is_empty())
        && (others.len() > 1 || !steps.is_empty() || !exp_lin.is_empty() || inv_t)
    {
        let trig = others[idx];
        if let Some((w0, b)) = linear_in(arena, arg, t)
            && arena.is_zero_structural(b)
            && !crate::base::walk::contains(arena, w0, i)
        {
            let mut g_factors: Vec<ExprId> =
                others.iter().copied().filter(|&o| o != trig).collect();
            for (a, b) in &exp_lin {
                let arg = expr_or_lin(arena, *a, *b, t);
                g_factors.push(arena.exp(arg));
            }
            for (a, b) in &steps {
                let arg = expr_or_lin(arena, *a, *b, t);
                g_factors.push(arena.heaviside(arg));
            }
            if inv_t {
                g_factors.push(arena.pow(t, arena.neg_one()));
            }
            if t_power > 0 {
                let n = arena.int(t_power as i64);
                g_factors.push(arena.pow(t, n));
            }
            let g = arena.mul(&g_factors);
            let gf = forward(arena, g, t, w, depth + 1)?;
            let wm = arena.sub(w, w0);
            let wp = arena.add(&[w, w0]);
            let g_minus = crate::transforms::subs::subs(arena, gf, w, wm);
            let g_plus = crate::transforms::subs::subs(arena, gf, w, wp);
            let two = arena.int(2);
            let r = if is_sin {
                let d = arena.sub(g_minus, g_plus);
                let two_i = arena.mul(&[two, i]);
                arena.div(d, two_i)
            } else {
                let s = arena.add(&[g_minus, g_plus]);
                arena.div(s, two)
            };
            return Ok(Some(r));
        }
    }

    // ── sinc: sin(a t)/t → π sign(a) [H(ω + |a|) − H(ω − |a|)] ──
    if inv_t && t_power == 0 && steps.is_empty() && exp_lin.is_empty() && others.len() == 1 {
        if let ExprNode::Sin(arg) = arena.node(others[0]).clone()
            && let Some((a, b)) = linear_in(arena, arg, t)
            && arena.is_zero_structural(b)
        {
            let pi = arena.pi();
            let (sgn, abs_a) = match param_sign(arena, a) {
                Some(s) if s > 0 => (arena.one(), a),
                Some(s) if s < 0 => (arena.neg_one(), arena.neg(a)),
                Some(_) => return Ok(Some(arena.zero())),
                None => return Err(need_sign(OP, arena, a, "the sign of a in sin(a t)/t")),
            };
            let wp = arena.add(&[w, abs_a]);
            let wm = arena.sub(w, abs_a);
            let hp = arena.heaviside(wp);
            let hm = arena.heaviside(wm);
            let rect = arena.sub(hp, hm);
            return Ok(Some(arena.mul(&[sgn, pi, rect])));
        }
        return Ok(None);
    }

    // ── tⁿ e^{at} H(±t) ──
    if others.is_empty() && !inv_t && steps.len() == 1 {
        let (sa, sb) = steps[0];
        if !arena.is_zero_structural(sb) {
            return Ok(None); // shifted step → shift rule
        }
        let step_sign = match param_sign(arena, sa) {
            Some(s) if s != 0 => s,
            _ => return Ok(None),
        };
        let a = match real_exp {
            Some((a, _)) => a,
            None => arena.zero(),
        };
        let eb = match real_exp {
            Some((_, b)) => arena.exp(b),
            None => arena.one(),
        };
        let a_sign = param_sign(arena, a);
        let iw = i_times(arena, w);
        let n = t_power;
        let n_fact = factorial_expr(arena, n);
        let np1 = arena.int(n as i64 + 1);
        if step_sign > 0 {
            // ∫₀^∞ tⁿ e^{(a − iω)t} dt = n!/(iω − a)^{n+1}, Re a < 0
            match a_sign {
                Some(s) if s < 0 => {}
                Some(0) if n == 0 => return Ok(None), // plain H(t): unary table
                Some(_) => {
                    return Err(fail(
                        OP,
                        "t^n e^{at} H(t) with a ≥ 0 is not Fourier transformable (as a function)",
                    ));
                }
                None => return Err(need_sign(OP, arena, a, "a < 0 in e^{at} H(t)")),
            }
            let den_base = arena.sub(iw, a);
            let den = arena.pow(den_base, np1);
            let r = arena.div(n_fact, den);
            return Ok(Some(arena.mul(&[eb, r])));
        }
        // H(−t): ∫_{−∞}^0 tⁿ e^{(a − iω)t} dt = (−1)ⁿ n!/(a − iω)^{n+1}, Re a > 0
        match a_sign {
            Some(s) if s > 0 => {}
            Some(0) if n == 0 => return Ok(None),
            Some(_) => {
                return Err(fail(
                    OP,
                    "t^n e^{at} H(-t) with a ≤ 0 is not Fourier transformable (as a function)",
                ));
            }
            None => return Err(need_sign(OP, arena, a, "a > 0 in e^{at} H(-t)")),
        }
        let den_base = arena.sub(a, iw);
        let den = arena.pow(den_base, np1);
        let r = arena.div(n_fact, den);
        let sign = if n.is_multiple_of(2) {
            arena.one()
        } else {
            arena.neg_one()
        };
        return Ok(Some(arena.mul(&[sign, eb, r])));
    }

    // ── tⁿ g(t) → iⁿ dⁿG/dωⁿ ──
    if t_power > 0 {
        let mut g_factors = others.clone();
        for (a, b) in &exp_lin {
            let arg = expr_or_lin(arena, *a, *b, t);
            g_factors.push(arena.exp(arg));
        }
        for (a, b) in &steps {
            let arg = expr_or_lin(arena, *a, *b, t);
            g_factors.push(arena.heaviside(arg));
        }
        if inv_t {
            g_factors.push(arena.pow(t, arena.neg_one()));
        }
        if g_factors.is_empty() {
            return Ok(None); // tⁿ alone: needs δ⁽ⁿ⁾
        }
        let g = arena.mul(&g_factors);
        let mut gf = forward(arena, g, t, w, depth + 1)?;
        let mut coeff = arena.one();
        for _ in 0..t_power {
            gf = crate::transforms::diff::diff(arena, gf, w);
            coeff = arena.mul(&[coeff, i]);
        }
        if crate::base::walk::has_unevaluated(arena, gf) {
            return Err(fail(OP, "result would require δ′ (derivative of a delta)"));
        }
        return Ok(Some(arena.mul(&[coeff, gf])));
    }

    Ok(None)
}

// ═══════════════════════════════════════════════════════════════════════════
// Shift / scaling rules (both directions)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy)]
enum Dir {
    Forward,
    Inverse,
}

/// Collect candidate shifts `c` (from sub-expressions `v + c`) and scalings
/// `a` (from `a·v`) occurring inside function arguments.
fn shift_scale_candidates(arena: &Arena, expr: ExprId, v: ExprId) -> (Vec<ExprId>, Vec<ExprId>) {
    let mut shifts = Vec::new();
    let mut scales = Vec::new();
    let mut stack = vec![expr];
    let mut visited = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        let node = arena.node(id);
        match node {
            ExprNode::Add(ch) if ch.contains(&v) => {
                let rest: Vec<ExprId> = ch.iter().copied().filter(|&c| c != v).collect();
                if !rest.is_empty()
                    && rest
                        .iter()
                        .all(|&c| !crate::base::walk::contains(arena, c, v))
                {
                    shifts.push(id);
                }
            }
            ExprNode::Mul(ch) if ch.contains(&v) => {
                let rest: Vec<ExprId> = ch.iter().copied().filter(|&c| c != v).collect();
                if !rest.is_empty()
                    && rest
                        .iter()
                        .all(|&c| !crate::base::walk::contains(arena, c, v))
                {
                    scales.push(id);
                }
            }
            _ => {}
        }
        node.for_each_child(|c| stack.push(c));
    }
    (shifts, scales)
}

fn try_shift_scale(
    arena: &mut Arena,
    expr: ExprId,
    src: ExprId,
    dst: ExprId,
    depth: usize,
    dir: Dir,
) -> Result<Option<ExprId>, SymplexError> {
    let (shifts, scales) = shift_scale_candidates(arena, expr, src);
    let i = arena.i_unit();

    // Shift: f(v) = g(v − v₀) with g(u) = f(u + v₀). At most three candidates.
    for add_id in shifts.into_iter().take(3) {
        let Some((a, c)) = linear_in(arena, add_id, src) else {
            continue;
        };
        if a != arena.one() {
            continue;
        }
        // sub-expression is v + c → v₀ = −c
        let v0 = arena.neg(c);
        let u_plus_v0 = arena.add(&[src, v0]);
        let g = crate::transforms::subs::subs(arena, expr, src, u_plus_v0);
        let g = crate::transforms::eval::eval(arena, g);
        if g == expr {
            continue;
        }
        let gt = match dir {
            Dir::Forward => forward(arena, g, src, dst, depth + 1),
            Dir::Inverse => inverse(arena, g, src, dst, depth + 1),
        };
        if let Ok(gt) = gt {
            // Forward: F{g(t − t₀)} = e^{−iωt₀} G(ω)
            // Inverse: F⁻¹{G(ω − ω₀)} = e^{iω₀t} g(t)
            let phase = match dir {
                Dir::Forward => {
                    let p = arena.mul(&[i, dst, v0]);
                    arena.neg(p)
                }
                Dir::Inverse => arena.mul(&[i, v0, dst]),
            };
            let e = arena.exp(phase);
            return Ok(Some(arena.mul(&[e, gt])));
        }
    }

    // Scaling: f(v) = g(a v) with g(u) = f(u / a) → F(ω) = G(ω/a)/|a|.
    for mul_id in scales.into_iter().take(3) {
        let Some((a, _)) = linear_in(arena, mul_id, src) else {
            continue;
        };
        if a == arena.one() || crate::base::walk::contains(arena, a, i) {
            continue;
        }
        let abs_a = match param_sign(arena, a) {
            Some(s) if s > 0 => a,
            Some(s) if s < 0 => arena.neg(a),
            _ => continue,
        };
        let u_over_a = arena.div(src, a);
        let g = crate::transforms::subs::subs(arena, expr, src, u_over_a);
        let g = crate::transforms::eval::eval(arena, g);
        if g == expr {
            continue;
        }
        let gt = match dir {
            Dir::Forward => forward(arena, g, src, dst, depth + 1),
            Dir::Inverse => inverse(arena, g, src, dst, depth + 1),
        };
        if let Ok(gt) = gt {
            let dst_over_a = arena.div(dst, a);
            let scaled = crate::transforms::subs::subs(arena, gt, dst, dst_over_a);
            let scaled = normalize_deltas(arena, scaled, dst);
            return Ok(Some(arena.div(scaled, abs_a)));
        }
    }

    Ok(None)
}

// ═══════════════════════════════════════════════════════════════════════════
// Inverse transform
// ═══════════════════════════════════════════════════════════════════════════

fn inverse(
    arena: &mut Arena,
    expr: ExprId,
    w: ExprId,
    t: ExprId,
    depth: usize,
) -> Result<ExprId, SymplexError> {
    const OP: &str = "inverse_fourier_transform";
    if depth > MAX_RULE_DEPTH {
        return Err(fail(OP, "rule nesting too deep"));
    }
    let expr = crate::transforms::eval::eval(arena, expr);
    let node = arena.node(expr).clone();

    if let ExprNode::Add(children) = &node {
        let kids = children.clone();
        let mut terms = Vec::with_capacity(kids.len());
        for child in kids {
            terms.push(inverse(arena, child, w, t, depth)?);
        }
        return Ok(arena.add(&terms));
    }
    if let ExprNode::Neg(inner) = node {
        let f = inverse(arena, inner, w, t, depth)?;
        return Ok(arena.neg(f));
    }

    // Constant c → c δ(t)
    if !crate::base::walk::contains(arena, expr, w) {
        let d = arena.dirac_delta(t);
        return Ok(arena.mul(&[expr, d]));
    }

    let (consts, deps) = split_factors(arena, expr, w);
    if !consts.is_empty() {
        let body = arena.mul(&deps);
        let f = inverse(arena, body, w, t, depth)?;
        let mut all = consts;
        all.push(f);
        return Ok(arena.mul(&all));
    }

    if let Some(r) = inverse_unary(arena, &node, w, t)? {
        return Ok(r);
    }

    // Rational functions of ω: 1/(iω − a)^n, 1/(ω² + a²), ω/(ω² + a²).
    if let Some(r) = inverse_rational(arena, expr, w, t)? {
        return Ok(r);
    }

    if let ExprNode::Mul(children) = &node {
        let kids: Vec<ExprId> = children.iter().copied().collect();
        if let Some(r) = inverse_product(arena, &kids, w, t, depth)? {
            return Ok(r);
        }
    }

    if let Some(r) = try_shift_scale(arena, expr, w, t, depth, Dir::Inverse)? {
        return Ok(r);
    }

    Err(fail(
        OP,
        format!("no inverse rule applies to {}", arena.display(expr)),
    ))
}

fn inverse_unary(
    arena: &mut Arena,
    node: &ExprNode,
    w: ExprId,
    t: ExprId,
) -> Result<Option<ExprId>, SymplexError> {
    const OP: &str = "inverse_fourier_transform";
    let i = arena.i_unit();
    let pi = arena.pi();
    let one = arena.one();
    let two = arena.int(2);
    match node.clone() {
        // δ(ω − ω₀) → e^{iω₀t}/(2π)
        ExprNode::DiracDelta(arg) => {
            let Some((a, b)) = linear_in(arena, arg, w) else {
                return Ok(None);
            };
            if arena.is_zero_structural(a) {
                return Ok(None);
            }
            let abs_a = match param_sign(arena, a) {
                Some(s) if s > 0 => a,
                Some(s) if s < 0 => arena.neg(a),
                Some(_) => return Ok(None),
                None => arena.abs(a),
            };
            // ω₀ = −b/a
            let w0 = arena.div(b, a);
            let w0 = arena.neg(w0);
            let phase = arena.mul(&[i, w0, t]);
            let e = arena.exp(phase);
            let tp = two_pi(arena);
            let den = arena.mul(&[tp, abs_a]);
            Ok(Some(arena.div(e, den)))
        }
        // H(ω) → δ(t)/2 + i/(2πt)
        ExprNode::Heaviside(arg) if arg == w => {
            let d = arena.dirac_delta(t);
            let half_d = arena.div(d, two);
            let tp = two_pi(arena);
            let den = arena.mul(&[tp, t]);
            let r = arena.div(i, den);
            Ok(Some(arena.add(&[half_d, r])))
        }
        ExprNode::Heaviside(arg) => {
            let Some((a, b)) = linear_in(arena, arg, w) else {
                return Ok(None);
            };
            if a == arena.neg_one() && arena.is_zero_structural(b) {
                let d = arena.dirac_delta(t);
                let half_d = arena.div(d, two);
                let tp = two_pi(arena);
                let den = arena.mul(&[tp, t]);
                let r = arena.div(i, den);
                return Ok(Some(arena.sub(half_d, r)));
            }
            Ok(None)
        }
        // sign(ω) → i/(πt)
        ExprNode::Sign(arg) if arg == w => {
            let den = arena.mul(&[pi, t]);
            Ok(Some(arena.div(i, den)))
        }
        // 1/ω → i sign(t)/2
        ExprNode::Pow(base, e) if base == w && e == arena.neg_one() => {
            let s = arena.sign(t);
            let r = arena.mul(&[i, s]);
            Ok(Some(arena.div(r, two)))
        }
        // e^{−bω²} → e^{−t²/(4b)}/(2√(πb))
        ExprNode::Exp(arg) => {
            if let Some(b) = neg_coeff_times_square(arena, arg, w) {
                return match param_sign(arena, b) {
                    Some(s) if s > 0 => {
                        let t2 = arena.pow(t, two);
                        let four = arena.int(4);
                        let four_b = arena.mul(&[four, b]);
                        let q = arena.div(t2, four_b);
                        let nq = arena.neg(q);
                        let e = arena.exp(nq);
                        // 1/(2√(πb)) written as ½·π^{-½}·b^{-½} so numeric `b` folds.
                        let neg_half = arena.rational(-1, 2);
                        let pi_r = arena.pow(pi, neg_half);
                        let inv_b = arena.div(one, b);
                        let b_r = arena.sqrt(inv_b);
                        let half = arena.rational(1, 2);
                        Ok(Some(arena.mul(&[half, pi_r, b_r, e])))
                    }
                    Some(_) => Err(fail(OP, "exp(bω²) with b ≥ 0 has no inverse transform")),
                    None => Err(need_sign(OP, arena, b, "b > 0 in exp(-bω²)")),
                };
            }
            // e^{−iωt₀} alone → δ(t − t₀)
            if let Some((a, b)) = linear_in(arena, arg, w)
                && arena.is_zero_structural(b)
            {
                let (re, im) = split_real_imag(arena, a);
                if arena.is_zero_structural(re) {
                    // e^{i·im·ω} = e^{−iω t₀} with t₀ = −im → δ(t + im)
                    let inner = arena.add(&[t, im]);
                    return Ok(Some(arena.dirac_delta(inner)));
                }
            }
            Ok(None)
        }
        // cos(aω) / sin(aω): cos(aω) = (e^{iaω} + e^{−iaω})/2 → [δ(t + a) + δ(t − a)]/2
        ExprNode::Cos(arg) | ExprNode::Sin(arg) => {
            let Some((a, b)) = linear_in(arena, arg, w) else {
                return Ok(None);
            };
            if !arena.is_zero_structural(b) {
                return Ok(None);
            }
            let tp = arena.add(&[t, a]);
            let tm = arena.sub(t, a);
            let dp = arena.dirac_delta(tp);
            let dm = arena.dirac_delta(tm);
            if matches!(node, ExprNode::Cos(_)) {
                let s = arena.add(&[dp, dm]);
                Ok(Some(arena.div(s, two)))
            } else {
                // sin(aω) = (e^{iaω} − e^{−iaω})/(2i) → [δ(t + a) − δ(t − a)]/(2i)
                let d = arena.sub(dp, dm);
                let two_i = arena.mul(&[two, i]);
                Ok(Some(arena.div(d, two_i)))
            }
        }
        _ => {
            let _ = one;
            Ok(None)
        }
    }
}

/// `1/(iω − a)^n`, `1/(ω² + a²)`, `ω/(ω² + a²)`.
fn inverse_rational(
    arena: &mut Arena,
    expr: ExprId,
    w: ExprId,
    t: ExprId,
) -> Result<Option<ExprId>, SymplexError> {
    const OP: &str = "inverse_fourier_transform";
    let i = arena.i_unit();
    let two = arena.int(2);
    let (consts, deps) = split_factors(arena, expr, w);
    let _ = consts;
    // Find a single Pow(base, −n) with the rest being ω-powers.
    let mut pow_part: Option<(ExprId, u64)> = None;
    let mut w_power: u64 = 0;
    for &d in &deps {
        if let Some(n) = power_of(arena, d, w) {
            w_power += n;
            continue;
        }
        if let ExprNode::Pow(base, e) = arena.node(d).clone()
            && let Some(r) = arena.as_num(e).cloned()
            && r.is_integer()
            && r.is_negative()
            && pow_part.is_none()
        {
            let n = (-r.to_integer()).to_u64().unwrap_or(0);
            pow_part = Some((base, n));
            continue;
        }
        return Ok(None);
    }
    let Some((base, n)) = pow_part else {
        return Ok(None);
    };
    if n == 0 {
        return Ok(None);
    }

    // base = i·ω − a  (linear in ω with imaginary slope)
    if let Some((slope, c)) = linear_in(arena, base, w)
        && w_power == 0
    {
        let (re, im) = split_real_imag(arena, slope);
        if !arena.is_zero_structural(re) || arena.is_zero_structural(im) {
            return Ok(None);
        }
        // base = im·(iω + c/im) ; normalise to (iω − a) with a = −c/im.
        let a = arena.div(c, im);
        let a = arena.neg(a);
        let a = crate::transforms::eval::eval(arena, a);
        let n_id = arena.int(n as i64);
        let scale = arena.pow(im, n_id);
        let one = arena.one();
        let scale_inv = arena.div(one, scale);
        match param_sign(arena, a) {
            Some(s) if s < 0 => {
                // 1/(iω − a)^n → t^{n−1} e^{at} H(t)/(n−1)!
                let at = arena.mul(&[a, t]);
                let e = arena.exp(at);
                let h = arena.heaviside(t);
                let nm1 = n - 1;
                let tp = if nm1 == 0 {
                    arena.one()
                } else {
                    let k = arena.int(nm1 as i64);
                    arena.pow(t, k)
                };
                let f = factorial_expr(arena, nm1);
                let num = arena.mul(&[tp, e, h]);
                let r = arena.div(num, f);
                Ok(Some(arena.mul(&[scale_inv, r])))
            }
            Some(s) if s > 0 => {
                // 1/(iω − a)^n = (−1)ⁿ/(a − iω)^n → (−1)ⁿ (−1)^{n−1} t^{n−1} e^{at} H(−t)/(n−1)!
                //              = −t^{n−1} e^{at} H(−t)/(n−1)!
                let at = arena.mul(&[a, t]);
                let e = arena.exp(at);
                let nt = arena.neg(t);
                let h = arena.heaviside(nt);
                let nm1 = n - 1;
                let tp = if nm1 == 0 {
                    arena.one()
                } else {
                    let k = arena.int(nm1 as i64);
                    arena.pow(t, k)
                };
                let f = factorial_expr(arena, nm1);
                let num = arena.mul(&[tp, e, h]);
                let r = arena.div(num, f);
                let r = arena.neg(r);
                Ok(Some(arena.mul(&[scale_inv, r])))
            }
            Some(_) => {
                if n == 1 {
                    // 1/(iω) → sign(t)/2
                    let s = arena.sign(t);
                    let r = arena.div(s, two);
                    Ok(Some(arena.mul(&[scale_inv, r])))
                } else {
                    Err(fail(
                        OP,
                        "1/(iω)^n with n > 1 requires |t|-type distributions",
                    ))
                }
            }
            None => Err(need_sign(OP, arena, a, "the sign of a in 1/(iω − a)")),
        }
    } else if let ExprNode::Add(ch) = arena.node(base).clone()
        && n == 1
        && w_power <= 1
    {
        // base = ω² + a²  (a² a positive constant)
        let w2 = arena.pow(w, two);
        let mut a2: Option<ExprId> = None;
        let mut ok = ch.contains(&w2) && ch.len() == 2;
        if ok {
            for &c in &ch {
                if c != w2 {
                    if crate::base::walk::contains(arena, c, w) {
                        ok = false;
                    } else {
                        a2 = Some(c);
                    }
                }
            }
        }
        let (true, Some(a2)) = (ok, a2) else {
            return Ok(None);
        };
        match param_sign(arena, a2) {
            Some(s) if s > 0 => {}
            Some(_) => return Ok(None),
            None => return Err(need_sign(OP, arena, a2, "a² > 0 in 1/(ω² + a²)")),
        }
        let a = positive_sqrt(arena, a2);
        let abs_t = arena.abs(t);
        let neg_a_abs_t = arena.mul(&[a, abs_t]);
        let neg_a_abs_t = arena.neg(neg_a_abs_t);
        let e = arena.exp(neg_a_abs_t);
        if w_power == 0 {
            // 1/(ω² + a²) → e^{−a|t|}/(2a)
            let den = arena.mul(&[two, a]);
            Ok(Some(arena.div(e, den)))
        } else {
            // ω/(ω² + a²) → (i/2) sign(t) e^{−a|t|}
            let s = arena.sign(t);
            let r = arena.mul(&[i, s, e]);
            Ok(Some(arena.div(r, two)))
        }
    } else {
        Ok(None)
    }
}

/// Products in ω: `sin(aω)/ω` (rect), modulation `e^{−iωt₀} G(ω)`, and
/// `ωⁿ G(ω) → (−i)ⁿ g⁽ⁿ⁾(t)`.
fn inverse_product(
    arena: &mut Arena,
    kids: &[ExprId],
    w: ExprId,
    t: ExprId,
    depth: usize,
) -> Result<Option<ExprId>, SymplexError> {
    const OP: &str = "inverse_fourier_transform";
    let i = arena.i_unit();
    let two = arena.int(2);

    let mut w_power: u64 = 0;
    let mut inv_w = false;
    let mut phases: Vec<ExprId> = Vec::new(); // e^{i·θ(ω)} factors with θ linear in ω, real
    let mut others: Vec<ExprId> = Vec::new();
    for &k in kids {
        if let Some(n) = power_of(arena, k, w) {
            w_power += n;
            continue;
        }
        match arena.node(k).clone() {
            ExprNode::Pow(base, e) if base == w && e == arena.neg_one() => inv_w = true,
            ExprNode::Exp(arg) => {
                if let Some((a, b)) = linear_in(arena, arg, w)
                    && arena.is_zero_structural(b)
                {
                    let (re, im) = split_real_imag(arena, a);
                    if arena.is_zero_structural(re) {
                        phases.push(im);
                        continue;
                    }
                }
                others.push(k);
            }
            _ => others.push(k),
        }
    }

    // ── Modulation in ω: e^{−iωt₀} G(ω) → g(t − t₀); here e^{i·im·ω}, t₀ = −im ──
    if !phases.is_empty() {
        let total = arena.add(&phases);
        let t0 = arena.neg(total);
        let mut g_factors = others.clone();
        if inv_w {
            g_factors.push(arena.pow(w, arena.neg_one()));
        }
        if w_power > 0 {
            let n = arena.int(w_power as i64);
            g_factors.push(arena.pow(w, n));
        }
        let g = if g_factors.is_empty() {
            arena.one()
        } else {
            arena.mul(&g_factors)
        };
        let gi = inverse(arena, g, w, t, depth + 1)?;
        let shifted_t = arena.sub(t, t0);
        let shifted = crate::transforms::subs::subs(arena, gi, t, shifted_t);
        return Ok(Some(shifted));
    }

    // ── sin(aω)/ω → [H(t + a) − H(t − a)]/2 ──
    if inv_w && w_power == 0 && others.len() == 1 {
        if let ExprNode::Sin(arg) = arena.node(others[0]).clone()
            && let Some((a, b)) = linear_in(arena, arg, w)
            && arena.is_zero_structural(b)
        {
            let (sgn, abs_a) = match param_sign(arena, a) {
                Some(s) if s > 0 => (arena.one(), a),
                Some(s) if s < 0 => (arena.neg_one(), arena.neg(a)),
                Some(_) => return Ok(Some(arena.zero())),
                None => return Err(need_sign(OP, arena, a, "the sign of a in sin(aω)/ω")),
            };
            let tp = arena.add(&[t, abs_a]);
            let tm = arena.sub(t, abs_a);
            let hp = arena.heaviside(tp);
            let hm = arena.heaviside(tm);
            let rect = arena.sub(hp, hm);
            let r = arena.mul(&[sgn, rect]);
            return Ok(Some(arena.div(r, two)));
        }
        return Ok(None);
    }

    // ── ωⁿ G(ω) → (−i)ⁿ dⁿg/dtⁿ ──
    if w_power > 0 && !others.is_empty() {
        let mut g_factors = others.clone();
        if inv_w {
            g_factors.push(arena.pow(w, arena.neg_one()));
        }
        let g = arena.mul(&g_factors);
        let mut gi = inverse(arena, g, w, t, depth + 1)?;
        let mut coeff = arena.one();
        let neg_i = arena.neg(i);
        for _ in 0..w_power {
            gi = crate::transforms::diff::diff(arena, gi, t);
            coeff = arena.mul(&[coeff, neg_i]);
        }
        if crate::base::walk::has_unevaluated(arena, gi) {
            return Err(fail(OP, "result would require δ′ (derivative of a delta)"));
        }
        return Ok(Some(arena.mul(&[coeff, gi])));
    }

    Ok(None)
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

    #[test]
    fn forward_delta() {
        let mut a = Arena::new();
        let t = sym(&mut a, "t");
        let omega = sym(&mut a, "omega");
        let delta_t = a.dirac_delta(t);
        let result = fourier_transform(&mut a, delta_t, t, omega).unwrap();
        assert_eq!(result, a.one(), "F{{δ(t)}} should be 1");
    }

    #[test]
    fn forward_constant() {
        let mut a = Arena::new();
        let t = sym(&mut a, "t");
        let omega = sym(&mut a, "omega");
        let five = a.int(5);
        let result = fourier_transform(&mut a, five, t, omega).unwrap();
        let ten = a.int(10);
        let pi = a.pi();
        let d = a.dirac_delta(omega);
        let expected = a.mul(&[ten, pi, d]);
        assert_eq!(result, expected, "{}", display(&a, result));
    }

    #[test]
    fn forward_sin_cos() {
        let mut a = Arena::new();
        let t = sym(&mut a, "t");
        let omega = sym(&mut a, "omega");
        let three = a.int(3);
        let three_t = a.mul(&[three, t]);
        let sin_3t = a.sin(three_t);
        let result = fourier_transform(&mut a, sin_3t, t, omega).unwrap();
        let pi = a.pi();
        let i = a.i_unit();
        let wp = a.add(&[omega, three]);
        let wm = a.sub(omega, three);
        let dp = a.dirac_delta(wp);
        let dm = a.dirac_delta(wm);
        // iπ[δ(ω+3) − δ(ω−3)]
        let tp = a.mul(&[i, pi, dp]);
        let tm = a.mul(&[i, pi, dm]);
        let expected = a.sub(tp, tm);
        assert_eq!(result, expected, "{}", display(&a, result));
        let cos_3t = a.cos(three_t);
        let result = fourier_transform(&mut a, cos_3t, t, omega).unwrap();
        let tp = a.mul(&[pi, dp]);
        let tm = a.mul(&[pi, dm]);
        let expected = a.add(&[tp, tm]);
        assert_eq!(result, expected, "{}", display(&a, result));
    }

    #[test]
    fn forward_exp_heaviside() {
        let mut a = Arena::new();
        let t = sym(&mut a, "t");
        let omega = sym(&mut a, "omega");
        let neg2 = a.int(-2);
        let neg2_t = a.mul(&[neg2, t]);
        let exp_neg2t = a.exp(neg2_t);
        let heaviside_t = a.heaviside(t);
        let expr = a.mul(&[exp_neg2t, heaviside_t]);
        let result = fourier_transform(&mut a, expr, t, omega).unwrap();
        let i = a.i_unit();
        let two = a.int(2);
        let iw = a.mul(&[i, omega]);
        let den = a.add(&[iw, two]);
        let one = a.one();
        let expected = a.div(one, den);
        assert_eq!(result, expected, "{}", display(&a, result));
    }

    #[test]
    fn forward_growing_exponential_is_rejected() {
        let mut a = Arena::new();
        let t = sym(&mut a, "t");
        let omega = sym(&mut a, "omega");
        let two = a.int(2);
        let two_t = a.mul(&[two, t]);
        let e = a.exp(two_t);
        let h = a.heaviside(t);
        let expr = a.mul(&[e, h]);
        assert!(fourier_transform(&mut a, expr, t, omega).is_err());
    }

    #[test]
    fn forward_linearity() {
        let mut a = Arena::new();
        let t = sym(&mut a, "t");
        let omega = sym(&mut a, "omega");
        let delta_t = a.dirac_delta(t);
        let three = a.int(3);
        let expr = a.mul(&[three, delta_t]);
        let result = fourier_transform(&mut a, expr, t, omega).unwrap();
        assert_eq!(display(&a, result), "3");
    }

    #[test]
    fn forward_t_must_be_symbol() {
        let mut a = Arena::new();
        let t = a.int(5);
        let omega = sym(&mut a, "omega");
        assert!(matches!(
            fourier_transform(&mut a, t, t, omega),
            Err(SymplexError::InvalidArgument { .. })
        ));
    }

    #[test]
    fn forward_heaviside() {
        let mut a = Arena::new();
        let t = sym(&mut a, "t");
        let omega = sym(&mut a, "omega");
        let heaviside_t = a.heaviside(t);
        let result = fourier_transform(&mut a, heaviside_t, t, omega).unwrap();
        // πδ(ω) + 1/(iω)
        let pi = a.pi();
        let d = a.dirac_delta(omega);
        let pd = a.mul(&[pi, d]);
        let i = a.i_unit();
        let iw = a.mul(&[i, omega]);
        let one = a.one();
        let r = a.div(one, iw);
        let expected = a.add(&[pd, r]);
        assert_eq!(result, expected, "{}", display(&a, result));
    }

    #[test]
    fn forward_two_sided_exponential() {
        let mut a = Arena::new();
        let t = sym(&mut a, "t");
        let omega = sym(&mut a, "omega");
        let abs_t = a.abs(t);
        let three = a.int(3);
        let arg = a.mul(&[three, abs_t]);
        let narg = a.neg(arg);
        let e = a.exp(narg);
        let result = fourier_transform(&mut a, e, t, omega).unwrap();
        assert_eq!(display(&a, result), "6/(omega^2 + 9)");
    }

    #[test]
    fn inverse_constant() {
        let mut a = Arena::new();
        let t = sym(&mut a, "t");
        let omega = sym(&mut a, "omega");
        let three = a.int(3);
        let result = inverse_fourier_transform(&mut a, three, omega, t).unwrap();
        assert_eq!(display(&a, result), "3*DiracDelta(t)");
    }

    #[test]
    fn inverse_delta_omega() {
        let mut a = Arena::new();
        let t = sym(&mut a, "t");
        let omega = sym(&mut a, "omega");
        let delta_omega = a.dirac_delta(omega);
        let result = inverse_fourier_transform(&mut a, delta_omega, omega, t).unwrap();
        let two = a.int(2);
        let pi = a.pi();
        let two_pi = a.mul(&[two, pi]);
        let one = a.one();
        let expected = a.div(one, two_pi);
        assert_eq!(result, expected, "{}", display(&a, result));
    }

    #[test]
    fn inverse_one_over_i_omega_is_half_sign() {
        let mut a = Arena::new();
        let t = sym(&mut a, "t");
        let omega = sym(&mut a, "omega");
        let i = a.i_unit();
        let iw = a.mul(&[i, omega]);
        let e = a.div(a.one(), iw);
        let result = inverse_fourier_transform(&mut a, e, omega, t).unwrap();
        assert_eq!(display(&a, result), "1/2*sign(t)");
    }

    #[test]
    fn linear_in_detects_affine_forms() {
        let mut a = Arena::new();
        let t = sym(&mut a, "t");
        let c = sym(&mut a, "c");
        let three = a.int(3);
        let e = a.mul(&[three, t]);
        let e = a.add(&[e, c]);
        let (slope, off) = linear_in(&mut a, e, t).unwrap();
        assert_eq!(slope, three);
        assert_eq!(off, c);
        let two = a.int(2);
        let t2 = a.pow(t, two);
        assert!(linear_in(&mut a, t2, t).is_none());
    }
}

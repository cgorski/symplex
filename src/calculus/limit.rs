//! Symbolic limit computation — two-sided and one-sided.
//!
//! [`limit`] computes the two-sided limit `lim_{var → point} expr`;
//! [`limit_dir`] additionally takes a [`Direction`] for one-sided limits
//! `x → a⁺` / `x → a⁻`.
//!
//! # Algorithm
//!
//! 1. **Constant:** if `expr` does not depend on `var`, return `eval(expr)`.
//! 2. **Point at ±∞:** the `1^∞` heuristic for `Pow` nodes, then a
//!    compositional rule for an outer unary function (`atan(∞) = π/2`,
//!    `Γ(∞) = ∞`, …), then the Gruntz algorithm, then polynomial-degree
//!    analysis as a last resort.
//! 3. **Finite point:**
//!    - **Safe direct substitution.** Every `var`-dependent sub-expression is
//!      evaluated at the point bottom-up; if none is singular (`∞`, `zoo`,
//!      `NaN`, `ln 0`, `0^0`, a pole of `Γ`, …) and no discontinuous node
//!      (`sign`, `H`, `⌊·⌋`, `⌈·⌉`, `Piecewise`) sits exactly at its
//!      discontinuity, the substituted value *is* the limit.
//!    - **Compositional rule** for an outer continuous unary function.
//!    - **One-sided Gruntz.** `x = a ± 1/w` reduces `x → a^±` to
//!      `w → +∞`, which the Gruntz algorithm handles (with `Abs`, `Sign`,
//!      `Heaviside`, `Floor`, `Ceiling`, `Piecewise`, `Min`, `Max` resolved
//!      by their eventual sign as `w → ∞`). For [`Direction::Both`] both
//!      one-sided limits are computed and must agree.
//!    - **L'Hôpital + series fallback** for expressions without
//!      discontinuous nodes.
//!
//! Every candidate result is validated: it must be free of `var`, free of
//! internal dummy symbols, free of unevaluated nodes, and must not be `NaN` or
//! complex infinity. `+∞` and `−∞` are legitimate limit values, and so are
//! the infinities of complex-valued functions written as `subs` writes values
//! at infinities: `oo + I·b` (`Re f → ∞`, `Im f → b`: `ln(−x) → oo + I·π`),
//! `I·oo + a`, and `c·oo` for any other direction (`e^{x + 2i} → e^{2i}·oo`);
//! see [`LimVal`].
//!
//! # Termination
//!
//! Every recursion is depth-capped, and the Gruntz engine additionally
//! carries a work budget (see `gruntz::Budget`) that bounds the total number
//! of MRV rewrites, leading-term extractions and series terms, and refuses
//! to expand intermediate expressions above a fixed size. A pathological
//! input therefore returns an unevaluated `Limit` node quickly instead of
//! spinning.

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{Signed, Zero};
use rustc_hash::FxHashMap;

use crate::base::arena::Arena;
use crate::base::errors::SymplexError;
use crate::base::extended::Extended;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::Q;
use crate::calculus::calculus_util::is_hidden_zero;

/// Maximum L'Hôpital iterations to prevent infinite loops.
const MAX_LHOPITAL: usize = 5;

/// Maximum nesting depth for the compositional rule.
const MAX_COMPOSE_DEPTH: usize = 48;

/// Name of the dummy used for the `x = a ± 1/w` substitution.
const LIM_DUMMY: &str = "__lim_w";

/// The direction from which a limit point is approached.
///
/// ```
/// use symplex::prelude::*;
///
/// let ctx = Context::new();
/// let x = ctx.symbol("x");
/// let f = 1 / &x;
/// assert_eq!(format!("{}", f.limit_right(&x, &ctx.int(0))), "oo");
/// assert_eq!(format!("{}", f.limit_left(&x, &ctx.int(0))), "-oo");
/// // The two-sided limit does not exist:
/// assert!(f.limit(&x, &ctx.int(0)).has_unevaluated());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Direction {
    /// Approach from below: `x → a⁻` (`x < a`).
    Left,
    /// Approach from above: `x → a⁺` (`x > a`).
    Right,
    /// Two-sided limit (the default). Both one-sided limits must exist and
    /// agree; otherwise the limit is reported as non-existent.
    #[default]
    Both,
}

fn fail(reason: impl Into<String>) -> SymplexError {
    SymplexError::ComputationFailed {
        operation: "limit",
        reason: reason.into(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Limit values: finite constants and infinities with a direction
// ═══════════════════════════════════════════════════════════════════════════
//
// An infinite limit of a complex-valued function is written as the values
// at infinities are (0.41): `±oo + i·b` when `Re f → ±∞` and `Im f → b`
// (`ln(−x) → oo + iπ`), `±I·oo + a` when `Im f → ±∞` and `Re f → a`, and
// `c·oo` for any other direction `c/|c|` of `f` (`e^{x + 2i} → e^{2i}·oo`),
// which claims nothing about a finite part.  A plain `±oo` (`±I·oo`) is
// also the value when the perpendicular component diverges more slowly
// than the parallel one (`x + i√x → oo`): such a value is *weak*, and
// the rules that need the perpendicular component (`exp`, `ln` at `−∞`)
// refuse it.  Up to 0.41 every infinite limit was `±oo`: `limit(x − i, x,
// ∞)` was `oo` while `subs` gave `oo − I`, `e^{x + 2i}` tended to `oo`.

/// A computed limit value; `weak` marks an infinity on an axis whose
/// perpendicular component is known to diverge (see above).
#[derive(Clone, Copy, Debug)]
struct LimVal {
    value: ExprId,
    weak: bool,
}

impl LimVal {
    fn exact(value: ExprId) -> Self {
        LimVal { value, weak: false }
    }
}

/// The direction of an infinite value `d·∞`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Axis {
    RealPos,
    RealNeg,
    ImagPos,
    ImagNeg,
    Other,
}

/// An infinite limit `d·∞ + o`: `|f| → ∞` and `f/|f| → d/|d|`; on an axis,
/// `offset` is a constant whose component perpendicular to the axis is the
/// limit of `f`'s (`None`: that component diverges, or off the axes).
#[derive(Clone, Copy, Debug)]
struct Infinite {
    dir: ExprId,
    axis: Axis,
    offset: Option<ExprId>,
}

enum Kind {
    Finite(ExprId),
    Infinite(Infinite),
}

/// The approach `var → point` (from `dir`).
#[derive(Clone, Copy)]
struct Approach {
    var: ExprId,
    point: ExprId,
    dir: Direction,
}

/// The direction factor of an infinite term `±oo`, `c·oo`, `c·(−oo)`.
fn infinite_term_dir(arena: &mut Arena, t: ExprId) -> Option<ExprId> {
    let (inf, ninf) = (arena.infinity(), arena.neg_infinity());
    if t == inf {
        return Some(arena.one());
    }
    if t == ninf {
        return Some(arena.neg_one());
    }
    match arena.node(t).clone() {
        ExprNode::Neg(a) => {
            let d = infinite_term_dir(arena, a)?;
            Some(arena.neg(d))
        }
        ExprNode::Mul(cs) => {
            let mut sign = 0;
            let mut others: Vec<ExprId> = Vec::with_capacity(cs.len());
            for c in cs {
                if c == inf || c == ninf {
                    if sign != 0 {
                        return None;
                    }
                    sign = if c == inf { 1 } else { -1 };
                } else if contains_singular_atom(arena, c) {
                    return None;
                } else {
                    others.push(c);
                }
            }
            if sign == 0 {
                return None;
            }
            let d = arena.mul(&others);
            Some(if sign > 0 { d } else { arena.neg(d) })
        }
        _ => None,
    }
}

/// `(d, o)` for a value `d·∞ + o`; `None` for a finite value (or a
/// malformed one).
fn split_infinite(arena: &mut Arena, v: ExprId) -> Option<(ExprId, ExprId)> {
    if let Some(d) = infinite_term_dir(arena, v) {
        return Some((d, arena.zero()));
    }
    let ExprNode::Add(cs) = arena.node(v).clone() else {
        return None;
    };
    let mut dir = None;
    let mut rest: Vec<ExprId> = Vec::with_capacity(cs.len());
    for c in cs {
        if let Some(d) = infinite_term_dir(arena, c) {
            if dir.is_some() {
                return None;
            }
            dir = Some(d);
        } else if contains_singular_atom(arena, c) {
            return None;
        } else {
            rest.push(c);
        }
    }
    let d = dir?;
    Some((d, arena.add(&rest)))
}

/// Is `v` an infinity written with a direction or a finite part (`oo + c`,
/// `c·oo`, `c·oo + d`, finite `c ≠ 0`, `d`)?  (The structural test of
/// [`split_infinite`], without building nodes.)
fn is_infinite_form(arena: &Arena, v: ExprId) -> bool {
    let (inf, ninf) = (arena.infinity(), arena.neg_infinity());
    let term = |t: ExprId| -> bool {
        let t = match arena.node(t) {
            ExprNode::Neg(a) => *a,
            _ => t,
        };
        if t == inf || t == ninf {
            return true;
        }
        let ExprNode::Mul(cs) = arena.node(t) else {
            return false;
        };
        let infinite = cs.iter().filter(|&&c| c == inf || c == ninf).count();
        infinite == 1
            && cs
                .iter()
                .all(|&c| c == inf || c == ninf || !contains_singular_atom(arena, c))
    };
    if term(v) {
        return true;
    }
    let ExprNode::Add(cs) = arena.node(v) else {
        return false;
    };
    let infinite = cs.iter().filter(|&&c| term(c)).count();
    infinite == 1
        && cs
            .iter()
            .all(|&c| term(c) || !contains_singular_atom(arena, c))
}

/// Real and imaginary parts of a constant, when exactly known.
fn const_parts(arena: &mut Arena, c: ExprId) -> Option<(ExprId, ExprId)> {
    if arena.as_num(c).is_some() {
        return Some((c, arena.zero()));
    }
    let p = crate::base::complex::decompose(arena, c);
    if !p.exact {
        // A real constant the decomposition does not see through
        // (`cosh(acos 4) = cos(acosh 4)`), certified numerically.
        let mut reals = crate::base::assumptions::AssumptionCache::new();
        return (crate::transforms::realness::constant_realness_as_declared(
            arena,
            c,
            NUMERIC_SIGN_DIGITS,
            &mut reals,
        ) == Some(true))
        .then(|| (c, arena.zero()));
    }
    let re = crate::transforms::eval::eval(arena, p.re);
    let im = crate::transforms::eval::eval(arena, p.im);
    Some((re, im))
}

/// The axis of a direction constant `d`, from the signs of its parts.
fn direction_axis(arena: &mut Arena, d: ExprId) -> Option<Axis> {
    let (re, im) = const_parts(arena, d)?;
    let sr = const_sign(arena, re)?;
    let si = const_sign(arena, im)?;
    Some(match (sr, si) {
        (0, 0) => return None,
        (s, 0) => {
            if s > 0 {
                Axis::RealPos
            } else {
                Axis::RealNeg
            }
        }
        (0, s) => {
            if s > 0 {
                Axis::ImagPos
            } else {
                Axis::ImagNeg
            }
        }
        _ => Axis::Other,
    })
}

/// The kind of a computed limit value (`None` for a malformed one).
fn kind_of(arena: &mut Arena, lv: LimVal) -> Option<Kind> {
    let v = lv.value;
    if !contains_singular_atom(arena, v) {
        return Some(Kind::Finite(v));
    }
    let (dir, offset) = split_infinite(arena, v)?;
    let axis = direction_axis(arena, dir)?;
    let offset = if lv.weak || axis == Axis::Other {
        None
    } else {
        Some(offset)
    };
    Some(Kind::Infinite(Infinite { dir, axis, offset }))
}

/// The value of an infinite limit (see the section comment); `None` when
/// it is not a valid limit value.
fn infinite_value(arena: &mut Arena, inf: Infinite, var: ExprId) -> Option<LimVal> {
    let oo = arena.infinity();
    let i = arena.i_unit();
    let base = match inf.axis {
        Axis::RealPos => oo,
        Axis::RealNeg => arena.neg_infinity(),
        Axis::ImagPos => arena.mul(&[i, oo]),
        Axis::ImagNeg => {
            let ni = arena.neg(i);
            arena.mul(&[ni, oo])
        }
        Axis::Other => arena.mul(&[inf.dir, oo]),
    };
    // Only the perpendicular component of the offset counts (`oo + c`
    // absorbs a real `c` itself, but not a real part it cannot see:
    // `ln(3 + i) − ln(3 − i)` is `2i·atan(1/3)`).
    let without_arg = |arena: &Arena, e: ExprId| {
        !crate::base::walk::post_order_ids(arena, e)
            .into_iter()
            .any(|id| {
                matches!(
                    arena.node(id),
                    ExprNode::Arg(_) | ExprNode::Re(_) | ExprNode::Im(_)
                )
            })
    };
    let value = match (inf.axis, inf.offset) {
        (Axis::Other, _) | (_, None) => base,
        (Axis::RealPos | Axis::RealNeg, Some(o)) => {
            let perp = match const_parts(arena, o) {
                Some((_, im)) if without_arg(arena, im) => {
                    let p = arena.mul(&[i, im]);
                    crate::transforms::eval::eval(arena, p)
                }
                _ => o,
            };
            // The shorter of the two spellings.
            let size = |arena: &Arena, e: ExprId| {
                crate::transforms::pattern::tree_size_capped(arena, e, MAX_EXPAND_SIZE)
            };
            let o = if size(arena, perp) <= size(arena, o) {
                perp
            } else {
                o
            };
            arena.add(&[base, o])
        }
        (_, Some(o)) => {
            let re = match const_parts(arena, o) {
                Some((re, _)) if without_arg(arena, re) => re,
                _ => o,
            };
            arena.add(&[base, re])
        }
    };
    let value = crate::transforms::eval::eval(arena, value);
    let weak = inf.offset.is_none() && inf.axis != Axis::Other;
    is_valid_limit_value(arena, value, var).then_some(LimVal { value, weak })
}

/// The principal argument of a direction constant, when known exactly
/// (`arg(e^{2i}) = 2`, `arg(1 + i) = π/4`).
fn const_arg(arena: &mut Arena, d: ExprId) -> Option<ExprId> {
    let a = arena.arg(d);
    let a = crate::transforms::eval::eval(arena, a);
    let opaque = crate::base::walk::post_order_ids(arena, a)
        .into_iter()
        .any(|id| matches!(arena.node(id), ExprNode::Arg(_)));
    if !opaque {
        return Some(a);
    }
    // `r·e^{iθ}` with `r > 0` and `−π < θ < π`.
    let factors: Vec<ExprId> = match arena.node(d) {
        ExprNode::Mul(cs) => cs.to_vec(),
        _ => vec![d],
    };
    let mut theta = None;
    for f in factors {
        match arena.node(f).clone() {
            ExprNode::Exp(w) if theta.is_none() => {
                let (re, im) = const_parts(arena, w)?;
                if !arena.is_zero_structural(re) {
                    return None;
                }
                theta = Some(im);
            }
            _ => {
                if const_sign(arena, f) != Some(1) {
                    return None;
                }
            }
        }
    }
    let theta = theta?;
    let v = crate::transforms::evalf::evalf_f64(arena, theta).ok()?;
    (v.abs() < std::f64::consts::PI - 1e-9).then_some(theta)
}

/// `d/|d|`, the value of `sign` at the infinity `d·∞`.
fn unit_direction(arena: &mut Arena, d: ExprId) -> ExprId {
    let a = arena.abs(d);
    let q = arena.div(d, a);
    crate::transforms::eval::eval(arena, q)
}

/// Is `g` real for every `var` near the limit point along the approach?
/// (`x → ∞` along the positive reals, `a±` from the side; other symbols
/// keep their assumptions: an unassumed one is complex.)
fn eventually_real(arena: &mut Arena, g: ExprId, ap: &Approach) -> bool {
    if !crate::base::walk::contains(arena, g, ap.var) {
        let mut cache = crate::base::assumptions::AssumptionCache::new();
        return cache.query(arena, g, crate::base::assumptions::Props::REAL) == Some(true);
    }
    let t = arena.positive_symbol("__lim_r");
    for &right in approach_sides(ap.dir) {
        let sub = approach_substitute(arena, ap, t, right);
        let e = crate::transforms::subs::subs(arena, g, ap.var, sub);
        let e = crate::transforms::eval::eval(arena, e);
        if !crate::calculus::gruntz::eventually_real_at_inf(arena, e, t) {
            return false;
        }
    }
    true
}

/// The imaginary part of a constant offset, when exactly known.
fn offset_im(arena: &mut Arena, o: ExprId) -> Option<ExprId> {
    const_parts(arena, o).map(|(_, im)| im)
}

/// The eventual sign of `Im g` for an infinite `g` (`None`: unknown).
fn im_sign_of(arena: &mut Arena, inf: &Infinite) -> Option<i32> {
    match inf.axis {
        Axis::ImagPos => Some(1),
        Axis::ImagNeg => Some(-1),
        Axis::Other => {
            let (_, im) = const_parts(arena, inf.dir)?;
            const_sign(arena, im)
        }
        Axis::RealPos | Axis::RealNeg => {
            let b = offset_im(arena, inf.offset?)?;
            const_sign(arena, b).filter(|&s| s != 0)
        }
    }
}

/// The eventual sign of `Re g` for an infinite `g` (`None`: unknown).
fn re_sign_of(arena: &mut Arena, inf: &Infinite) -> Option<i32> {
    match inf.axis {
        Axis::RealPos => Some(1),
        Axis::RealNeg => Some(-1),
        Axis::Other => {
            let (re, _) = const_parts(arena, inf.dir)?;
            const_sign(arena, re)
        }
        Axis::ImagPos | Axis::ImagNeg => {
            let (a, _) = const_parts(arena, inf.offset?)?;
            const_sign(arena, a).filter(|&s| s != 0)
        }
    }
}

/// `var` near the limit point as an expression in the positive dummy `t →
/// ∞`: `t`, `−t`, or `a ± 1/t`.
fn approach_substitute(arena: &mut Arena, ap: &Approach, t: ExprId, right: bool) -> ExprId {
    if ap.point == arena.infinity() {
        t
    } else if ap.point == arena.neg_infinity() {
        arena.neg(t)
    } else {
        let one = arena.one();
        let inv = arena.div(one, t);
        let shift = if right { inv } else { arena.neg(inv) };
        arena.add(&[ap.point, shift])
    }
}

/// The sides an approach comes from (`true`: from the right).
fn approach_sides(dir: Direction) -> &'static [bool] {
    match dir {
        Direction::Right => &[true],
        Direction::Left => &[false],
        Direction::Both => &[true, false],
    }
}

/// Is the real part of `g` identically zero near the limit point (`g =
/// i·h` with `h` real there)?
fn real_part_vanishes(arena: &mut Arena, g: ExprId, ap: &Approach) -> bool {
    let t = arena.positive_symbol("__lim_r");
    for &right in approach_sides(ap.dir) {
        let sub = approach_substitute(arena, ap, t, right);
        let e = crate::transforms::subs::subs(arena, g, ap.var, sub);
        let e = crate::transforms::eval::eval(arena, e);
        let p = crate::base::complex::decompose(arena, e);
        if !p.exact {
            return false;
        }
        let re = crate::transforms::eval::eval(arena, p.re);
        if !arena.is_zero_structural(re) {
            return false;
        }
    }
    true
}

// ═══════════════════════════════════════════════════════════════════════════
// Entry points
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the two-sided limit of `expr` as `var` approaches `point`, for
/// generic values of any other symbols (see [`limit_dir_generic`]).
pub(crate) fn limit(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Result<ExprId, SymplexError> {
    limit_dir_generic(arena, expr, var, point, Direction::Both)
}

/// Compute the limit of `expr` as `var` approaches `point` from `dir`.
///
/// Returns `Err` when the limit does not exist (e.g. the one-sided limits
/// differ, or the expression oscillates) or cannot be determined.  With
/// one free parameter, the values of it that make a coefficient vanish are
/// checked and get their own `Piecewise` case when the limit differs there
/// ([`guard_degenerate_parameters`]); the internal callers use
/// [`limit_dir_generic`].
pub(crate) fn limit_dir(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    dir: Direction,
) -> Result<ExprId, SymplexError> {
    let value = limit_dir_generic(arena, expr, var, point, dir)?;
    Ok(guard_degenerate_parameters(
        arena, expr, var, point, dir, value,
    ))
}

/// [`limit_dir`] for a generic value of every other symbol: a symbolic
/// coefficient is taken to be non-zero (`x²/(a + x²) → 0`), as the
/// definite integrator, the residue and transform code and the series
/// engine expect.
pub(crate) fn limit_dir_generic(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    dir: Direction,
) -> Result<ExprId, SymplexError> {
    if !matches!(arena.node(var), ExprNode::Symbol(_)) {
        return Err(SymplexError::InvalidArgument {
            operation: "limit",
            reason: "the limit variable must be a symbol".into(),
        });
    }
    limit_impl(arena, expr, var, point, dir, 0)
}

/// Most parameter values [`guard_degenerate_parameters`] re-examines.
const MAX_DEGENERATE_VALUES: usize = 4;

/// The limit `value` of `expr`, computed for a generic value of its one
/// free parameter `p` (every method treats a symbolic coefficient as
/// non-zero), checked at the values of `p` that make a coefficient vanish:
/// the zeros of the maximal `var`-free sub-expressions containing `p`, and
/// of the `var`-free part of every sum.  Where the limit there differs
/// from `value` evaluated there, the answer becomes a `Piecewise`
/// (`(Lᵥ, p = v)` for a limit found there, and `value` under `p ≠ v` for
/// one that does not exist or cannot be found).  Before, `x²/(a + x²)` at
/// `0` was `0` also for `a = 0` (the limit is `1`), and `(a·x + 1)/(a·x + 2)`
/// at `∞` was `1` also for `a = 0` (it is `1/2`).  A value that is
/// undefined at `p = v` (`1/a`) claims nothing there and is kept.
fn guard_degenerate_parameters(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    dir: Direction,
    value: ExprId,
) -> ExprId {
    let params: Vec<ExprId> = crate::base::walk::free_symbols(arena, expr)
        .into_iter()
        .filter(|&s| s != var)
        .filter(|&s| match arena.node(s) {
            ExprNode::Symbol(sid) => !arena.symbol_name(*sid).starts_with('_'),
            _ => false,
        })
        .collect();
    let [p] = params.as_slice() else {
        return value;
    };
    let p = *p;
    let p_sym = match arena.node(p) {
        ExprNode::Symbol(s) => *s,
        _ => return value,
    };
    // Coefficient-like sub-expressions: maximal var-free subtrees in p, and
    // the var-free part of each sum that depends on var.
    let mut candidates: Vec<ExprId> = Vec::new();
    for id in crate::base::walk::post_order_ids(arena, expr) {
        if !crate::base::walk::contains(arena, id, var) {
            continue;
        }
        let node = arena.node(id).clone();
        let mut free_part: Vec<ExprId> = Vec::new();
        node.for_each_child(|c| {
            if !crate::base::walk::contains(arena, c, var)
                && crate::base::walk::has_free_symbol(arena, c, p_sym)
            {
                candidates.push(c);
                if matches!(node, ExprNode::Add(_)) {
                    free_part.push(c);
                }
            }
        });
        if let ExprNode::Add(children) = &node {
            let constant: Vec<ExprId> = children
                .iter()
                .copied()
                .filter(|&c| !crate::base::walk::contains(arena, c, var))
                .collect();
            if free_part.len() < constant.len() && !free_part.is_empty() {
                let s = arena.add(&constant);
                candidates.push(s);
            }
        }
    }
    candidates.sort_unstable();
    candidates.dedup();
    let mut values: Vec<ExprId> = Vec::new();
    for c in candidates {
        for s in crate::transforms::solve::solve(arena, c, p) {
            let v = crate::transforms::eval::eval(arena, s.value);
            if crate::base::walk::free_symbols(arena, v).is_empty()
                && is_finite_limit_value(arena, v, var)
                && !values.contains(&v)
            {
                values.push(v);
            }
        }
        if values.len() >= MAX_DEGENERATE_VALUES {
            break;
        }
    }
    values.truncate(MAX_DEGENERATE_VALUES);
    let mut special: Vec<(ExprId, ExprId)> = Vec::new();
    let mut excluded: Vec<ExprId> = Vec::new();
    for v in values {
        // A value of p the assumptions exclude needs no case.
        let diff = arena.sub(p, v);
        let diff = crate::transforms::eval::eval(arena, diff);
        let mut cache = crate::base::assumptions::AssumptionCache::new();
        if cache.query(arena, diff, crate::base::assumptions::Props::NONZERO) == Some(true) {
            continue;
        }
        let claimed = crate::transforms::subs::subs(arena, value, p, v);
        let claimed = crate::transforms::eval::eval(arena, claimed);
        if !is_valid_limit_value(arena, claimed, var) {
            continue;
        }
        let at_v = crate::transforms::subs::subs(arena, expr, p, v);
        let at_v = crate::transforms::eval::eval(arena, at_v);
        match limit_impl(arena, at_v, var, point, dir, 1) {
            Ok(actual) if same_value(arena, actual, claimed) => {}
            Ok(actual) => special.push((actual, v)),
            Err(_) => excluded.push(v),
        }
    }
    if special.is_empty() && excluded.is_empty() {
        return value;
    }
    let mut pairs: Vec<(ExprId, ExprId)> = Vec::with_capacity(special.len() + 1);
    for (actual, v) in special {
        let cond = arena.eq_(p, v);
        pairs.push((actual, cond));
    }
    let generic = if excluded.is_empty() {
        arena.bool_true()
    } else {
        let conds: Vec<ExprId> = excluded.iter().map(|&v| arena.ne_(p, v)).collect();
        arena.and(&conds)
    };
    pairs.push((value, generic));
    arena.piecewise(&pairs)
}

fn limit_impl(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    dir: Direction,
    depth: usize,
) -> Result<ExprId, SymplexError> {
    limit_val(arena, expr, var, point, dir, depth).map(|v| v.value)
}

/// [`limit_impl`] with what is known about an infinite value's
/// perpendicular component ([`LimVal`]).
fn limit_val(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    dir: Direction,
    depth: usize,
) -> Result<LimVal, SymplexError> {
    // ── Constant expression ─────────────────────────────────────────
    if !crate::base::walk::contains(arena, expr, var) {
        let v = crate::transforms::eval::eval(arena, expr);
        return if is_valid_limit_value(arena, v, var) {
            Ok(LimVal::exact(v))
        } else {
            Err(fail(
                "expression is a constant that is not a valid limit value",
            ))
        };
    }

    if point == arena.infinity() || point == arena.neg_infinity() {
        let positive = point == arena.infinity();
        return limit_at_infinity_dir(arena, expr, var, positive, depth);
    }

    if point == arena.nan() || point == arena.complex_infinity() {
        return Err(fail("limit point must be finite or ±∞"));
    }

    // ── Finite point ────────────────────────────────────────────────
    tracing::debug!(dir = ?dir, "limit: finite point");

    // (a) Safe direct substitution.
    if let Some(v) = try_direct_substitution(arena, expr, var, point) {
        tracing::debug!("limit: safe direct substitution succeeded");
        return Ok(LimVal::exact(v));
    }

    // (b) Compositional rule.
    if depth < MAX_COMPOSE_DEPTH
        && let Some(r) = try_compose(arena, expr, var, point, dir, depth)
    {
        return r;
    }

    // (c) One-sided Gruntz.
    let directional = has_directional_nodes(arena, expr, var);
    match dir {
        Direction::Right | Direction::Left => {
            let right = dir == Direction::Right;
            match one_sided_gruntz(arena, expr, var, point, right, depth) {
                Ok(v) => Ok(v),
                Err(e) => {
                    if directional {
                        return Err(e);
                    }
                    // One-sided fallback: L'Hôpital / series are two-sided
                    // tools, but for expressions built only from analytic
                    // functions a two-sided limit (when it exists) equals
                    // each one-sided limit.
                    analytic_fallback(arena, expr, var, point)
                        .map(LimVal::exact)
                        .map_err(|_| e)
                }
            }
        }
        Direction::Both => {
            let r = one_sided_gruntz(arena, expr, var, point, true, depth);
            let l = one_sided_gruntz(arena, expr, var, point, false, depth);
            match (r, l) {
                (Ok(r), Ok(l)) => {
                    if same_value(arena, r.value, l.value) {
                        Ok(LimVal {
                            value: r.value,
                            weak: r.weak || l.weak,
                        })
                    } else {
                        let rs = arena.display(r.value).to_string();
                        let ls = arena.display(l.value).to_string();
                        Err(fail(format!(
                            "left and right limits differ: left = {ls}, right = {rs}"
                        )))
                    }
                }
                (Ok(_), Err(e)) | (Err(e), Ok(_)) | (Err(e), Err(_)) => {
                    if directional {
                        return Err(e);
                    }
                    analytic_fallback(arena, expr, var, point)
                        .map(LimVal::exact)
                        .map_err(|_| e)
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Result validation
// ═══════════════════════════════════════════════════════════════════════════

/// A limit value is acceptable when it is a genuine constant: free of the
/// limit variable and of internal dummies, contains no unevaluated nodes,
/// and is neither `NaN` nor complex infinity. `±∞` on their own are fine,
/// and so are infinities with a direction or a finite part (`oo + I·π`,
/// `e^{2i}·oo`, `1 + I·oo`; see [`LimVal`]), but `∞` buried inside another
/// expression (`exp(oo)`, `atan(oo)`) is not.
pub(crate) fn is_valid_limit_value(arena: &Arena, id: ExprId, var: ExprId) -> bool {
    if id == arena.nan() || id == arena.complex_infinity() {
        return false;
    }
    if id == arena.infinity() || id == arena.neg_infinity() {
        return true;
    }
    if crate::base::walk::contains(arena, id, var) || contains_dummy(arena, id) {
        return false;
    }
    if crate::base::walk::has_unevaluated(arena, id) {
        return false;
    }
    (!contains_singular_atom(arena, id) || is_infinite_form(arena, id))
        && !contains_hidden_singularity(arena, id)
}

/// A valid limit value that is finite (no `±∞`, and no infinity with a
/// direction or finite part).
pub(crate) fn is_finite_limit_value(arena: &Arena, id: ExprId, var: ExprId) -> bool {
    is_valid_limit_value(arena, id, var) && !contains_singular_atom(arena, id)
}

/// Detect symbolic forms that `eval` leaves untouched but that denote
/// poles or undefined values: `ln 0`, `Γ(−n)`, `ψ(−n)`, `(−n)!`, `0^{−k}`,
/// `tan(π/2 + kπ)` is caught separately as `zoo`.
fn contains_hidden_singularity(arena: &Arena, root: ExprId) -> bool {
    let mut stack = vec![root];
    let mut visited = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        let node = arena.node(id);
        match node {
            ExprNode::Ln(a) | ExprNode::LogGamma(a) if arena.is_zero_structural(*a) => {
                return true;
            }
            ExprNode::Gamma(a) | ExprNode::Digamma(a) if is_nonpositive_integer(arena, *a) => {
                return true;
            }
            ExprNode::Factorial(a) => {
                if let Some(r) = arena.as_num(*a)
                    && r.is_integer()
                    && r.is_negative()
                {
                    return true;
                }
            }
            ExprNode::Pow(b, e) if arena.is_zero_structural(*b) => match arena.as_num(*e) {
                Some(r) if r.is_positive() => {}
                _ => return true,
            },
            _ => {}
        }
        node.for_each_child(|c| stack.push(c));
    }
    false
}

/// `true` if the expression contains `∞`, `−∞`, `zoo`, or `NaN` anywhere.
fn contains_singular_atom(arena: &Arena, id: ExprId) -> bool {
    crate::base::walk::contains(arena, id, arena.infinity())
        || crate::base::walk::contains(arena, id, arena.neg_infinity())
        || crate::base::walk::contains(arena, id, arena.complex_infinity())
        || crate::base::walk::contains(arena, id, arena.nan())
}

/// `true` if `id` is `±∞`, `zoo`, or `NaN`, or contains one of them.
fn is_singular(arena: &Arena, id: ExprId) -> bool {
    id == arena.infinity()
        || id == arena.neg_infinity()
        || id == arena.complex_infinity()
        || id == arena.nan()
        || contains_singular_atom(arena, id)
}

/// `true` if the expression contains an internal dummy symbol (`__gw…`,
/// `__lim…`, `__limit_t`).
pub(crate) fn contains_dummy(arena: &Arena, root: ExprId) -> bool {
    let mut stack = vec![root];
    let mut visited = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        if let ExprNode::Symbol(sid) = arena.node(id) {
            let name = arena.symbol_name(*sid);
            if name.starts_with("__gw") || name.starts_with("__lim") {
                return true;
            }
        }
        arena.node(id).for_each_child(|c| stack.push(c));
    }
    false
}

/// Structural check whether two computed limit values agree.
fn same_value(arena: &mut Arena, a: ExprId, b: ExprId) -> bool {
    if a == b {
        return true;
    }
    let a_inf = a == arena.infinity() || a == arena.neg_infinity();
    let b_inf = b == arena.infinity() || b == arena.neg_infinity();
    if a_inf || b_inf {
        return false;
    }
    if let (Some(ra), Some(rb)) = (arena.as_num(a), arena.as_num(b)) {
        return ra == rb;
    }
    let diff = arena.sub(a, b);
    let diff = crate::transforms::eval::eval(arena, diff);
    if arena.is_zero_structural(diff) || is_hidden_zero(arena, diff) {
        return true;
    }
    let simplified = crate::transforms::expand::expand(arena, diff);
    let simplified = crate::transforms::eval::eval(arena, simplified);
    arena.is_zero_structural(simplified) || is_hidden_zero(arena, simplified)
}

// ═══════════════════════════════════════════════════════════════════════════
// Directional (discontinuous) node detection
// ═══════════════════════════════════════════════════════════════════════════

/// `true` if `expr` contains a node whose value or derivative is
/// discontinuous and whose argument depends on `var` (`Abs`, `Sign`,
/// `Heaviside`, `DiracDelta`, `Floor`, `Ceiling`, `Piecewise`, `Min`, `Max`).
///
/// For such expressions the two-sided L'Hôpital / series fallbacks are not
/// trustworthy, so only the one-sided Gruntz path is used.
pub(crate) fn has_directional_nodes(arena: &Arena, expr: ExprId, var: ExprId) -> bool {
    let mut stack = vec![expr];
    let mut visited = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        let node = arena.node(id);
        let directional = matches!(
            node,
            ExprNode::Abs(_)
                | ExprNode::Sign(_)
                | ExprNode::Heaviside(_)
                | ExprNode::DiracDelta(_)
                | ExprNode::Floor(_)
                | ExprNode::Ceiling(_)
                | ExprNode::Piecewise(_)
                | ExprNode::Min(_)
                | ExprNode::Max(_)
        );
        if directional && crate::base::walk::contains(arena, id, var) {
            return true;
        }
        node.for_each_child(|c| stack.push(c));
    }
    false
}

// ═══════════════════════════════════════════════════════════════════════════
// (a) Safe direct substitution
// ═══════════════════════════════════════════════════════════════════════════

/// Substitute `var = point` and evaluate, but only accept the result when the
/// substitution is provably continuous: every `var`-dependent sub-expression
/// is evaluated at the point and must be finite, and discontinuous functions
/// must not be evaluated exactly at their discontinuity.
///
/// This guards against canonicalization folding indeterminate forms
/// (e.g. `0 · 0⁻¹ → 0`) into plausible-looking finite numbers.
fn try_direct_substitution(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    let v = safe_substitute(arena, expr, var, point)?;
    is_finite_limit_value(arena, v, var).then_some(v)
}

/// Bottom-up substitution `var = point` that returns `None` as soon as any
/// `var`-dependent sub-expression is singular at the point (`∞`, `zoo`,
/// `NaN`, `ln 0`, `0^0`, a pole of `Γ`, a discontinuous node exactly at its
/// discontinuity, …).
///
/// Unlike a plain `subs` + `eval`, this never lets canonicalization fold an
/// indeterminate form (`0 · Γ(zoo) → 0`) into a plausible number. The value
/// returned may still contain other symbols (including internal dummies);
/// see [`try_direct_substitution`] for the fully validated variant.
pub(crate) fn safe_substitute(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    let post = crate::base::walk::post_order_ids(arena, expr);
    let mut values: FxHashMap<ExprId, ExprId> = FxHashMap::default();

    for &id in &post {
        if !crate::base::walk::contains(arena, id, var) {
            continue;
        }
        let node = arena.node(id).clone();

        // Value of a child at the point (constants are their own value).
        let val_of = |arena: &mut Arena, values: &FxHashMap<ExprId, ExprId>, c: ExprId| {
            if let Some(&v) = values.get(&c) {
                v
            } else {
                crate::transforms::eval::eval(arena, c)
            }
        };

        // A special function with a branch cut (`Ei`, `W`, `polylog`, `Kν`,
        // …) at a point of its cut is continuous there only along the cut;
        // `Ei(−2 + ix) → Ei(−2) + iπ` from `x > 0`.  (`ln`, powers and
        // the inverse functions are checked below.)
        if !matches!(
            node,
            ExprNode::Ln(_)
                | ExprNode::LogGamma(_)
                | ExprNode::Pow(..)
                | ExprNode::Asin(_)
                | ExprNode::Acos(_)
                | ExprNode::Acosh(_)
                | ExprNode::Atanh(_)
                | ExprNode::Asinh(_)
                | ExprNode::Atan(_)
        ) && let Some((a, cut)) = branch_cut_of(arena, id)
            && crate::base::walk::contains(arena, a, var)
        {
            let av = val_of(arena, &values, a);
            if on_branch_cut(arena, cut, av) && !inner_known_real(arena, a) {
                return None;
            }
        }

        // Node-specific continuity checks on the *argument* values.
        match node {
            ExprNode::Piecewise(_) => return None,
            ExprNode::Ln(a) | ExprNode::LogGamma(a) => {
                let av = val_of(arena, &values, a);
                if is_zero_value(arena, av) || !is_definite_constant(arena, av) {
                    return None;
                }
                if on_branch_cut(arena, BranchCut::NegativeReals, av) && !inner_known_real(arena, a)
                {
                    return None;
                }
            }
            ExprNode::Asin(a) | ExprNode::Acos(a) | ExprNode::Acosh(a) | ExprNode::Atanh(a) => {
                let av = val_of(arena, &values, a);
                let cut = if matches!(node, ExprNode::Acosh(_)) {
                    BranchCut::BelowOne
                } else {
                    BranchCut::BeyondOne
                };
                if on_branch_cut(arena, cut, av) && !inner_known_real(arena, a) {
                    return None;
                }
                if let ExprNode::Atanh(_) = node {
                    let one = arena.one();
                    let minus = arena.sub(av, one);
                    let plus = arena.add(&[av, one]);
                    let (minus, plus) = (
                        crate::transforms::eval::eval(arena, minus),
                        crate::transforms::eval::eval(arena, plus),
                    );
                    if is_zero_value(arena, minus) || is_zero_value(arena, plus) {
                        return None;
                    }
                }
            }
            ExprNode::Asinh(a) | ExprNode::Atan(a) => {
                let av = val_of(arena, &values, a);
                if on_branch_cut(arena, BranchCut::ImaginaryAxis, av) {
                    return None;
                }
            }
            ExprNode::Gamma(a) | ExprNode::Digamma(a) => {
                let av = val_of(arena, &values, a);
                if is_nonpositive_integer(arena, av) {
                    return None;
                }
            }
            ExprNode::Factorial(a) => {
                let av = val_of(arena, &values, a);
                if let Some(r) = arena.as_num(av)
                    && r.is_integer()
                    && r.is_negative()
                {
                    return None;
                }
            }
            ExprNode::Tan(a) => {
                let av = val_of(arena, &values, a);
                let c = arena.cos(av);
                let c = crate::transforms::eval::eval(arena, c);
                if is_zero_value(arena, c) {
                    return None;
                }
            }
            ExprNode::Pow(b, e) => {
                let bv = val_of(arena, &values, b);
                let ev = val_of(arena, &values, e);
                if is_zero_value(arena, bv) {
                    // 0^e is continuous only for a definite positive exponent.
                    match arena.as_num(ev) {
                        Some(r) if r.is_positive() => {}
                        _ => return None,
                    }
                }
                // A fractional power at a point of its cut: continuous only
                // along the cut (a real base); `√(−1 − i·x)` at `x → 0⁺`
                // tends to `−i`, not to `√(−1) = i`.
                if arena.as_num(e).is_none_or(|r| !r.is_integer())
                    && on_branch_cut(arena, BranchCut::NegativeReals, bv)
                    && !inner_known_real(arena, b)
                {
                    return None;
                }
                // 1^∞ / ∞^0 forms are caught by the singular check below
                // because the offending sub-expression evaluates to ∞.
            }
            ExprNode::Sign(a) | ExprNode::Heaviside(a) | ExprNode::DiracDelta(a) => {
                let av = val_of(arena, &values, a);
                match arena.as_num(av) {
                    Some(r) if !r.is_zero() => {}
                    _ => return None,
                }
            }
            ExprNode::Floor(a) | ExprNode::Ceiling(a) => {
                let av = val_of(arena, &values, a);
                match arena.as_num(av) {
                    Some(r) if !r.is_integer() => {}
                    _ => return None,
                }
            }
            _ => {}
        }

        let subst = crate::transforms::subs::subs(arena, id, var, point);
        let v = crate::transforms::eval::eval(arena, subst);
        if is_singular(arena, v) {
            return None;
        }
        values.insert(id, v);
    }

    values.get(&expr).copied()
}

/// Largest tree the one-sided Gruntz preprocessing expands.
const MAX_EXPAND_SIZE: usize = 400;

/// Largest [`expansion_cost_estimate`] the one-sided Gruntz preprocessing
/// expands.
const MAX_EXPANSION_COST: f64 = 4_000.0;

/// The number of terms `expand` would produce, summed over the nodes of
/// `e`: a sum has the terms of its children, a product their product, a
/// positive integer power `bⁿ` the `n`-th power of `b`'s (an upper bound
/// on the multinomial count), anything else one.  A small tree can expand
/// to an enormous one (`(1/w − 1/2)³⁰`).
fn expansion_cost_estimate(arena: &Arena, e: ExprId) -> f64 {
    let post = crate::base::walk::post_order_ids(arena, e);
    let mut terms: FxHashMap<ExprId, f64> = FxHashMap::default();
    let mut cost = 0.0;
    for &id in &post {
        let t = |c: ExprId| terms.get(&c).copied().unwrap_or(1.0);
        let v = match arena.node(id) {
            ExprNode::Add(cs) => cs.iter().map(|&c| t(c)).sum(),
            ExprNode::Mul(cs) => cs.iter().map(|&c| t(c)).product(),
            ExprNode::Pow(b, n) => match arena.as_num(*n) {
                Some(r) if r.is_integer() && r.is_positive() => {
                    let k = num_traits::ToPrimitive::to_f64(r).unwrap_or(f64::INFINITY);
                    t(*b).powf(k)
                }
                _ => 1.0,
            },
            _ => 1.0,
        };
        let v: f64 = v.min(1e12);
        cost += v;
        terms.insert(id, v);
    }
    cost
}

/// Is the value `v` zero, structurally or as a sum of constants that
/// cancels numerically (`asinh 2 − ln(2 + √5)`)?  A structural test let
/// `x/(asinh(x + 2) − ln(2 + √5))` substitute to `0·(1/0) = 0` at `x = 0`.
fn is_zero_value(arena: &Arena, v: ExprId) -> bool {
    arena.is_zero_structural(v) || is_hidden_zero(arena, v)
}

/// A "definite" constant: numeric, or a symbolic constant with a known sign.
fn is_definite_constant(arena: &mut Arena, id: ExprId) -> bool {
    arena.as_num(id).is_some() || const_sign(arena, id).is_some()
}

fn is_nonpositive_integer(arena: &Arena, id: ExprId) -> bool {
    match arena.as_num(id) {
        Some(r) => r.is_integer() && !r.is_positive(),
        None => false,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Sign of constants (with assumptions)
// ═══════════════════════════════════════════════════════════════════════════

/// Sign of a constant expression: `Some(1)`, `Some(-1)`, `Some(0)`, or
/// `None` if unknown. Consults symbol assumptions (`a > 0`).
pub(crate) fn const_sign(arena: &mut Arena, e: ExprId) -> Option<i32> {
    const_sign_depth(arena, e, 0)
}

fn const_sign_depth(arena: &mut Arena, e: ExprId, depth: usize) -> Option<i32> {
    if depth > 16 {
        return None;
    }
    if let Some(r) = arena.as_num(e) {
        return Some(sign_of_ratio(r));
    }
    if e == arena.pi() || e == arena.e_const() || e == arena.infinity() {
        return Some(1);
    }
    if e == arena.neg_infinity() {
        return Some(-1);
    }
    if e == arena.nan() || e == arena.complex_infinity() || e == arena.i_unit() {
        return None;
    }
    let evaled = crate::transforms::eval::eval(arena, e);
    if evaled != e
        && let Some(r) = arena.as_num(evaled)
    {
        return Some(sign_of_ratio(r));
    }
    match arena.node(e).clone() {
        ExprNode::Neg(inner) => const_sign_depth(arena, inner, depth + 1).map(|s| -s),
        ExprNode::Mul(children) => {
            let mut result = 1;
            for c in children {
                let s = const_sign_depth(arena, c, depth + 1)?;
                if s == 0 {
                    return Some(0);
                }
                result *= s;
            }
            Some(result)
        }
        ExprNode::Add(children) => {
            // All terms share a sign → that sign.
            let mut sign = 0;
            for c in children {
                let s = const_sign_depth(arena, c, depth + 1)?;
                if s == 0 {
                    continue;
                }
                if sign == 0 {
                    sign = s;
                } else if sign != s {
                    return assumption_sign(arena, e).or_else(|| numeric_sign(arena, e));
                }
            }
            Some(sign)
        }
        // Positive for a real argument only: `cosh(acos 4) = cos(acosh 4)`
        // is −0.47, and `exp` of a non-real number is not positive either.
        // (Taking both for positive turned `lim_{x→−∞} (asinh x − 2x)·cosh(acos 4)`
        // into `+∞`.)  A constant with a known sign is real.
        ExprNode::Exp(a) | ExprNode::Cosh(a) => {
            if const_sign_depth(arena, a, depth + 1).is_some() {
                Some(1)
            } else {
                assumption_sign(arena, e).or_else(|| numeric_sign(arena, e))
            }
        }
        ExprNode::Abs(inner) => {
            let s = const_sign_depth(arena, inner, depth + 1)?;
            Some(if s == 0 { 0 } else { 1 })
        }
        ExprNode::Pow(base, exp) => {
            let bs = const_sign_depth(arena, base, depth + 1)?;
            if let Some(r) = arena.as_num(exp).cloned() {
                if r.is_integer() {
                    let odd = (r.to_integer() % BigInt::from(2)) != BigInt::zero();
                    if bs == 0 {
                        return if r.is_positive() { Some(0) } else { None };
                    }
                    return Some(if odd { bs } else { 1 });
                }
                if bs > 0 {
                    return Some(1);
                }
                if bs == 0 && r.is_positive() {
                    return Some(0);
                }
                return None;
            }
            // b^e > 0 for b > 0 needs a real exponent (`2^i` is not real).
            if bs > 0 && const_sign_depth(arena, exp, depth + 1).is_some() {
                Some(1)
            } else {
                assumption_sign(arena, e).or_else(|| numeric_sign(arena, e))
            }
        }
        ExprNode::Ln(inner) => {
            let r = arena.as_num(inner).cloned()?;
            if !r.is_positive() {
                return None;
            }
            let one = Ratio::from_integer(BigInt::from(1));
            Some(if r > one {
                1
            } else if r == one {
                0
            } else {
                -1
            })
        }
        _ => assumption_sign(arena, e).or_else(|| numeric_sign(arena, e)),
    }
}

/// Sign of a constant free of symbols by certified evaluation: the value
/// must be real (see [`constant_realness`]), and then its digits decide;
/// an exact 0 from `evalf` (a sum cancelling to its working precision) is
/// 0.  `None` for a non-real or unevaluable constant.
///
/// [`constant_realness`]: crate::transforms::realness::constant_realness
fn numeric_sign(arena: &mut Arena, e: ExprId) -> Option<i32> {
    if !crate::base::walk::free_symbols(arena, e).is_empty()
        || crate::base::walk::has_unevaluated(arena, e)
    {
        return None;
    }
    let z = crate::transforms::evalf::evalf_complex(arena, e, NUMERIC_SIGN_DIGITS).ok()?;
    if z.0.is_zero() && z.1.is_zero() {
        return Some(0);
    }
    let mut reals = crate::base::assumptions::AssumptionCache::new();
    let real =
        crate::transforms::realness::constant_realness(arena, e, NUMERIC_SIGN_DIGITS, &mut reals);
    if real != Some(true) {
        return None;
    }
    if z.0.is_positive() {
        Some(1)
    } else if z.0.is_negative() {
        Some(-1)
    } else {
        None
    }
}

/// Digits at which [`numeric_sign`] evaluates a constant.
const NUMERIC_SIGN_DIGITS: u32 = 20;

fn sign_of_ratio(r: &Q) -> i32 {
    if r.is_positive() {
        1
    } else if r.is_negative() {
        -1
    } else {
        0
    }
}

/// Sign of a symbolic constant via the assumption system.
fn assumption_sign(arena: &Arena, e: ExprId) -> Option<i32> {
    use crate::base::assumptions::{AssumptionCache, Props};
    let mut cache = AssumptionCache::new();
    if cache.query(arena, e, Props::POSITIVE) == Some(true) {
        return Some(1);
    }
    if cache.query(arena, e, Props::NEGATIVE) == Some(true) {
        return Some(-1);
    }
    if cache.query(arena, e, Props::ZERO) == Some(true) {
        return Some(0);
    }
    None
}

// ═══════════════════════════════════════════════════════════════════════════
// (b) Compositional rule
// ═══════════════════════════════════════════════════════════════════════════

/// Wrap a finite candidate: evaluate and validate.
fn finite_candidate(arena: &mut Arena, v: ExprId, var: ExprId) -> Option<ExprId> {
    let v = crate::transforms::eval::eval(arena, v);
    is_finite_limit_value(arena, v, var).then_some(v)
}

/// A rule's result from a finite candidate ([`finite_candidate`]).
fn finite_result(
    arena: &mut Arena,
    v: ExprId,
    var: ExprId,
) -> Option<Result<LimVal, SymplexError>> {
    finite_candidate(arena, v, var).map(|v| Ok(LimVal::exact(v)))
}

/// A rule's result that is exactly `v`.
fn exact_result(v: ExprId) -> Option<Result<LimVal, SymplexError>> {
    Some(Ok(LimVal::exact(v)))
}

/// The rule proves that the limit does not exist.
fn no_limit() -> Option<Result<LimVal, SymplexError>> {
    Some(Err(fail(
        "the function oscillates or is undefined as its argument diverges",
    )))
}

/// `±oo + i·im` as a rule's result.
fn real_infinity_plus(
    arena: &mut Arena,
    positive: bool,
    im: ExprId,
    var: ExprId,
) -> Option<Result<LimVal, SymplexError>> {
    let (dir, axis) = if positive {
        (arena.one(), Axis::RealPos)
    } else {
        (arena.neg_one(), Axis::RealNeg)
    };
    let i = arena.i_unit();
    let offset = arena.mul(&[i, im]);
    let offset = crate::transforms::eval::eval(arena, offset);
    infinite_value(
        arena,
        Infinite {
            dir,
            axis,
            offset: Some(offset),
        },
        var,
    )
    .map(Ok)
}

/// `±I·oo + re` as a rule's result.
fn imaginary_infinity_plus(
    arena: &mut Arena,
    positive: bool,
    re: ExprId,
    var: ExprId,
) -> Option<Result<LimVal, SymplexError>> {
    let i = arena.i_unit();
    let (dir, axis) = if positive {
        (i, Axis::ImagPos)
    } else {
        (arena.neg(i), Axis::ImagNeg)
    };
    let re = crate::transforms::eval::eval(arena, re);
    infinite_value(
        arena,
        Infinite {
            dir,
            axis,
            offset: Some(re),
        },
        var,
    )
    .map(Ok)
}

/// `c·g` for a constant `c` and `g` tending to the infinity `inf`: the
/// direction turns by `arg c`, the perpendicular component scales with it
/// (and stays unknown when an off-axis direction turns onto an axis).
fn scale_inf(arena: &mut Arena, inf: Infinite, c: ExprId) -> Option<Infinite> {
    let d = arena.mul(&[c, inf.dir]);
    let d = crate::transforms::eval::eval(arena, d);
    let axis = direction_axis(arena, d)?;
    let offset = match (inf.axis, axis) {
        (_, Axis::Other) => None,
        (Axis::Other, _) => return None,
        _ => inf.offset.map(|o| {
            let p = arena.mul(&[c, o]);
            crate::transforms::eval::eval(arena, p)
        }),
    };
    Some(Infinite {
        dir: d,
        axis,
        offset,
    })
}

/// `e^g` for `g` tending to the infinity `inf`.
fn exp_of_infinite(
    arena: &mut Arena,
    inf: Infinite,
    g: ExprId,
    ap: &Approach,
) -> Option<Result<LimVal, SymplexError>> {
    match inf.axis {
        Axis::RealNeg => exact_result(arena.zero()),
        Axis::RealPos => {
            let Some(o) = inf.offset else {
                // |e^g| → ∞ while its argument `Im g` diverges.
                return no_limit();
            };
            if eventually_real(arena, g, ap) {
                return exact_result(arena.infinity());
            }
            let b = offset_im(arena, o)?;
            if const_sign(arena, b)? == 0 {
                // Direction 1, but the imaginary part `e^{Re g}·sin(Im g)`
                // is unknown: left to the Gruntz algorithm.
                return None;
            }
            let i = arena.i_unit();
            let ib = arena.mul(&[i, b]);
            let d = arena.exp(ib);
            let d = crate::transforms::eval::eval(arena, d);
            if direction_axis(arena, d)? != Axis::Other {
                return None;
            }
            infinite_value(
                arena,
                Infinite {
                    dir: d,
                    axis: Axis::Other,
                    offset: None,
                },
                ap.var,
            )
            .map(Ok)
        }
        // |e^g| → e^{Re g} while the argument `Im g` diverges.
        Axis::ImagPos | Axis::ImagNeg => inf.offset.and_then(|_| no_limit()),
        Axis::Other => match re_sign_of(arena, &inf)? {
            s if s > 0 => no_limit(),
            s if s < 0 => exact_result(arena.zero()),
            _ => None,
        },
    }
}

/// `ln g` for `g` tending to the infinity `inf`: `ln|g| + i·arg g` with
/// `arg g → arg d`, and `±π` along the negative axis by the side of the cut
/// `g` approaches from (`ln(−x) → oo + iπ`, `ln(2ix) → oo + iπ/2`).
fn ln_of_infinite(
    arena: &mut Arena,
    inf: Infinite,
    g: ExprId,
    ap: &Approach,
) -> Option<Result<LimVal, SymplexError>> {
    let pi = arena.pi();
    let half = arena.rational(1, 2);
    let im = match inf.axis {
        Axis::RealPos => arena.zero(),
        Axis::ImagPos => arena.mul(&[half, pi]),
        Axis::ImagNeg => {
            let h = arena.mul(&[half, pi]);
            arena.neg(h)
        }
        Axis::Other => const_arg(arena, inf.dir)?,
        Axis::RealNeg => {
            if eventually_real(arena, g, ap) || im_sign_of(arena, &inf)? > 0 {
                pi
            } else {
                arena.neg(pi)
            }
        }
    };
    real_infinity_plus(arena, true, im, ap.var)
}

/// `g^r` (constant rational `r`) for `g` tending to the infinity `inf`.
fn pow_of_infinite(
    arena: &mut Arena,
    inf: Infinite,
    base: ExprId,
    exp: ExprId,
    r: &Q,
    ap: &Approach,
) -> Option<Result<LimVal, SymplexError>> {
    if r.is_zero() {
        return exact_result(arena.one());
    }
    if r.is_negative() {
        return exact_result(arena.zero());
    }
    let one = Q::from_integer(BigInt::from(1));
    let weak_oo = |arena: &Arena| {
        Some(Ok(LimVal {
            value: arena.infinity(),
            weak: true,
        }))
    };
    match inf.axis {
        Axis::RealPos => {
            if eventually_real(arena, base, ap) {
                return exact_result(arena.infinity());
            }
            // `Im(g^r) ≈ r·Im g·(Re g)^{r−1}`: it tends to 0 for `r < 1`
            // and a bounded `Im g`, and diverges for `r > 1` when `Im g`
            // tends to `b ≠ 0` or diverges.
            match inf.offset {
                Some(_) if *r < one => exact_result(arena.infinity()),
                Some(o) => {
                    let b = offset_im(arena, o)?;
                    if const_sign(arena, b)? != 0 {
                        weak_oo(arena)
                    } else {
                        None
                    }
                }
                None if *r > one => weak_oo(arena),
                None => None,
            }
        }
        Axis::RealNeg => {
            if !eventually_real(arena, base, ap) {
                // The side of the cut decides: left to the Gruntz algorithm.
                return None;
            }
            // `g^r = |g|^r·e^{iπr}` exactly.
            let d = if r.is_integer() {
                let odd = (r.to_integer() % BigInt::from(2)) != BigInt::zero();
                if odd { arena.neg_one() } else { arena.one() }
            } else {
                let i = arena.i_unit();
                let pi = arena.pi();
                let ph = arena.mul(&[i, pi, exp]);
                let d = arena.exp(ph);
                crate::transforms::eval::eval(arena, d)
            };
            let axis = direction_axis(arena, d)?;
            let offset = (axis != Axis::Other).then(|| arena.zero());
            infinite_value(
                arena,
                Infinite {
                    dir: d,
                    axis,
                    offset,
                },
                ap.var,
            )
            .map(Ok)
        }
        _ => {
            // `arg g → arg d`, away from the cut: the direction is `d^r`.
            let d = arena.pow(inf.dir, exp);
            let d = crate::transforms::eval::eval(arena, d);
            if direction_axis(arena, d)? != Axis::Other {
                return None;
            }
            infinite_value(
                arena,
                Infinite {
                    dir: d,
                    axis: Axis::Other,
                    offset: None,
                },
                ap.var,
            )
            .map(Ok)
        }
    }
}

/// `Γ(g)` (or `g!`) for `g` tending to the infinity `inf`.  Off the
/// positive axis `|Γ(z)| → 0` (Stirling: `Re(z ln z − z) → −∞`); along it
/// a non-real `g` turns: `arg Γ(x + ib) ≈ b·ln x`, no limit.
fn gamma_of_infinite(
    arena: &mut Arena,
    inf: Infinite,
    g: ExprId,
    ap: &Approach,
    factorial: bool,
) -> Option<Result<LimVal, SymplexError>> {
    let real = eventually_real(arena, g, ap);
    match inf.axis {
        Axis::RealPos => {
            if real {
                return exact_result(arena.infinity());
            }
            let b = offset_im(arena, inf.offset?)?;
            if const_sign(arena, b)? != 0 {
                return no_limit();
            }
            None
        }
        Axis::RealNeg => {
            if real {
                return if factorial {
                    None
                } else {
                    Some(Err(fail("Γ(x) has no limit as x → −∞")))
                };
            }
            let b = offset_im(arena, inf.offset?)?;
            if const_sign(arena, b)? != 0 {
                return exact_result(arena.zero());
            }
            None
        }
        Axis::ImagPos | Axis::ImagNeg => inf.offset.and_then(|_| exact_result(arena.zero())),
        Axis::Other => match re_sign_of(arena, &inf)? {
            s if s < 0 => exact_result(arena.zero()),
            s if s > 0 => no_limit(),
            _ => None,
        },
    }
}

/// `F(g)` for a unary function `F` (the node `node`) and `g` tending to
/// the infinity `inf`: the asymptotic values of `F` in each direction
/// (checked against mpmath at `10²⁰` in the directions `±1`, `±i`, `±1 ± i`).
/// `None` when the rule does not decide.
fn unary_of_infinite(
    arena: &mut Arena,
    node: &ExprNode,
    g: ExprId,
    inf: Infinite,
    ap: &Approach,
) -> Option<Result<LimVal, SymplexError>> {
    let var = ap.var;
    let pi = arena.pi();
    let half = arena.rational(1, 2);
    let half_pi = arena.mul(&[half, pi]);
    let neg_half_pi = arena.neg(half_pi);
    let positive = matches!(inf.axis, Axis::RealPos);
    let negative = matches!(inf.axis, Axis::RealNeg);
    match node {
        // `atan z → ±π/2` with the sign of `Re z` (and on its cut, the
        // imaginary axis, `+π/2` above, `−π/2` below).
        ExprNode::Atan(_) => {
            let s = match re_sign_of(arena, &inf) {
                Some(s) if s != 0 => s,
                _ if matches!(inf.axis, Axis::ImagPos | Axis::ImagNeg)
                    && real_part_vanishes(arena, g, ap) =>
                {
                    if inf.axis == Axis::ImagPos {
                        1
                    } else {
                        -1
                    }
                }
                _ => return None,
            };
            exact_result(if s > 0 { half_pi } else { neg_half_pi })
        }
        // `tanh z → ±1` as `Re z → ±∞`.
        ExprNode::Tanh(_) => match re_sign_of(arena, &inf)? {
            s if s > 0 => exact_result(arena.one()),
            s if s < 0 => exact_result(arena.neg_one()),
            _ => None,
        },
        ExprNode::Erf(_) if positive => exact_result(arena.one()),
        ExprNode::Erf(_) if negative => exact_result(arena.neg_one()),
        ExprNode::Erfc(_) if positive => exact_result(arena.zero()),
        ExprNode::Erfc(_) if negative => exact_result(arena.int(2)),
        ExprNode::Abs(_) => exact_result(arena.infinity()),
        // `sinh g ≈ ±e^{±g}/2`, `cosh g ≈ e^{±g}/2` as `Re g → ±∞`.
        ExprNode::Sinh(_) | ExprNode::Cosh(_) => {
            let is_sinh = matches!(node, ExprNode::Sinh(_));
            match inf.axis {
                Axis::RealPos | Axis::RealNeg => {
                    let Some(o) = inf.offset else {
                        return no_limit();
                    };
                    if eventually_real(arena, g, ap) {
                        return exact_result(if positive || !is_sinh {
                            arena.infinity()
                        } else {
                            arena.neg_infinity()
                        });
                    }
                    let b = offset_im(arena, o)?;
                    if const_sign(arena, b)? == 0 {
                        return None;
                    }
                    let i = arena.i_unit();
                    let ib = arena.mul(&[i, b]);
                    let ib = if positive { ib } else { arena.neg(ib) };
                    let d = arena.exp(ib);
                    let d = if is_sinh && negative { arena.neg(d) } else { d };
                    let d = crate::transforms::eval::eval(arena, d);
                    if direction_axis(arena, d)? != Axis::Other {
                        return None;
                    }
                    infinite_value(
                        arena,
                        Infinite {
                            dir: d,
                            axis: Axis::Other,
                            offset: None,
                        },
                        var,
                    )
                    .map(Ok)
                }
                // A bounded oscillation (`Re g → a`).
                Axis::ImagPos | Axis::ImagNeg => inf.offset.and_then(|_| no_limit()),
                // `|F(g)| → ∞` while the argument `Im g` diverges.
                Axis::Other => no_limit(),
            }
        }
        // `sin`, `cos`: a bounded oscillation along the real axis, growth
        // with a turning argument off the axes; along the imaginary axis
        // `sin(±iy + a) ≈ ±(i/2)e^{y ∓ ia}`, `cos(±iy + a) ≈ e^{y ∓ ia}/2`.
        ExprNode::Sin(_) | ExprNode::Cos(_) => {
            let is_sin = matches!(node, ExprNode::Sin(_));
            match inf.axis {
                Axis::RealPos | Axis::RealNeg | Axis::Other => no_limit(),
                Axis::ImagPos | Axis::ImagNeg => {
                    let up = inf.axis == Axis::ImagPos;
                    if real_part_vanishes(arena, g, ap) {
                        // `cos(ih) = cosh h`, `sin(ih) = i·sinh h`.
                        return if is_sin {
                            imaginary_infinity_plus(arena, up, arena.zero(), var)
                        } else {
                            exact_result(arena.infinity())
                        };
                    }
                    let (a, _) = const_parts(arena, inf.offset?)?;
                    let i = arena.i_unit();
                    let ia = arena.mul(&[i, a]);
                    let ph = if up { arena.neg(ia) } else { ia };
                    let mut d = arena.exp(ph);
                    if is_sin {
                        let f = if up { i } else { arena.neg(i) };
                        d = arena.mul(&[f, d]);
                    }
                    let d = crate::transforms::eval::eval(arena, d);
                    if direction_axis(arena, d)? != Axis::Other {
                        return None;
                    }
                    infinite_value(
                        arena,
                        Infinite {
                            dir: d,
                            axis: Axis::Other,
                            offset: None,
                        },
                        var,
                    )
                    .map(Ok)
                }
            }
        }
        // `asinh z ≈ ln(2z)` for `Re z > 0`, `−ln(−2z)` for `Re z < 0`; on
        // its cut (`z = iy`) `±(ln 2|y|) ± iπ/2`.
        ExprNode::Asinh(_) => {
            let rs = match re_sign_of(arena, &inf) {
                Some(s) if s != 0 => s,
                _ if matches!(inf.axis, Axis::ImagPos | Axis::ImagNeg)
                    && real_part_vanishes(arena, g, ap) =>
                {
                    if inf.axis == Axis::ImagPos {
                        1
                    } else {
                        -1
                    }
                }
                _ => return None,
            };
            let im = match inf.axis {
                Axis::RealPos | Axis::RealNeg => arena.zero(),
                Axis::ImagPos => half_pi,
                Axis::ImagNeg => neg_half_pi,
                Axis::Other => {
                    let a = if rs > 0 {
                        const_arg(arena, inf.dir)?
                    } else {
                        let nd = arena.neg(inf.dir);
                        let a = const_arg(arena, nd)?;
                        arena.neg(a)
                    };
                    crate::transforms::eval::eval(arena, a)
                }
            };
            real_infinity_plus(arena, rs > 0, im, var)
        }
        // `acosh z ≈ ln(2z)` (on the cut below `−1`: `ln(2|z|) + iπ`).
        ExprNode::Acosh(_) => {
            let im = match inf.axis {
                Axis::RealPos => arena.zero(),
                Axis::ImagPos => half_pi,
                Axis::ImagNeg => neg_half_pi,
                Axis::Other => const_arg(arena, inf.dir)?,
                Axis::RealNeg => {
                    if eventually_real(arena, g, ap) || im_sign_of(arena, &inf)? > 0 {
                        pi
                    } else {
                        arena.neg(pi)
                    }
                }
            };
            real_infinity_plus(arena, true, im, var)
        }
        // `atanh z → ±iπ/2` with the sign of `Im z` (on the cut `−iπ/2`
        // beyond `1`, `+iπ/2` beyond `−1`).
        ExprNode::Atanh(_) => {
            let s = match im_sign_of(arena, &inf) {
                Some(s) if s != 0 => s,
                _ if (positive || negative) && eventually_real(arena, g, ap) => {
                    if positive {
                        -1
                    } else {
                        1
                    }
                }
                _ => return None,
            };
            let i = arena.i_unit();
            let v = arena.mul(&[i, if s > 0 { half_pi } else { neg_half_pi }]);
            finite_result(arena, v, var)
        }
        // `asin z ≈ −arg(−iz) + i·ln|2z|` for `Im z > 0`, `arg(iz) − i·ln|2z|`
        // for `Im z < 0` (on the cut: `π/2 − i∞` beyond `1`, `−π/2 + i∞`
        // beyond `−1`); `acos z = π/2 − asin z`.
        ExprNode::Asin(_) | ExprNode::Acos(_) => {
            let s = match im_sign_of(arena, &inf) {
                Some(s) if s != 0 => s,
                _ if (positive || negative) && eventually_real(arena, g, ap) => {
                    if positive {
                        -1
                    } else {
                        1
                    }
                }
                _ => return None,
            };
            let i = arena.i_unit();
            let re = if s > 0 {
                let ni = arena.neg(i);
                let w = arena.mul(&[ni, inf.dir]);
                let w = crate::transforms::eval::eval(arena, w);
                let a = const_arg(arena, w)?;
                arena.neg(a)
            } else {
                let w = arena.mul(&[i, inf.dir]);
                let w = crate::transforms::eval::eval(arena, w);
                const_arg(arena, w)?
            };
            if matches!(node, ExprNode::Asin(_)) {
                imaginary_infinity_plus(arena, s > 0, re, var)
            } else {
                let re = arena.sub(half_pi, re);
                imaginary_infinity_plus(arena, s < 0, re, var)
            }
        }
        _ => None,
    }
}

/// `lim F(g(x)) = F(lim g(x))` for an outer function `F` that is continuous
/// at the inner limit, extended to infinite inner limits via the
/// asymptotic values of `F` in their direction ([`unary_of_infinite`]).
/// Returns `None` when the rule does not apply (the caller then falls
/// through to the Gruntz algorithm); returns `Some(Err(_))` when the rule
/// proves the limit does not exist.
fn try_compose(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    dir: Direction,
    depth: usize,
) -> Option<Result<LimVal, SymplexError>> {
    let node = arena.node(expr).clone();
    let ap = Approach { var, point, dir };

    // Helper: limit of a sub-expression with the same direction.
    let inner_limit = |arena: &mut Arena, inner: ExprId| -> Option<LimVal> {
        limit_val(arena, inner, var, point, dir, depth + 1).ok()
    };

    match node {
        // ── Linear combinations with a single var-dependent part ──
        // An infinite part keeps the constant's component perpendicular to
        // its direction: `x − i → oo − I` (it was `oo`).
        ExprNode::Add(ref children) => {
            let (consts, deps) = split_by_var(arena, children, var);
            if deps.len() > 1 {
                return sum_of_limits(arena, expr, &consts, &deps, &ap, &inner_limit);
            }
            if deps.len() != 1 || consts.is_empty() {
                return None;
            }
            let c = arena.add(&consts);
            let l = inner_limit(arena, deps[0])?;
            match kind_of(arena, l)? {
                Kind::Finite(l) => {
                    let s = arena.add(&[c, l]);
                    finite_result(arena, s, var)
                }
                Kind::Infinite(inf) => {
                    let offset = inf.offset.map(|o| {
                        let s = arena.add(&[o, c]);
                        crate::transforms::eval::eval(arena, s)
                    });
                    infinite_value(arena, Infinite { offset, ..inf }, var).map(Ok)
                }
            }
        }
        ExprNode::Mul(ref children) => {
            let (consts, deps) = split_by_var(arena, children, var);
            if deps.len() > 1 {
                return product_of_limits(arena, expr, &consts, &deps, &ap, &inner_limit);
            }
            if deps.len() != 1 || consts.is_empty() {
                return None;
            }
            let c = arena.mul(&consts);
            let l = inner_limit(arena, deps[0])?;
            match kind_of(arena, l)? {
                Kind::Finite(l) => {
                    let p = arena.mul(&[c, l]);
                    finite_result(arena, p, var)
                }
                Kind::Infinite(inf) => {
                    let scaled = scale_inf(arena, inf, c)?;
                    infinite_value(arena, scaled, var).map(Ok)
                }
            }
        }
        ExprNode::Neg(inner) => {
            let l = inner_limit(arena, inner)?;
            match kind_of(arena, l)? {
                Kind::Finite(l) => {
                    let n = arena.neg(l);
                    finite_result(arena, n, var)
                }
                Kind::Infinite(inf) => {
                    let m1 = arena.neg_one();
                    let scaled = scale_inf(arena, inf, m1)?;
                    infinite_value(arena, scaled, var).map(Ok)
                }
            }
        }

        // ── Powers with a constant exponent ──
        ExprNode::Pow(base, exp) if !crate::base::walk::contains(arena, exp, var) => {
            let k = arena.as_num(exp).cloned();
            let l = inner_limit(arena, base)?;
            match kind_of(arena, l)? {
                Kind::Finite(l) => {
                    // A zero in disguise counts (`−π/2 − asin(−2) + i·ln(2 + √3)`,
                    // the limit of `asin(−2 + ix) − asin(−2)` at `0⁺`: its
                    // reciprocal was a "finite" `1/0`).
                    if is_zero_value(arena, l) {
                        // 0^k: continuous only for definite positive k.
                        return match &k {
                            Some(r) if r.is_positive() => exact_result(arena.zero()),
                            _ => None,
                        };
                    }
                    let fractional = k.as_ref().is_none_or(|r| !r.is_integer());
                    let l_sign = const_sign(arena, l);
                    if l_sign.is_none() && fractional {
                        // Symbolic base of unknown sign with a non-integer
                        // exponent — could be complex; leave to Gruntz.
                        return None;
                    }
                    // A negative limit of the base lies on the cut of a
                    // fractional power: the principal value there is the
                    // limit only for a real base (see `limit_on_branch_cut`).
                    if l_sign == Some(-1) && fractional && !inner_known_real(arena, base) {
                        return None;
                    }
                    let p = arena.pow(l, exp);
                    finite_result(arena, p, var)
                }
                Kind::Infinite(inf) => pow_of_infinite(arena, inf, base, exp, &k?, &ap),
            }
        }

        // ── Constant base, variable exponent: c^g = e^{g·ln c} ──
        ExprNode::Pow(base, exp) if !crate::base::walk::contains(arena, base, var) => {
            let c = arena.as_num(base).cloned()?;
            if !c.is_positive() {
                return None;
            }
            let l = inner_limit(arena, exp)?;
            let one = Ratio::from_integer(BigInt::from(1));
            match kind_of(arena, l)? {
                Kind::Finite(l) => {
                    let p = arena.pow(base, l);
                    finite_result(arena, p, var)
                }
                Kind::Infinite(_) if c == one => exact_result(arena.one()),
                Kind::Infinite(inf) => {
                    let ln_c = arena.ln(base);
                    let ln_c = crate::transforms::eval::eval(arena, ln_c);
                    let scaled = scale_inf(arena, inf, ln_c)?;
                    exp_of_infinite(arena, scaled, exp, &ap)
                }
            }
        }

        // ── f^g with both parts variable: exp(g·ln f) ──
        // The principal power is `exp(g·Log f)` for every `f ≠ 0`; with a
        // positive (or `+∞`) limit of `f` the logarithm is continuous there,
        // so the limit is `exp` of the limit of `g·ln f`.  Handing the whole
        // power to Gruntz instead split `(sin x/x)^(1/x²)` into
        // `x^(−x⁻²)·sin(x)^(x⁻²)`, two exponentials of the same class whose
        // cancellation exhausted the recursion depth (`e^(−1/6)`).
        ExprNode::Pow(base, exp) => {
            let lb = inner_limit(arena, base)?;
            let positive = match kind_of(arena, lb)? {
                Kind::Finite(l) => const_sign(arena, l) == Some(1),
                Kind::Infinite(inf) => inf.axis == Axis::RealPos,
            };
            if !positive {
                return None;
            }
            let ln_b = arena.ln(base);
            let prod = arena.mul(&[exp, ln_b]);
            let l = inner_limit(arena, prod)?;
            match kind_of(arena, l)? {
                Kind::Finite(l) => {
                    let p = arena.exp(l);
                    finite_result(arena, p, var)
                }
                Kind::Infinite(inf) => exp_of_infinite(arena, inf, prod, &ap),
            }
        }

        // ── Min / Max: continuous, extended to ±∞ ──
        ExprNode::Min(ref args) | ExprNode::Max(ref args) => {
            let is_min = matches!(node, ExprNode::Min(_));
            let args = args.clone();
            let mut limits = Vec::with_capacity(args.len());
            for &a in &args {
                limits.push(inner_limit(arena, a)?.value);
            }
            // Exact if all are numbers or real infinities: then the values
            // are points of the extended line and `Extended`'s order decides.
            let mut best: Option<(ExprId, Extended<Q>)> = None;
            let mut all_ordered = true;
            let (inf, ninf) = (arena.infinity(), arena.neg_infinity());
            for &l in &limits {
                let value: Extended<Q> = if l == inf {
                    Extended::PosInf
                } else if l == ninf {
                    Extended::NegInf
                } else if contains_singular_atom(arena, l) {
                    return None;
                } else {
                    match arena.as_num(l) {
                        Some(r) => Extended::Finite(r.clone()),
                        None => {
                            all_ordered = false;
                            break;
                        }
                    }
                };
                let better = match &best {
                    None => true,
                    Some((_, bvalue)) => {
                        let ord = value.cmp(bvalue);
                        if is_min {
                            ord == std::cmp::Ordering::Less
                        } else {
                            ord == std::cmp::Ordering::Greater
                        }
                    }
                };
                if better {
                    best = Some((l, value));
                }
            }
            if all_ordered {
                return best.map(|(l, _)| Ok(LimVal::exact(l)));
            }
            // Symbolic finite limits: rebuild the node and evaluate.
            if limits.iter().any(|&l| contains_singular_atom(arena, l)) {
                return None;
            }
            let sv: smallvec::SmallVec<[ExprId; 4]> = limits.iter().copied().collect();
            let rebuilt = if is_min {
                arena.intern(ExprNode::Min(sv))
            } else {
                arena.intern(ExprNode::Max(sv))
            };
            finite_result(arena, rebuilt, var)
        }

        // ── Unary functions ──
        ExprNode::Exp(a) => {
            let l = inner_limit(arena, a)?;
            match kind_of(arena, l)? {
                Kind::Finite(l) => {
                    let v = arena.exp(l);
                    finite_result(arena, v, var)
                }
                Kind::Infinite(inf) => exp_of_infinite(arena, inf, a, &ap),
            }
        }
        ExprNode::Ln(a) => {
            let l = inner_limit(arena, a)?;
            match kind_of(arena, l)? {
                Kind::Finite(l) => {
                    if is_zero_value(arena, l) {
                        return None; // ln 0: one-sided behaviour, leave to Gruntz
                    }
                    match const_sign(arena, l) {
                        Some(s) if s > 0 => {
                            let v = arena.ln(l);
                            finite_result(arena, v, var)
                        }
                        _ => None,
                    }
                }
                Kind::Infinite(inf) => ln_of_infinite(arena, inf, a, &ap),
            }
        }
        ExprNode::Atan(a) => unary_with_asymptotes(
            arena,
            &node,
            BranchCut::ImaginaryAxis,
            a,
            &ap,
            &inner_limit,
            |arena, l| arena.atan(l),
        ),
        ExprNode::Erf(a) => unary_with_asymptotes(
            arena,
            &node,
            BranchCut::None,
            a,
            &ap,
            &inner_limit,
            |arena, l| arena.erf(l),
        ),
        ExprNode::Erfc(a) => unary_with_asymptotes(
            arena,
            &node,
            BranchCut::None,
            a,
            &ap,
            &inner_limit,
            |arena, l| arena.erfc(l),
        ),
        ExprNode::Tanh(a) => unary_with_asymptotes(
            arena,
            &node,
            BranchCut::None,
            a,
            &ap,
            &inner_limit,
            |arena, l| arena.tanh(l),
        ),
        ExprNode::Sinh(a) => unary_with_asymptotes(
            arena,
            &node,
            BranchCut::None,
            a,
            &ap,
            &inner_limit,
            |arena, l| arena.sinh(l),
        ),
        ExprNode::Cosh(a) => unary_with_asymptotes(
            arena,
            &node,
            BranchCut::None,
            a,
            &ap,
            &inner_limit,
            |arena, l| arena.cosh(l),
        ),
        ExprNode::Asinh(a) => unary_with_asymptotes(
            arena,
            &node,
            BranchCut::ImaginaryAxis,
            a,
            &ap,
            &inner_limit,
            |arena, l| arena.asinh(l),
        ),
        ExprNode::Acosh(a) => unary_with_asymptotes(
            arena,
            &node,
            BranchCut::BelowOne,
            a,
            &ap,
            &inner_limit,
            |arena, l| arena.acosh(l),
        ),
        ExprNode::Atanh(a) => unary_with_asymptotes(
            arena,
            &node,
            BranchCut::BeyondOne,
            a,
            &ap,
            &inner_limit,
            |arena, l| arena.atanh(l),
        ),
        ExprNode::Abs(a) => unary_with_asymptotes(
            arena,
            &node,
            BranchCut::None,
            a,
            &ap,
            &inner_limit,
            |arena, l| arena.abs(l),
        ),
        ExprNode::Sin(a) => unary_with_asymptotes(
            arena,
            &node,
            BranchCut::None,
            a,
            &ap,
            &inner_limit,
            |arena, l| arena.sin(l),
        ),
        ExprNode::Cos(a) => unary_with_asymptotes(
            arena,
            &node,
            BranchCut::None,
            a,
            &ap,
            &inner_limit,
            |arena, l| arena.cos(l),
        ),
        ExprNode::Asin(a) => unary_with_asymptotes(
            arena,
            &node,
            BranchCut::BeyondOne,
            a,
            &ap,
            &inner_limit,
            |arena, l| arena.asin(l),
        ),
        ExprNode::Acos(a) => unary_with_asymptotes(
            arena,
            &node,
            BranchCut::BeyondOne,
            a,
            &ap,
            &inner_limit,
            |arena, l| arena.acos(l),
        ),
        ExprNode::Gamma(a) => {
            let l = inner_limit(arena, a)?;
            match kind_of(arena, l)? {
                Kind::Infinite(inf) => gamma_of_infinite(arena, inf, a, &ap, false),
                Kind::Finite(l) => {
                    if is_nonpositive_integer(arena, l) {
                        return None;
                    }
                    let v = arena.gamma(l);
                    finite_result(arena, v, var)
                }
            }
        }
        ExprNode::LogGamma(a) | ExprNode::Digamma(a) | ExprNode::LambertW(a) => {
            let l = inner_limit(arena, a)?;
            match kind_of(arena, l)? {
                Kind::Infinite(inf) => match (&node, inf.axis) {
                    // `ln Γ(z) ≈ z ln z`: real for a real `g`; for `Im g → b ≠ 0`
                    // its imaginary part `≈ b·ln x` diverges more slowly.
                    (ExprNode::LogGamma(_), Axis::RealPos) => {
                        if eventually_real(arena, a, &ap) {
                            return exact_result(arena.infinity());
                        }
                        let b = offset_im(arena, inf.offset?)?;
                        (const_sign(arena, b)? != 0).then(|| {
                            Ok(LimVal {
                                value: arena.infinity(),
                                weak: true,
                            })
                        })
                    }
                    // `ψ(z) ≈ ln z` off the negative axis (poles on it).
                    (ExprNode::Digamma(_), Axis::RealNeg) => None,
                    (ExprNode::Digamma(_), _) => ln_of_infinite(arena, inf, a, &ap),
                    // `W(z) ≈ ln z − ln ln z`.
                    (ExprNode::LambertW(_), Axis::RealPos) => exact_result(arena.infinity()),
                    _ => None,
                },
                Kind::Finite(l) => {
                    if matches!(node, ExprNode::LogGamma(_) | ExprNode::Digamma(_))
                        && is_nonpositive_integer(arena, l)
                    {
                        return None;
                    }
                    // On the cut of `ln Γ` or `W` (see `limit_on_branch_cut`):
                    // `ln Γ(−5/2 − ix) → ln Γ(−5/2) + 6πi` from `x > 0`.
                    if let Some((_, cut)) = branch_cut_of(arena, expr)
                        && limit_on_branch_cut(arena, cut, l, a)
                    {
                        return None;
                    }
                    let v = match node {
                        ExprNode::LogGamma(_) => arena.log_gamma(l),
                        ExprNode::Digamma(_) => arena.digamma(l),
                        _ => arena.lambertw(l),
                    };
                    finite_result(arena, v, var)
                }
            }
        }
        ExprNode::Factorial(a) => {
            let l = inner_limit(arena, a)?;
            match kind_of(arena, l)? {
                Kind::Infinite(inf) => gamma_of_infinite(arena, inf, a, &ap, true),
                Kind::Finite(l) => {
                    // A pole of x! = Γ(x + 1): leave it to the one-sided
                    // analysis, as for Γ itself.
                    if arena
                        .as_num(l)
                        .is_some_and(|r| r.is_integer() && r.is_negative())
                    {
                        return None;
                    }
                    let v = arena.factorial(l);
                    finite_result(arena, v, var)
                }
            }
        }
        ExprNode::Sign(a) | ExprNode::Heaviside(a) => {
            let is_sign = matches!(node, ExprNode::Sign(_));
            let l = inner_limit(arena, a)?;
            let s = match kind_of(arena, l)? {
                // `sign z = z/|z|` tends to the direction of an infinite
                // `g`: `sign(cosh(ln x + i)) → e^i` (it was 1).
                Kind::Infinite(inf) if is_sign => {
                    let v = match inf.axis {
                        Axis::RealPos => arena.one(),
                        Axis::RealNeg => arena.neg_one(),
                        Axis::ImagPos => arena.i_unit(),
                        Axis::ImagNeg => {
                            let i = arena.i_unit();
                            arena.neg(i)
                        }
                        Axis::Other => unit_direction(arena, inf.dir),
                    };
                    return finite_result(arena, v, var);
                }
                // `H` of a real argument only.
                Kind::Infinite(inf) => {
                    if !eventually_real(arena, a, &ap) {
                        return None;
                    }
                    match inf.axis {
                        Axis::RealPos => 1,
                        Axis::RealNeg => -1,
                        _ => return None,
                    }
                }
                Kind::Finite(l) => match arena.as_num(l) {
                    Some(r) if !r.is_zero() => sign_of_ratio(r),
                    _ => return None,
                },
            };
            exact_result(if is_sign {
                arena.int(s as i64)
            } else if s > 0 {
                arena.one()
            } else {
                arena.zero()
            })
        }
        ExprNode::Floor(a) | ExprNode::Ceiling(a) => {
            let is_floor = matches!(node, ExprNode::Floor(_));
            let l = inner_limit(arena, a)?;
            match kind_of(arena, l)? {
                Kind::Infinite(inf) => {
                    if !eventually_real(arena, a, &ap) {
                        return None;
                    }
                    match inf.axis {
                        Axis::RealPos => exact_result(arena.infinity()),
                        Axis::RealNeg => exact_result(arena.neg_infinity()),
                        _ => None,
                    }
                }
                Kind::Finite(l) => {
                    let r = arena.as_num(l).cloned()?;
                    if r.is_integer() {
                        return None;
                    }
                    let v = if is_floor { r.floor() } else { r.ceil() };
                    let nid = arena.intern_num(v);
                    exact_result(arena.intern(ExprNode::Num(nid)))
                }
            }
        }
        _ => None,
    }
}

/// A sum of several `var`-dependent terms that is not eventually real
/// (real ones are left to the Gruntz algorithm, which sees cancellations):
/// the sum of the terms' limits when at most one of them is infinite, or
/// all infinite ones lie on one axis with known perpendicular parts
/// (`e^{x² + 2i} − erf(ln((−2 − i)x)) → e^{2i}·oo`; it was `oo`).
fn sum_of_limits(
    arena: &mut Arena,
    expr: ExprId,
    consts: &[ExprId],
    deps: &[ExprId],
    ap: &Approach,
    inner_limit: &dyn Fn(&mut Arena, ExprId) -> Option<LimVal>,
) -> Option<Result<LimVal, SymplexError>> {
    if eventually_real(arena, expr, ap) {
        return None;
    }
    let mut finite: Vec<ExprId> = consts.to_vec();
    let mut infinite: Vec<Infinite> = Vec::new();
    let mut infinite_terms: Vec<ExprId> = Vec::new();
    for &d in deps {
        let l = inner_limit(arena, d)?;
        match kind_of(arena, l)? {
            Kind::Finite(v) => finite.push(v),
            Kind::Infinite(inf) => {
                infinite.push(inf);
                infinite_terms.push(d);
            }
        }
    }
    let c = arena.add(&finite);
    let c = crate::transforms::eval::eval(arena, c);
    tracing::debug!(
        finite = %arena.display(c).to_string(),
        infinite = infinite.len(),
        "limit: sum of the terms' limits"
    );
    let Some(&first) = infinite.first() else {
        return finite_result(arena, c, ap.var);
    };
    // Two infinities in different directions: one that dominates the other
    // (their ratio tends to `∞`) and lies off the axes gives the direction
    // (`acos(x/2) + e^{πx² + 1 − i} → e^{−i}·oo`; it was `oo`).
    if let ([a, b], [ta, tb]) = (infinite.as_slice(), infinite_terms.as_slice())
        && a.axis != b.axis
    {
        // `|a/b| → ∞` or `0`, tried both ways round (the quotient with the
        // dominant term below has a finite limit more often).
        let mut order = None;
        for (num, den, first) in [(*tb, *ta, false), (*ta, *tb, true)] {
            let q = arena.div(num, den);
            let Some(lq) = inner_limit(arena, q) else {
                continue;
            };
            order = match kind_of(arena, lq)? {
                Kind::Infinite(_) => Some(first),
                Kind::Finite(l) if is_zero_value(arena, l) => Some(!first),
                Kind::Finite(_) => return None,
            };
            break;
        }
        let (dominant, other) = if order? { (*a, *b) } else { (*b, *a) };
        let real_axis = |x: Axis| matches!(x, Axis::RealPos | Axis::RealNeg);
        let imag_axis = |x: Axis| matches!(x, Axis::ImagPos | Axis::ImagNeg);
        let offset = match (dominant.axis, dominant.offset, other.offset) {
            (Axis::Other, _, _) => None,
            // Opposite directions on one axis: the perpendicular parts add
            // (`√(x − i) − acosh(x − i) → oo`).
            (d, Some(od), Some(oo_))
                if (real_axis(d) && real_axis(other.axis))
                    || (imag_axis(d) && imag_axis(other.axis)) =>
            {
                let s = arena.add(&[od, oo_, c]);
                Some(crate::transforms::eval::eval(arena, s))
            }
            // The other one diverges across the dominant axis: so does the
            // perpendicular part of the sum (a weak value).
            (d, Some(_), _)
                if (real_axis(d) && !real_axis(other.axis))
                    || (imag_axis(d) && !imag_axis(other.axis)) =>
            {
                None
            }
            _ => return None,
        };
        return infinite_value(arena, Infinite { offset, ..dominant }, ap.var).map(Ok);
    }
    if infinite.len() > 1
        && (first.axis == Axis::Other
            || infinite
                .iter()
                .any(|i| i.axis != first.axis || i.offset.is_none()))
    {
        return None;
    }
    let offset = if infinite.len() > 1 {
        let mut parts: Vec<ExprId> = infinite.iter().filter_map(|i| i.offset).collect();
        parts.push(c);
        let s = arena.add(&parts);
        Some(crate::transforms::eval::eval(arena, s))
    } else {
        first.offset.map(|o| {
            let s = arena.add(&[o, c]);
            crate::transforms::eval::eval(arena, s)
        })
    };
    infinite_value(arena, Infinite { offset, ..first }, ap.var).map(Ok)
}

/// A product of several `var`-dependent factors that is not eventually
/// real: the product of their limits when none of them is 0 while another
/// is infinite, with the directions multiplied (`|1/(x − 1)|·sinh(1/(x − 1) +
/// i)` at `1⁻` tends to `−e^{−i}·oo`; it was `oo`).
fn product_of_limits(
    arena: &mut Arena,
    expr: ExprId,
    consts: &[ExprId],
    deps: &[ExprId],
    ap: &Approach,
    inner_limit: &dyn Fn(&mut Arena, ExprId) -> Option<LimVal>,
) -> Option<Result<LimVal, SymplexError>> {
    if eventually_real(arena, expr, ap) {
        return None;
    }
    let mut finite: Vec<ExprId> = consts.to_vec();
    let mut infinite: Vec<Infinite> = Vec::new();
    let mut varying = false;
    for &d in deps {
        let l = inner_limit(arena, d)?;
        match kind_of(arena, l)? {
            Kind::Finite(v) => {
                finite.push(v);
                varying = true;
            }
            Kind::Infinite(inf) => infinite.push(inf),
        }
    }
    let c = arena.mul(&finite);
    let c = crate::transforms::eval::eval(arena, c);
    let Some(&first) = infinite.first() else {
        return finite_result(arena, c, ap.var);
    };
    // `0·∞`, or a finite factor of unknown sign: undecided here.
    direction_axis(arena, c)?;
    if infinite.len() == 1 {
        let scaled = scale_inf(arena, first, c)?;
        // A factor `h → c` that is not constant adds `Re g·Im(h − c)` to the
        // perpendicular part (`x·e^{i/x} → oo + I`): unknown on an axis.
        if varying && scaled.axis != Axis::Other {
            return None;
        }
        return infinite_value(arena, scaled, ap.var).map(Ok);
    }
    let mut dirs: Vec<ExprId> = infinite.iter().map(|i| i.dir).collect();
    dirs.push(c);
    let d = arena.mul(&dirs);
    let d = crate::transforms::eval::eval(arena, d);
    if direction_axis(arena, d)? != Axis::Other {
        // The perpendicular part of a product of infinities is unknown.
        return None;
    }
    infinite_value(
        arena,
        Infinite {
            dir: d,
            axis: Axis::Other,
            offset: None,
        },
        ap.var,
    )
    .map(Ok)
}

/// Split n-ary children into (`var`-free, `var`-dependent).
fn split_by_var(arena: &Arena, children: &[ExprId], var: ExprId) -> (Vec<ExprId>, Vec<ExprId>) {
    let mut consts = Vec::new();
    let mut deps = Vec::new();
    for &c in children {
        if crate::base::walk::contains(arena, c, var) {
            deps.push(c);
        } else {
            consts.push(c);
        }
    }
    (consts, deps)
}

/// Compositional rule for a unary function `F` (the node `node`) that is
/// continuous on its domain off its branch cut `cut`, with `build`
/// constructing `F(l)`; an infinite inner limit goes to
/// [`unary_of_infinite`].
fn unary_with_asymptotes(
    arena: &mut Arena,
    node: &ExprNode,
    cut: BranchCut,
    inner: ExprId,
    ap: &Approach,
    inner_limit: &dyn Fn(&mut Arena, ExprId) -> Option<LimVal>,
    build: impl Fn(&mut Arena, ExprId) -> ExprId,
) -> Option<Result<LimVal, SymplexError>> {
    let l = inner_limit(arena, inner)?;
    match kind_of(arena, l)? {
        Kind::Finite(l) => {
            let v = build(arena, l);
            if limit_on_branch_cut(arena, cut, l, inner) {
                return None;
            }
            finite_result(arena, v, ap.var)
        }
        Kind::Infinite(inf) => unary_of_infinite(arena, node, inner, inf, ap),
    }
}

/// Does the composition `f(g) → f(l)` (with `v = f(l)` built) take `f` at a
/// point `l` of its branch cut, where `f` is continuous only from one side?
/// `asinh` and `atan` have their cuts on the imaginary axis beyond `±i`,
/// `acosh` on `(−∞, 1)`, `asin`, `acos`, `atanh` beyond `±1`.  A real `g`
/// approaching a real point of a real-axis cut stays on the cut, where the
/// principal values are continuous along it; a non-real `g` may approach
/// from the other side: `atanh(1 − x) → iπ/2` from `Re < 0` as `x → ∞`,
/// and `lim asinh(atanh(1 − x))` is `−acosh(π/2) + iπ/2`, not the value
/// `asinh(iπ/2) = acosh(π/2) + iπ/2` on the cut.  Such limits are left to
/// the one-sided analysis.
fn limit_on_branch_cut(arena: &mut Arena, cut: BranchCut, l: ExprId, g: ExprId) -> bool {
    match cut {
        BranchCut::ImaginaryAxis => on_branch_cut(arena, cut, l),
        _ => on_branch_cut(arena, cut, l) && !inner_known_real(arena, g),
    }
}

/// Is the constant `l` on the branch cut `cut`?  (`NegativeReals` is the
/// cut of `ln` and of fractional powers.)
pub(crate) fn on_branch_cut(arena: &mut Arena, cut: BranchCut, l: ExprId) -> bool {
    if cut == BranchCut::None || !crate::base::walk::free_symbols(arena, l).is_empty() {
        return false;
    }
    let re = arena.re(l);
    let re = crate::transforms::eval::eval(arena, re);
    let im = arena.im(l);
    let im = crate::transforms::eval::eval(arena, im);
    let (Ok(re), Ok(im)) = (
        crate::transforms::evalf::evalf_complex(arena, re, 20),
        crate::transforms::evalf::evalf_complex(arena, im, 20),
    ) else {
        return false;
    };
    let is_zero = |x: &crate::base::bigcomplex::Complex| x.0.is_zero() && x.1.is_zero();
    let magnitude = |x: &crate::base::bigcomplex::Complex| {
        crate::transforms::evalf::abs_to_f64(x).unwrap_or(0.0)
    };
    match cut {
        BranchCut::None => false,
        BranchCut::ImaginaryAxis => is_zero(&re) && magnitude(&im) >= 1.0,
        BranchCut::NegativeReals => is_zero(&im) && re.0.is_negative(),
        BranchCut::BelowOne => is_zero(&im) && (re.0.is_negative() || magnitude(&re) < 1.0),
        BranchCut::BeyondOne => is_zero(&im) && magnitude(&re) > 1.0,
        BranchCut::AboveOne => is_zero(&im) && re.0.is_positive() && magnitude(&re) > 1.0,
        BranchCut::BelowMinusInvE => {
            if !is_zero(&im) || !re.0.is_negative() {
                return false;
            }
            let inv_e = (-1.0f64).exp();
            let m = magnitude(&re);
            if (m - inv_e).abs() > 1e-9 {
                return m > inv_e;
            }
            // Next to the branch point `−1/e`: decide `l + 1/e < 0`
            // exactly; an undecided sign counts as on the cut.
            let re_l = arena.re(l);
            let m1 = arena.neg_one();
            let e_inv = arena.exp(m1);
            let d = arena.add(&[re_l, e_inv]);
            const_sign(arena, d).is_none_or(|s| s < 0)
        }
    }
}

/// The argument of node `id` and the branch cut of its function, for the
/// functions with one: `ln`, `ln Γ`, a fractional power, the inverse
/// trigonometric and hyperbolic functions, and (since 0.37) the special
/// functions with a cut — `Ei`, `Ci`, `Chi`, `Eₙ`, `Kν`, `Yν`, `Jν`/`Iν`
/// of a non-integer order and `Γ(s, ·)`, `γ(s, ·)` of an `s` other than a
/// positive integer on the negative reals, `li` and `acosh` on `(−∞, 1)`,
/// `polylog(s, ·)` and the complete elliptic integrals on `(1, ∞)`, and
/// `W` on `(−∞, −1/e)`.  Before, a limit there took the value on the cut
/// from both sides: `limit(Ei(−2 + ix), x, 0, '+')` was `Ei(−2)` (it is
/// `Ei(−2) + iπ`, mpmath), `limit(Ci(−2 + ix), x, 0, '-')` was
/// `Ci(−2) = Ci(2) + iπ` (it is `Ci(2) − iπ`).
pub(crate) fn branch_cut_of(arena: &Arena, id: ExprId) -> Option<(ExprId, BranchCut)> {
    use crate::base::libfn::LibFn;
    let not_positive_integer = |e: ExprId| {
        arena
            .as_num(e)
            .is_none_or(|r| !(r.is_integer() && r.is_positive()))
    };
    let not_integer = |e: ExprId| arena.as_num(e).is_none_or(|r| !r.is_integer());
    // `E₀`, `E₋ₙ` and `Li₋ₙ` are rational in `e^{−z}`, `z`: no cut.
    let not_nonpositive_integer = |e: ExprId| {
        arena
            .as_num(e)
            .is_none_or(|r| !(r.is_integer() && !r.is_positive()))
    };
    Some(match *arena.node(id) {
        ExprNode::Ln(a) | ExprNode::LogGamma(a) | ExprNode::Ei(a) | ExprNode::Ci(a) => {
            (a, BranchCut::NegativeReals)
        }
        ExprNode::Pow(b, e) if arena.as_num(e).is_some_and(|r| !r.is_integer()) => {
            (b, BranchCut::NegativeReals)
        }
        ExprNode::Asin(a) | ExprNode::Acos(a) | ExprNode::Atanh(a) => (a, BranchCut::BeyondOne),
        ExprNode::Acosh(a) | ExprNode::Li(a) => (a, BranchCut::BelowOne),
        ExprNode::Asinh(a) | ExprNode::Atan(a) => (a, BranchCut::ImaginaryAxis),
        ExprNode::LambertW(a) => (a, BranchCut::BelowMinusInvE),
        ExprNode::Apply(f, ref args) => match (arena.lib_fn(f)?, args.as_slice()) {
            (LibFn::Chi, &[a]) => (a, BranchCut::NegativeReals),
            (LibFn::ExpInt, &[n, z]) if not_nonpositive_integer(n) => (z, BranchCut::NegativeReals),
            (LibFn::BesselK | LibFn::BesselY, &[_, z]) => (z, BranchCut::NegativeReals),
            (LibFn::BesselJ | LibFn::BesselI, &[nu, z]) if not_integer(nu) => {
                (z, BranchCut::NegativeReals)
            }
            (LibFn::UpperGamma | LibFn::LowerGamma, &[s, z]) if not_positive_integer(s) => {
                (z, BranchCut::NegativeReals)
            }
            (LibFn::PolyLog, &[s, z]) if not_nonpositive_integer(s) => (z, BranchCut::AboveOne),
            (LibFn::EllipticK | LibFn::EllipticE, &[m]) => (m, BranchCut::AboveOne),
            _ => return None,
        },
        _ => return None,
    })
}

/// `f(u)` for a node `id` with a branch cut ([`branch_cut_of`]) whose
/// argument `u` tends to the point `l` of the cut, as an expression in `u`
/// that is the principal `f(u)` on the side of the cut `u` approaches from
/// and is analytic at `l` (the continuation of `f` across the cut from that
/// side), so that the limit and series machinery may expand it there.
/// `side` is the eventual sign of `Im u` (of `Re u` for the cuts on the
/// imaginary axis), `0` when `u` moves along the cut, where the principal
/// value is the limit from one side.  `along` is the eventual sign of
/// `u − l` along the cut, needed only where `u` moves along the cut to the
/// branch point `−1` of `acosh` (`0`: unknown).
///
/// * `ln u = ln(−u) ± iπ`, `u^r = (−u)^r·e^{±iπr}` (`+` above or on the cut),
/// * `asin u = π/2 ± i·acosh u` (`l > 1`; `−` below or on the cut),
///   `−π/2 ± i·acosh(−u)` (`l < −1`; `+` above or on it), `acos u = π/2 − asin u`,
/// * `atanh u = atanh(1/u) ± iπ/2` (`+` above, `−` below; on the cut `−` for
///   `l > 1`, `+` for `l < −1`),
/// * `acosh u = ±i·acos u` (`−1 < l < 1`), `acosh(−u) ± iπ` (`l ≤ −1`; `+` above
///   or on the cut),
/// * `asinh u = iπ/2 ± acosh(−iu)` (`l = iy`, `y ≥ 1`; `+` for `Re u ≥ 0`),
///   `−iπ/2 ± acosh(iu)` (`y ≤ −1`; `+` for `Re u > 0`),
/// * `atan u = ±π/2 − atan(1/u)` (`+` for `Re u > 0`, and on the cut for `y > 1`).
///
/// These are SymPy's one-sided expansions at a branch cut in closed form
/// (`log._eval_nseries` adds `−2πi` below the negative axis,
/// `asin._eval_nseries` gives `π − asin` above `(1, ∞)` and `−π − asin`
/// below `(−∞, −1)`, `atanh`, `acosh`, `asinh`, `atan` likewise); each form
/// is the principal value on its whole open half-plane and on the part of
/// the cut the principal value is continuous with (checked against mpmath
/// at 3,960 points).  `None` for `ln Γ`, and at a branch point (`±i` for
/// `asinh`, `atan`; `−1` for `acosh`) reached along the cut.
pub(crate) fn continuation_across_cut(
    arena: &mut Arena,
    id: ExprId,
    l: ExprId,
    side: i32,
    along: i32,
) -> Option<ExprId> {
    let (u, _) = branch_cut_of(arena, id)?;
    let node = arena.node(id).clone();
    let i = arena.i_unit();
    let pi = arena.pi();
    let half = arena.rational(1, 2);
    let half_pi = arena.mul(&[half, pi]);
    let i_pi = arena.mul(&[i, pi]);
    let i_half_pi = arena.mul(&[i, half_pi]);
    let neg_u = arena.neg(u);
    let signed = |arena: &mut Arena, s: i32, v: ExprId| if s >= 0 { v } else { arena.neg(v) };
    // `+1` above (or right of) the cut, `−1` below; on the cut `on_cut`.
    let pick = |on_cut: i32| match side {
        s if s > 0 => 1,
        s if s < 0 => -1,
        _ => on_cut,
    };
    let re_l = arena.re(l);
    let re_l = crate::transforms::eval::eval(arena, re_l);
    let im_l = arena.im(l);
    let im_l = crate::transforms::eval::eval(arena, im_l);
    let g = match node {
        ExprNode::Ln(_) => {
            let ln = arena.ln(neg_u);
            let shift = signed(arena, pick(1), i_pi);
            arena.add(&[ln, shift])
        }
        ExprNode::Pow(_, r) => {
            let p = arena.pow(neg_u, r);
            let phase = arena.mul(&[i_pi, r]);
            let phase = signed(arena, pick(1), phase);
            let rot = arena.exp(phase);
            arena.mul(&[p, rot])
        }
        ExprNode::Asin(_) | ExprNode::Acos(_) => {
            let beyond = const_sign(arena, re_l)?;
            let (c, v, t) = if beyond > 0 {
                (half_pi, u, pick(-1))
            } else {
                (arena.neg(half_pi), neg_u, pick(1))
            };
            let ach = arena.acosh(v);
            let term = arena.mul(&[i, ach]);
            let term = signed(arena, t, term);
            let asin = arena.add(&[c, term]);
            if matches!(node, ExprNode::Asin(_)) {
                asin
            } else {
                arena.sub(half_pi, asin)
            }
        }
        ExprNode::Atanh(_) => {
            let below_minus_one = const_sign(arena, re_l)? < 0;
            let t = pick(if below_minus_one { 1 } else { -1 });
            let inv = arena.pow(u, arena.neg_one());
            let at = arena.atanh(inv);
            let shift = signed(arena, t, i_half_pi);
            arena.add(&[at, shift])
        }
        ExprNode::Acosh(_) => {
            let one = arena.one();
            let lp1 = arena.add(&[re_l, one]);
            let inside = const_sign(arena, lp1)?;
            let inside = if inside == 0 && side == 0 {
                along
            } else {
                inside
            };
            match inside {
                // −1 < l < 1 (or l = −1 reached from the right along the cut)
                s if s > 0 => {
                    let ac = arena.acos(u);
                    let iac = arena.mul(&[i, ac]);
                    signed(arena, pick(1), iac)
                }
                0 if side == 0 => return None,
                _ => {
                    let ach = arena.acosh(neg_u);
                    let shift = signed(arena, pick(1), i_pi);
                    arena.add(&[ach, shift])
                }
            }
        }
        ExprNode::Asinh(_) | ExprNode::Atan(_) => {
            let upper = const_sign(arena, im_l)?;
            let one = arena.one();
            let abs_y = if upper > 0 { im_l } else { arena.neg(im_l) };
            let beyond = arena.sub(abs_y, one);
            let beyond = const_sign(arena, beyond)?;
            if beyond < 0 || (beyond == 0 && side == 0) {
                return None;
            }
            if matches!(node, ExprNode::Atan(_)) {
                let t = pick(upper);
                let inv = arena.pow(u, arena.neg_one());
                let at = arena.atan(inv);
                let c = signed(arena, t, half_pi);
                arena.sub(c, at)
            } else if upper > 0 {
                let v = arena.mul(&[i, u]);
                let v = arena.neg(v);
                let ach = arena.acosh(v);
                let term = signed(arena, pick(1), ach);
                arena.add(&[i_half_pi, term])
            } else {
                let v = arena.mul(&[i, u]);
                let ach = arena.acosh(v);
                let term = signed(arena, pick(-1), ach);
                let c = arena.neg(i_half_pi);
                arena.add(&[c, term])
            }
        }
        // `Ei u = −E₁(−u) ± iπ` off the axis, `−E₁(−u)` on it (the principal
        // `Ei` of a negative real is real: the mean of its two sides).
        ExprNode::Ei(_) => {
            let one = arena.one();
            let e1 = arena.lib_apply(crate::base::libfn::LibFn::ExpInt, &[one, neg_u]);
            let ne1 = arena.neg(e1);
            match side.signum() {
                0 => ne1,
                s => {
                    let shift = signed(arena, s, i_pi);
                    arena.add(&[ne1, shift])
                }
            }
        }
        // `Ci u = Ci(−u) ± iπ`, `Chi u = Chi(−u) ± iπ` (`+` above or on the cut).
        ExprNode::Ci(_) => {
            let c = arena.ci(neg_u);
            let shift = signed(arena, pick(1), i_pi);
            arena.add(&[c, shift])
        }
        ExprNode::Apply(f, _) if arena.lib_fn(f) == Some(crate::base::libfn::LibFn::Chi) => {
            let c = arena.lib_apply(crate::base::libfn::LibFn::Chi, &[neg_u]);
            let shift = signed(arena, pick(1), i_pi);
            arena.add(&[c, shift])
        }
        // `ln Γ(u) = ln((−1)ⁿ⁺¹Γ(u)) ∓ (n + 1)iπ` for `−n − 1 < l < −n`
        // (`−` above or on the cut): `(−1)ⁿ⁺¹Γ(l) > 0` there.
        ExprNode::LogGamma(_) => {
            let n = floor_of_negated(arena, l)?;
            let g = arena.gamma(u);
            let g = if n % 2 == 0 { arena.neg(g) } else { g };
            let ln = arena.ln(g);
            let k = arena.int(n + 1);
            let shift = arena.mul(&[k, i_pi]);
            let shift = signed(arena, -pick(1), shift);
            arena.add(&[ln, shift])
        }
        _ => return None,
    };
    Some(g)
}

/// `⌊−l⌋` for a negative real constant `l` that is not an integer (`None`
/// for an integer, or a value too close to one to tell).
pub(crate) fn floor_of_negated(arena: &mut Arena, l: ExprId) -> Option<i64> {
    use num_traits::ToPrimitive;
    if let Some(r) = arena.as_num(l) {
        if r.is_integer() {
            return None;
        }
        return (-r).floor().to_integer().to_i64();
    }
    let v = crate::transforms::evalf::evalf_f64(arena, l).ok()?;
    let m = -v;
    if !m.is_finite() || m > 1e15 || (m - m.round()).abs() < 1e-9 {
        return None;
    }
    Some(m.floor() as i64)
}

/// Where a function composed in [`unary_with_asymptotes`] has its branch
/// cut (see [`limit_on_branch_cut`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BranchCut {
    /// Continuous everywhere it is finite.
    None,
    /// `iy` with `|y| ≥ 1` (`asinh`, `atan`).
    ImaginaryAxis,
    /// Negative reals (`ln`, fractional powers).
    NegativeReals,
    /// Reals below 1 (`acosh`).
    BelowOne,
    /// Reals beyond ±1 (`asin`, `acos`).
    BeyondOne,
    /// Reals above 1 (`polylog`, the complete elliptic integrals).
    AboveOne,
    /// Reals below `−1/e` (Lambert `W`).
    BelowMinusInvE,
}

/// Is `g` real for real values of its symbols (undeclared ones count as
/// real)?
pub(crate) fn inner_known_real(arena: &Arena, g: ExprId) -> bool {
    use crate::base::assumptions::{AssumptionCache, Assumptions, Props};
    let mut cache = AssumptionCache::new();
    for s in crate::base::walk::free_symbols(arena, g) {
        if let ExprNode::Symbol(sid) = *arena.node(s)
            && arena.symbol_assumptions(sid) == Assumptions::default()
        {
            let mut real = Assumptions::default();
            real.known_true |= Props::REAL;
            cache.set_symbol_assumptions(s, real);
        }
    }
    cache.query(arena, g, Props::REAL) == Some(true)
}

// ═══════════════════════════════════════════════════════════════════════════
// (c) One-sided limits via Gruntz
// ═══════════════════════════════════════════════════════════════════════════

/// `lim_{x → a^±} f(x)` via the substitution `x = a ± 1/w`, `w → +∞`.
fn one_sided_gruntz(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    right: bool,
    depth: usize,
) -> Result<LimVal, SymplexError> {
    // w → +∞ along the reals: declared positive, like Gruntz's own variable.
    let w = arena.positive_symbol(LIM_DUMMY);
    let one = arena.one();
    let inv_w = arena.div(one, w);
    let shift = if right { inv_w } else { arena.neg(inv_w) };
    let sub_point = arena.add(&[point, shift]);
    let e = crate::transforms::subs::subs(arena, expr, var, sub_point);

    // Clear nested fractions and simplify (same preprocessing as `gruntz`).
    let e = crate::poly::polybridge::together(arena, e);
    let e = crate::poly::polybridge::cancel(arena, e, w);
    let e = crate::transforms::eval::eval(arena, e);
    // Expanding is a convenience, and on a large expression (a high
    // derivative from the series fallback: `acosh((x − 1/2)·sign x)`)
    // it multiplied out powers of sums for minutes.
    let e = if crate::transforms::pattern::tree_size_capped(arena, e, MAX_EXPAND_SIZE + 1)
        <= MAX_EXPAND_SIZE
        && expansion_cost_estimate(arena, e) <= MAX_EXPANSION_COST
    {
        let e = crate::transforms::expand::expand(arena, e);
        crate::transforms::eval::eval(arena, e)
    } else {
        e
    };

    let side = if right { "+" } else { "-" };
    tracing::debug!(side, expr = %arena.display(e).to_string(), "limit: one-sided Gruntz");

    let r = crate::calculus::gruntz::limit_pos_inf(arena, e, w)?;
    if is_valid_limit_value(arena, r, var) && !crate::base::walk::contains(arena, r, w) {
        with_imaginary_part(arena, e, w, r, depth)
    } else {
        Err(fail(format!(
            "could not determine the limit from the {side} side"
        )))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// (d) L'Hôpital + series fallback (analytic expressions only)
// ═══════════════════════════════════════════════════════════════════════════

fn analytic_fallback(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Result<ExprId, SymplexError> {
    if meets_branch_cut_off_axis(arena, expr, var, point) {
        return Err(fail(
            "a non-real argument meets a branch cut at the point: the one-sided limits may differ",
        ));
    }
    let (numer, denom) = crate::poly::polybridge::as_numer_denom(arena, expr);
    if denom != arena.one()
        && let Some(r) = try_lhopital(arena, numer, denom, var, point, 0)
        && is_valid_limit_value(arena, r, var)
    {
        return Ok(r);
    }

    Err(fail(
        "could not determine the limit via substitution, the Gruntz algorithm, or L'Hôpital's rule",
    ))
}

/// Does a function of `expr` with a branch cut (`ln`, a fractional power,
/// the inverse trigonometric and hyperbolic functions) have an argument
/// that lies on its cut at `var = point` without being real for real
/// `var`?  Then the function is not analytic at the point along the
/// approach, and [`analytic_fallback`]'s premise — that a one-sided limit
/// equals the two-sided one — fails: `x/(ln(−2 + i·x) − ln(−2))` is `2i`
/// from the right and `0` from the left (`ln` jumps by `2πi` across the
/// cut; SymPy: `limit(…, x, 0, '-')` → `0`), and L'Hôpital's rule gave
/// `2i` from both sides.  An argument that is real for real `var` stays
/// on one side of the cut (`ln(x − 3)` at `x = 1`).
fn meets_branch_cut_off_axis(arena: &mut Arena, expr: ExprId, var: ExprId, point: ExprId) -> bool {
    for id in crate::base::walk::post_order_ids(arena, expr) {
        let Some((arg, cut)) = branch_cut_of(arena, id) else {
            continue;
        };
        if !crate::base::walk::contains(arena, arg, var) || inner_known_real(arena, arg) {
            continue;
        }
        let at = crate::transforms::subs::subs(arena, arg, var, point);
        let at = crate::transforms::eval::eval(arena, at);
        if on_branch_cut(arena, cut, at) {
            return true;
        }
    }
    false
}

/// Try L'Hôpital's rule: lim f/g = lim f'/g' when f(a)=g(a)=0 or both →∞.
fn try_lhopital(
    arena: &mut Arena,
    numer: ExprId,
    denom: ExprId,
    var: ExprId,
    point: ExprId,
    depth: usize,
) -> Option<ExprId> {
    if depth >= MAX_LHOPITAL {
        return None;
    }

    // Evaluate numerator and denominator at the point *separately*
    // to avoid premature cancellation in Mul canonicalization.
    let n_subst = crate::transforms::subs::subs(arena, numer, var, point);
    let n_at_point = crate::transforms::eval::eval(arena, n_subst);
    let d_subst = crate::transforms::subs::subs(arena, denom, var, point);
    let d_at_point = crate::transforms::eval::eval(arena, d_subst);

    let n_is_zero = arena.is_zero_structural(n_at_point);
    let d_is_zero = arena.is_zero_structural(d_at_point);

    let n_is_inf = n_at_point == arena.infinity() || n_at_point == arena.neg_infinity();
    let d_is_inf = d_at_point == arena.infinity() || d_at_point == arena.neg_infinity();

    // Only apply L'Hôpital to genuine indeterminate forms: 0/0 or ∞/∞.
    if !((n_is_zero && d_is_zero) || (n_is_inf && d_is_inf)) {
        return None;
    }

    tracing::debug!(depth, "L'Hôpital: indeterminate form detected");

    let n_prime = crate::transforms::diff::diff(arena, numer, var);
    let d_prime = crate::transforms::diff::diff(arena, denom, var);

    // Evaluate the differentiated num/denom SEPARATELY first so we can
    // detect another 0/0 indeterminate form.
    let np_subst = crate::transforms::subs::subs(arena, n_prime, var, point);
    let np_val = crate::transforms::eval::eval(arena, np_subst);
    let dp_subst = crate::transforms::subs::subs(arena, d_prime, var, point);
    let dp_val = crate::transforms::eval::eval(arena, dp_subst);

    let np_zero = arena.is_zero_structural(np_val);
    let dp_zero = arena.is_zero_structural(dp_val);

    if np_zero && dp_zero {
        return try_lhopital(arena, n_prime, d_prime, var, point, depth + 1);
    }

    if is_finite_constant(arena, np_val, var) && is_finite_constant(arena, dp_val, var) && !dp_zero
    {
        let result_ratio = arena.div(np_val, dp_val);
        let result = crate::transforms::eval::eval(arena, result_ratio);
        if is_finite_constant(arena, result, var) {
            return Some(result);
        }
    }

    // Combined ratio as a fallback.
    let ratio = arena.div(n_prime, d_prime);
    let subst = crate::transforms::subs::subs(arena, ratio, var, point);
    let evaled = crate::transforms::eval::eval(arena, subst);
    if is_finite_constant(arena, evaled, var) {
        return Some(evaled);
    }

    try_lhopital(arena, n_prime, d_prime, var, point, depth + 1)
}

/// A valid, finite limit value (excludes every infinity).
fn is_finite_constant(arena: &Arena, id: ExprId, var: ExprId) -> bool {
    is_finite_limit_value(arena, id, var)
}

// ═══════════════════════════════════════════════════════════════════════════
// Limits at ±∞
// ═══════════════════════════════════════════════════════════════════════════

fn limit_at_infinity_dir(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    positive: bool,
    depth: usize,
) -> Result<LimVal, SymplexError> {
    let point = if positive {
        arena.infinity()
    } else {
        arena.neg_infinity()
    };

    // ── 1^∞ heuristic for Pow expressions ──────────────────────
    //
    // When expr = base^exp and the exponent depends on var, try the
    // rewrite  b^e → exp(e · (b − 1)).  This is mathematically exact
    // in the limit when b → 1:
    //
    //   b^e = exp(e·ln(b)) = exp(e·ln(1 + (b−1))) ≈ exp(e·(b−1))
    //
    // If the inner product e·(b−1) converges to a finite value L,
    // the answer is exp(L).  If it diverges or fails, we fall through
    // to the normal Gruntz path.
    if let ExprNode::Pow(base, exponent) = arena.node(expr).clone()
        && crate::base::walk::contains(arena, exponent, var)
    {
        let base_tends_to_one = if !crate::base::walk::contains(arena, base, var) {
            base == arena.one()
        } else {
            match crate::calculus::gruntz::gruntz(arena, base, var, point) {
                Ok(lim_base) => lim_base == arena.one(),
                Err(_) => false,
            }
        };

        if base_tends_to_one {
            tracing::debug!("limit: base → 1 confirmed, applying 1^∞ heuristic");
            let one = arena.one();
            let base_minus_1 = arena.sub(base, one);
            let product = arena.mul(&[exponent, base_minus_1]);
            if let Ok(inner_lim) = crate::calculus::gruntz::gruntz(arena, product, var, point)
                && is_finite_constant(arena, inner_lim, var)
            {
                let result = arena.exp(inner_lim);
                let result = crate::transforms::eval::eval(arena, result);
                if is_finite_limit_value(arena, result, var) {
                    return Ok(LimVal::exact(result));
                }
            }
        }
    }

    // ── Compositional rule ──
    if depth < MAX_COMPOSE_DEPTH
        && let Some(r) = try_compose(arena, expr, var, point, Direction::Both, depth)
    {
        return r;
    }

    // ── Gruntz ──
    // Its `±∞` comes from the leading term alone: the imaginary part is
    // completed by `with_imaginary_part` (in `x` declared positive, or
    // `−x` for `−∞`).
    tracing::debug!("limit: at infinity, trying Gruntz algorithm");
    let t = arena.positive_symbol("__lim_t");
    let along = if positive { t } else { arena.neg(t) };
    let gruntz_err = match crate::calculus::gruntz::gruntz(arena, expr, var, point) {
        Ok(result) if is_valid_limit_value(arena, result, var) => {
            let e = crate::transforms::subs::subs(arena, expr, var, along);
            let e = crate::transforms::eval::eval(arena, e);
            return with_imaginary_part(arena, e, t, result, depth);
        }
        Ok(_) => fail("Gruntz algorithm produced an indeterminate result"),
        Err(e) => e,
    };

    // ── Polynomial-degree fallback ──
    tracing::debug!("limit: Gruntz failed, falling back to polynomial degree analysis");
    match limit_at_infinity(arena, expr, var, positive) {
        Ok(r) if is_valid_limit_value(arena, r, var) => {
            let e = crate::transforms::subs::subs(arena, expr, var, along);
            let e = crate::transforms::eval::eval(arena, e);
            with_imaginary_part(arena, e, t, r, depth).map_err(|_| gruntz_err)
        }
        _ => Err(gruntz_err),
    }
}

/// The limit `value` of `e` as the positive variable `t → ∞`, found by the
/// Gruntz algorithm, with the imaginary part of an infinite one: Gruntz
/// decides `±∞` from the leading term `c₀·ω^{e₀}` alone (`c₀` real), and
/// `limit(ln(−x) + x, x, ∞)` was `oo` (it is `oo + iπ`).  `±oo + i·b` when
/// `Im e → b`, a weak `±oo` when `Im e` diverges (more slowly than `Re e`,
/// since `c₀` is real), `Err` when it is not determined.  A finite value,
/// and an eventually real `e`, pass unchanged.
fn with_imaginary_part(
    arena: &mut Arena,
    e: ExprId,
    t: ExprId,
    value: ExprId,
    depth: usize,
) -> Result<LimVal, SymplexError> {
    if value != arena.infinity() && value != arena.neg_infinity() {
        return Ok(LimVal::exact(tidy_phases(arena, value)));
    }
    if crate::calculus::gruntz::eventually_real_at_inf(arena, e, t) {
        return Ok(LimVal::exact(value));
    }
    let undetermined = || fail("the imaginary part of an infinite limit could not be determined");
    if depth >= MAX_COMPOSE_DEPTH {
        return Err(undetermined());
    }
    // Real arguments on a cut: their principal values written off the cut.
    let Some(e) = crate::calculus::gruntz::continue_real_on_cut_at_inf(arena, e, t) else {
        return Err(undetermined());
    };
    if crate::calculus::gruntz::eventually_real_at_inf(arena, e, t) {
        return Ok(LimVal::exact(value));
    }
    // `Im e` by the exact decomposition, by reflection, or by the
    // decomposition of `e` with the inverse functions written as
    // logarithms; a candidate much larger than `e` (the polar forms of
    // nested roots) is not tried.
    let decomposed = |arena: &mut Arena, e: ExprId| {
        let p = crate::base::complex::decompose(arena, e);
        p.exact.then(|| crate::transforms::eval::eval(arena, p.im))
    };
    let size = crate::transforms::pattern::tree_size_capped(arena, e, MAX_EXPAND_SIZE + 1);
    let cap = (4 * size).clamp(64, MAX_EXPAND_SIZE);
    let mut tried: Vec<ExprId> = Vec::with_capacity(3);
    for method in 0..3 {
        let im = match method {
            0 => decomposed(arena, e),
            1 => imaginary_part_by_reflection(arena, e, t),
            _ => {
                let r = inverse_functions_as_logs(arena, e);
                if r == e { None } else { decomposed(arena, r) }
            }
        };
        let Some(im) = im else {
            continue;
        };
        if tried.contains(&im)
            || crate::transforms::pattern::tree_size_capped(arena, im, cap + 1) > cap
        {
            continue;
        }
        tried.push(im);
        if arena.is_zero_structural(im) {
            return Ok(LimVal::exact(value));
        }
        // `Im e` is real: the Gruntz algorithm alone decides its limit
        // (finite or `±∞`), within its own work budget.
        let b = crate::calculus::gruntz::limit_pos_inf(arena, im, t);
        tracing::debug!(
            method,
            im = %arena.display(im).to_string(),
            ok = b.is_ok(),
            "limit: imaginary part of an infinite limit"
        );
        let Ok(b) = b else {
            continue;
        };
        if !is_valid_limit_value(arena, b, t) {
            continue;
        }
        match kind_of(arena, LimVal::exact(b)) {
            Some(Kind::Finite(b)) => {
                // A real constant written with complex logarithms
                // (`(ln(4i − 2) − ln(−4i − 2))/(2i)`): its real part.
                let b = match const_parts(arena, b) {
                    Some((re, im))
                        if crate::base::walk::contains(arena, b, arena.i_unit())
                            && is_zero_value(arena, im)
                            && !crate::base::walk::contains(arena, re, arena.i_unit()) =>
                    {
                        re
                    }
                    _ => b,
                };
                let i = arena.i_unit();
                let ib = arena.mul(&[i, b]);
                let v = arena.add(&[value, ib]);
                let v = crate::transforms::eval::eval(arena, v);
                if is_valid_limit_value(arena, v, t) {
                    return Ok(LimVal::exact(v));
                }
            }
            Some(Kind::Infinite(i)) if matches!(i.axis, Axis::RealPos | Axis::RealNeg) => {
                return Ok(LimVal { value, weak: true });
            }
            _ => {}
        }
    }
    Err(undetermined())
}

/// A finite value from the Gruntz algorithm with the phases `cos b + i·sin b`
/// of exponentials of non-real arguments written as `e^{ib}` again when that
/// is shorter (`coth(x + i) → 1`, not `(cos 2 + i·sin 2)/(cos 2 + i·sin 2)`).
fn tidy_phases(arena: &mut Arena, v: ExprId) -> ExprId {
    if !crate::base::walk::contains(arena, v, arena.i_unit()) {
        return v;
    }
    let trig = crate::base::walk::post_order_ids(arena, v)
        .into_iter()
        .any(|id| matches!(arena.node(id), ExprNode::Sin(_) | ExprNode::Cos(_)));
    if !trig {
        return v;
    }
    let r = crate::simplify::rewrite::rewrite_as_exp(arena, v);
    let r = crate::poly::polybridge::together(arena, r);
    let r = crate::transforms::eval::eval(arena, r);
    let size = |arena: &Arena, e: ExprId| {
        crate::transforms::pattern::tree_size_capped(arena, e, MAX_EXPAND_SIZE + 1)
    };
    if size(arena, r) < size(arena, v) && !contains_singular_atom(arena, r) {
        r
    } else {
        v
    }
}

/// `e` with the inverse trigonometric and hyperbolic functions written as
/// logarithms (SymPy's `rewrite(log)`, the principal values on all of ℂ
/// off the cuts): `asin z = −i·ln(iz + √(1 − z²))`, `acos z = π/2 − asin z`,
/// `atan z = (i/2)(ln(1 − iz) − ln(1 + iz))`, `asinh z = ln(z + √(z² + 1))`,
/// `acosh z = ln(z + √(z + 1)·√(z − 1))`, `atanh z = (ln(1 + z) − ln(1 −
/// z))/2` — forms the complex decomposition sees through.
fn inverse_functions_as_logs(arena: &mut Arena, e: ExprId) -> ExprId {
    let post = crate::base::walk::post_order_ids(arena, e);
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();
    let mut changed = false;
    for &id in &post {
        let r = crate::base::walk::rebuild_with_cache(arena, id, &cache);
        let (i, one, two) = (arena.i_unit(), arena.one(), arena.int(2));
        let half = arena.rational(1, 2);
        let asin = |arena: &mut Arena, z: ExprId| {
            let iz = arena.mul(&[i, z]);
            let z2 = arena.pow(z, two);
            let d = arena.sub(one, z2);
            let root = arena.sqrt(d);
            let s = arena.add(&[iz, root]);
            let l = arena.ln(s);
            let ni = arena.neg(i);
            arena.mul(&[ni, l])
        };
        let new = match arena.node(r).clone() {
            ExprNode::Asin(z) => asin(arena, z),
            ExprNode::Acos(z) => {
                let a = asin(arena, z);
                let pi = arena.pi();
                let hp = arena.mul(&[half, pi]);
                arena.sub(hp, a)
            }
            ExprNode::Atan(z) => {
                let iz = arena.mul(&[i, z]);
                let m = arena.sub(one, iz);
                let p = arena.add(&[one, iz]);
                let lm = arena.ln(m);
                let lp = arena.ln(p);
                let d = arena.sub(lm, lp);
                arena.mul(&[half, i, d])
            }
            ExprNode::Asinh(z) => {
                let z2 = arena.pow(z, two);
                let s = arena.add(&[z2, one]);
                let root = arena.sqrt(s);
                let a = arena.add(&[z, root]);
                arena.ln(a)
            }
            ExprNode::Acosh(z) => {
                let zp = arena.add(&[z, one]);
                let zm = arena.sub(z, one);
                let rp = arena.sqrt(zp);
                let rm = arena.sqrt(zm);
                let root = arena.mul(&[rp, rm]);
                let a = arena.add(&[z, root]);
                arena.ln(a)
            }
            ExprNode::Atanh(z) => {
                let p = arena.add(&[one, z]);
                let m = arena.sub(one, z);
                let lp = arena.ln(p);
                let lm = arena.ln(m);
                let d = arena.sub(lp, lm);
                arena.mul(&[half, d])
            }
            _ => r,
        };
        changed |= new != r;
        cache.insert(id, new);
    }
    if !changed {
        return e;
    }
    let r = cache.get(&e).copied().unwrap_or(e);
    crate::transforms::eval::eval(arena, r)
}

/// `Im e` for real `t → ∞`, as `(e − ē)/(2i)` with the reflection `ē` (`i`
/// replaced by `−i`), which is the conjugate of `e` where every function
/// with a branch cut has its argument off the cut (real arguments on a cut
/// were written by their principal value off it, `ln u = ln(−u) + iπ`):
/// every argument must be eventually real off the cut or have an eventually
/// non-zero component perpendicular to it.  The other symbols must be real.
/// `None` when the reflection is not known to be the conjugate.
fn imaginary_part_by_reflection(arena: &mut Arena, e: ExprId, t: ExprId) -> Option<ExprId> {
    use crate::base::assumptions::{AssumptionCache, Props};
    let mut cache = AssumptionCache::new();
    for s in crate::base::walk::free_symbols(arena, e) {
        if s != t && cache.query(arena, s, Props::REAL) != Some(true) {
            return None;
        }
    }
    for id in crate::base::walk::post_order_ids(arena, e) {
        let Some((arg, cut)) = branch_cut_of(arena, id) else {
            continue;
        };
        if !reflection_is_conjugate(arena, arg, cut, t) {
            return None;
        }
    }
    let i = arena.i_unit();
    let ni = arena.neg(i);
    let reflected = crate::transforms::subs::subs(arena, e, i, ni);
    let diff = arena.sub(e, reflected);
    let two = arena.int(2);
    let two_i = arena.mul(&[two, i]);
    let im = arena.div(diff, two_i);
    let im = crate::transforms::eval::eval(arena, im);
    // Over a common denominator the leading terms cancel exactly
    // (`e^t/(a + iπ) − e^t/(a − iπ)`), which the Gruntz algorithm cannot
    // always see.
    let combined = crate::poly::polybridge::together(arena, im);
    let combined = crate::transforms::eval::eval(arena, combined);
    let size = |arena: &Arena, e: ExprId| {
        crate::transforms::pattern::tree_size_capped(arena, e, MAX_EXPAND_SIZE + 1)
    };
    Some(if size(arena, combined) <= size(arena, im) + 16 {
        combined
    } else {
        im
    })
}

/// Does a function with the cut `cut` take its argument `u` off the cut
/// for all large `t` (so that it commutes with conjugation there)?
fn reflection_is_conjugate(arena: &mut Arena, u: ExprId, cut: BranchCut, t: ExprId) -> bool {
    if !crate::base::walk::contains(arena, u, t) {
        if crate::base::walk::free_symbols(arena, u).is_empty() {
            return !on_branch_cut(arena, cut, u);
        }
        let mut cache = crate::base::assumptions::AssumptionCache::new();
        return matches!(cut, BranchCut::None | BranchCut::NegativeReals)
            && cache.query(arena, u, crate::base::assumptions::Props::POSITIVE) == Some(true);
    }
    let sign = |arena: &mut Arena, e: ExprId| {
        let e = crate::transforms::eval::eval(arena, e);
        crate::calculus::gruntz::eventual_sign_at_inf(arena, e, t)
    };
    if crate::calculus::gruntz::eventually_real_at_inf(arena, u, t) {
        let one = arena.one();
        return match cut {
            BranchCut::None | BranchCut::ImaginaryAxis => true,
            BranchCut::NegativeReals => sign(arena, u) == Some(1),
            BranchCut::BelowOne => {
                let d = arena.sub(u, one);
                sign(arena, d) == Some(1)
            }
            BranchCut::AboveOne => {
                let d = arena.sub(one, u);
                sign(arena, d) == Some(1)
            }
            BranchCut::BeyondOne => {
                let two = arena.int(2);
                let u2 = arena.pow(u, two);
                let d = arena.sub(one, u2);
                sign(arena, d) == Some(1)
            }
            BranchCut::BelowMinusInvE => {
                let m1 = arena.neg_one();
                let inv_e = arena.exp(m1);
                let d = arena.add(&[u, inv_e]);
                sign(arena, d) == Some(1)
            }
        };
    }
    let p = crate::base::complex::decompose(arena, u);
    if !p.exact {
        return false;
    }
    let perpendicular = if cut == BranchCut::ImaginaryAxis {
        p.re
    } else {
        p.im
    };
    sign(arena, perpendicular).is_some_and(|s| s != 0)
}

/// Compute lim(x→∞) expr or lim(x→-∞) expr by polynomial-degree analysis.
///
/// Strategies:
/// 0. Direct polynomial degree comparison on the original expression
/// 1. Substitution x = 1/t, together+cancel to clear nested fractions, lim(t→0)
/// 2. Polynomial degree analysis on the substituted form
/// 3. L'Hôpital on the substituted form
pub(crate) fn limit_at_infinity(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    positive: bool, // true for +∞, false for -∞
) -> Result<ExprId, SymplexError> {
    // ── Strategy 0: Direct polynomial degree comparison on original expr ──
    let (orig_numer, orig_denom) = crate::poly::polybridge::as_numer_denom(arena, expr);
    if orig_denom != arena.one() {
        let n_deg = crate::poly::polybridge::poly_degree(arena, orig_numer, var);
        let d_deg = crate::poly::polybridge::poly_degree(arena, orig_denom, var);

        if let (Some(nd), Some(dd)) = (n_deg, d_deg) {
            if nd < dd {
                return Ok(arena.zero());
            }
            if nd == dd {
                let n_coeffs = crate::poly::polybridge::poly_coefficients(arena, orig_numer, var);
                let d_coeffs = crate::poly::polybridge::poly_coefficients(arena, orig_denom, var);
                if let (Some(nc), Some(dc)) = (n_coeffs, d_coeffs)
                    && let (Some(n_lead), Some(d_lead)) = (nc.last(), dc.last())
                {
                    let ratio = arena.div(*n_lead, *d_lead);
                    let result = crate::transforms::eval::eval(arena, ratio);
                    if is_finite_result(arena, result) {
                        return Ok(result);
                    }
                }
            }
            if nd > dd {
                let n_coeffs = crate::poly::polybridge::poly_coefficients(arena, orig_numer, var);
                let d_coeffs = crate::poly::polybridge::poly_coefficients(arena, orig_denom, var);
                if let (Some(nc), Some(dc)) = (n_coeffs, d_coeffs)
                    && let (Some(n_lead), Some(d_lead)) = (nc.last(), dc.last())
                    && let (Some(nr), Some(dr)) = (arena.as_num(*n_lead), arena.as_num(*d_lead))
                {
                    let ratio_positive = nr.is_positive() == dr.is_positive();
                    // For x → -∞, an odd degree difference flips the sign
                    let flip = !positive && ((nd - dd) % 2 == 1);
                    let result_positive = ratio_positive ^ flip;
                    if result_positive {
                        return Ok(arena.infinity());
                    } else {
                        return Ok(arena.neg_infinity());
                    }
                }
            }
        }
    } else {
        let deg = crate::poly::polybridge::poly_degree(arena, expr, var);
        if let Some(d) = deg
            && d == 0
        {
            let result = crate::transforms::eval::eval(arena, expr);
            if is_finite_result(arena, result) {
                return Ok(result);
            }
        }
    }

    // ── Strategy 1: Substitution x = 1/t, then together+cancel ──
    // x = ±1/t with t → 0⁺: t is positive.
    let t = arena.positive_symbol("__limit_t");
    let one = arena.one();
    let t_inv = arena.div(one, t);

    let sub_expr = if positive {
        arena.subs_structural(expr, var, t_inv)
    } else {
        let neg_t_inv = arena.neg(t_inv);
        arena.subs_structural(expr, var, neg_t_inv)
    };

    let together = crate::poly::polybridge::together(arena, sub_expr);
    let cancelled = crate::poly::polybridge::cancel(arena, together, t);
    let simplified = crate::transforms::eval::eval(arena, cancelled);
    let expanded = crate::transforms::expand::expand(arena, simplified);
    let evaled = crate::transforms::eval::eval(arena, expanded);

    // Only a *provably continuous* substitution `t = 0` is trusted: a raw
    // `subs` + `eval` folds forms such as `0·Γ(zoo)` into `0`.
    let zero = arena.zero();
    if let Some(at_zero) = try_direct_substitution(arena, evaled, t, zero)
        && is_finite_result(arena, at_zero)
    {
        return Ok(at_zero);
    }

    // ── Strategy 2: Polynomial degree analysis on the substituted form ──
    let (numer, denom) = crate::poly::polybridge::as_numer_denom(arena, evaled);
    if denom != arena.one() {
        let n_deg = crate::poly::polybridge::poly_degree(arena, numer, t);
        let d_deg = crate::poly::polybridge::poly_degree(arena, denom, t);

        if let (Some(nd), Some(dd)) = (n_deg, d_deg) {
            if nd < dd {
                return Ok(arena.zero());
            }
            if nd == dd {
                let cancelled = crate::poly::polybridge::cancel(arena, evaled, t);
                let result = arena.subs_structural(cancelled, t, zero);
                let result = crate::transforms::eval::eval(arena, result);
                if is_finite_result(arena, result) {
                    return Ok(result);
                }
            }
            if nd > dd {
                let n_coeffs = crate::poly::polybridge::poly_coefficients(arena, numer, t);
                let d_coeffs = crate::poly::polybridge::poly_coefficients(arena, denom, t);
                if let (Some(nc), Some(dc)) = (n_coeffs, d_coeffs)
                    && let (Some(n_lead), Some(d_lead)) = (nc.last(), dc.last())
                    && let (Some(nr), Some(dr)) = (arena.as_num(*n_lead), arena.as_num(*d_lead))
                {
                    let ratio_positive = nr.is_positive() == dr.is_positive();
                    if ratio_positive {
                        return Ok(arena.infinity());
                    } else {
                        return Ok(arena.neg_infinity());
                    }
                }
            }
        }
    }

    // Strategy 3: L'Hôpital on the substituted form (t → 0⁺ ≡ two-sided
    // for the analytic expressions this fallback is meant for).
    if has_directional_nodes(arena, evaled, t) {
        return Err(fail("could not compute limit at infinity"));
    }
    match analytic_fallback(arena, evaled, t, zero) {
        Ok(result) if is_valid_limit_value(arena, result, var) => Ok(result),
        _ => Err(fail("could not compute limit at infinity")),
    }
}

/// Check if a result is a finite number (not infinity, NaN, etc.)
fn is_finite_result(arena: &Arena, id: ExprId) -> bool {
    matches!(
        arena.node(id),
        ExprNode::Num(_) | ExprNode::Pi | ExprNode::E
    )
}

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
    fn limit_polynomial_direct_sub() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        // lim_{x→2} x^2 = 4
        let x2 = a.pow(x, two);
        let result = limit(&mut a, x2, x, two).unwrap();
        assert_eq!(display(&a, result), "4");
    }

    #[test]
    fn limit_sin_x_over_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let zero = a.zero;
        let sin_x = a.sin(x);
        let expr = a.div(sin_x, x);
        let result = limit(&mut a, expr, x, zero).unwrap();
        assert_eq!(display(&a, result), "1");
    }

    #[test]
    fn limit_constant() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let five = a.int(5);
        let zero = a.zero;
        let result = limit(&mut a, five, x, zero).unwrap();
        assert_eq!(display(&a, result), "5");
    }

    #[test]
    fn limit_symbolic_constant() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let c = sym(&mut a, "c");
        let zero = a.zero;
        let result = limit(&mut a, c, x, zero).unwrap();
        assert_eq!(result, c);
    }

    #[test]
    fn limit_x_squared_minus_one_over_x_minus_one() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.one;
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let numer = a.sub(x2, one);
        let denom = a.sub(x, one);
        let expr = a.div(numer, denom);
        let result = limit(&mut a, expr, x, one).unwrap();
        assert_eq!(display(&a, result), "2");
    }

    #[test]
    fn limit_direct_sub_works() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let three = a.int(3);
        let expr = a.add(&[x, a.one]);
        let result = limit(&mut a, expr, x, three).unwrap();
        assert_eq!(display(&a, result), "4");
    }

    #[test]
    fn one_sided_reciprocal() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let zero = a.zero;
        let one = a.one;
        let expr = a.div(one, x);
        let r = limit_dir(&mut a, expr, x, zero, Direction::Right).unwrap();
        assert_eq!(r, a.infinity());
        let l = limit_dir(&mut a, expr, x, zero, Direction::Left).unwrap();
        assert_eq!(l, a.neg_infinity());
        assert!(limit(&mut a, expr, x, zero).is_err());
    }

    #[test]
    fn two_sided_differ_error_message() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let zero = a.zero;
        let abs_x = a.abs(x);
        let expr = a.div(abs_x, x);
        let err = limit(&mut a, expr, x, zero).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("differ"), "unexpected message: {msg}");
    }

    #[test]
    fn direct_substitution_rejects_folded_indeterminate() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.one;
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let numer = a.sub(x2, one);
        let denom = a.sub(x, one);
        let expr = a.div(numer, denom);
        // subs folds (0)·(0)⁻¹ → 0; direct substitution must refuse it.
        assert!(try_direct_substitution(&mut a, expr, x, one).is_none());
    }

    #[test]
    fn var_must_be_symbol() {
        let mut a = Arena::new();
        let two = a.int(2);
        let zero = a.zero;
        assert!(limit(&mut a, two, two, zero).is_err());
    }
}

//! Arbitrary-precision numerical evaluation with complex number support.
//!
//! This module implements [`evalf`], which evaluates a symbolic expression
//! to a decimal string with a specified number of significant digits,
//! using the [`astro_float`] crate for arbitrary-precision arithmetic.
//!
//! # Design
//!
//! Evaluation is performed bottom-up using an explicit post-order
//! traversal (no recursion).  Each sub-expression is evaluated to a
//! [`Complex`] (a pair of [`BigFloat`] values for the real and imaginary
//! parts) and cached, so shared sub-expressions (common in a hash-consed
//! DAG) are only evaluated once.
//!
//! The working precision is set higher than the requested output
//! precision to absorb rounding errors from intermediate computations.
//! If a sub-expression cannot be evaluated (e.g., it contains free
//! symbols), the function returns an error.
//!
//! # Sums
//!
//! A finite `Sum` or `Product` of at most 10,000 terms is evaluated term by
//! term.  An infinite `Sum` of a rational term (or `(−1)^k` times one) is
//! summed directly up to a cut-off and its tail from the Laurent expansion
//! of the term and the Hurwitz zeta function by Euler–Maclaurin, with a
//! proven remainder (`emsum`); one of another hypergeometric term by its term
//! ratio, with a rigorous bound on the tail (`hypsum`, after SymPy's
//! `hypsum`); a smooth term that converges only polynomially (`ln k/k²`,
//! `1/(k² + √2)`, `(−1)^k ln(k + 1)/(k + 1)`, `H_k/k³`) by its expansion
//! at infinity with a Taylor-model remainder, summed against `Σ k^(−s)(ln
//! k)^m` by Euler–Maclaurin or Euler–Boole (`asymsum`; refused before 0.44).
//! A divergent sum is [`SymplexError::Divergent`]; any other term that is not
//! hypergeometric is refused.
//!
//! # Error bounds and branch cuts
//!
//! Every value carries an error bound on each of its parts (`accuracy`),
//! so an exactly real value is told apart from one whose imaginary part is
//! only within its error of 0; the side of a branch cut is taken from a
//! part outside its error ball, or by convention for an exact 0, and is
//! otherwise undecidable (more precision, then a refusal).  A sum or
//! product of complex-conjugate pairs is exactly real by its structure
//! (`conjugate`).
//!
//! # Definite integrals
//!
//! An unevaluated `DefiniteIntegral(body, var, lo, hi)` node is evaluated
//! at any precision by tanh-sinh quadrature over `BigFloat` (`quad`, after
//! mpmath's `quad`), the body evaluated at each node by this module's own
//! node evaluator with its error bounds, interior breakpoints split off and
//! infinite ranges mapped; its bound is the difference of the last two
//! levels plus the truncated tail and the rounding of the values (see
//! `quad` for what that certifies), and digits beyond it are refused
//! ([`SymplexError::PrecisionExhausted`]).  At the lowest working precision
//! the cheaper `f64` adaptive Gauss–Kronrod quadrature from
//! [`crate::calculus::definite`] is tried first and served when it resolves
//! the integral to [`QUADRATURE_F64_DIGITS`] digits (`Ex::eval_f64`'s
//! request; see `eval_definite_integral`).  Before 0.43 that was the only
//! route: more than 16 digits were refused with
//! [`SymplexError::NotImplemented`], and an endpoint singularity left its
//! bound wrong (`∫₀¹ dx/√(1 − x)` was `1.9999999898808887` with a bound of
//! `2·10⁻⁹`).  When the quadratures refuse, or the bounds lie so far out
//! next to the width that no node can be evaluated cheaply, the closed form
//! of the definite integrator is evaluated instead, when it has one (since
//! 0.52; `eval_definite_integral`).

use crate::base::bigfloat::{self, is_sign_positive, is_strictly_positive};
use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::Zero;
use rustc_hash::FxHashMap;

use astro_float::{BigFloat, Consts, Sign};
use num_complex::Complex64;

use crate::base::arena::Arena;
use crate::base::errors::SymplexError;
use crate::base::libfn::LibFn;
use crate::base::node::{ExprId, ExprNode};
use crate::base::walk;
use ball::{Balls, CBall, Cache};
use tracing::debug;
use wp::{Guard, Prec, RM, Recurred, Summed, Wp};

mod accuracy;
mod asymsum;
mod ball;
pub(crate) mod bernoulli;
mod bessel_debye;
mod bessel_order;
mod bessel_recur;
mod cbessel;
mod cerf;
mod cexpint;
mod conjugate;
mod cspecial;
mod decimal;
mod elliptic;
mod emsum;
mod exact;
mod extended;
mod factorials;
mod hypergeom;
mod hypsum;
mod improper;
mod lambertw;
mod loggamma;
mod polylog;
mod quad;
mod rszeta;
mod sensitivity;
mod wp;
mod zeta;

// ═══════════════════════════════════════════════════════════════════════════
// Complex type
// ═══════════════════════════════════════════════════════════════════════════

/// A complex number represented as (real_part, imaginary_part).
use crate::base::bigcomplex::Complex;

pub(crate) use decimal::Decimal;
pub(crate) use quad::nonzero_on;

mod airy;
mod bessel;
mod betainc;
mod carlson;
mod celem;
mod constants;
mod erf;
mod expint;
mod f64_special;
mod format;
mod gamma;
mod incgamma;
mod integral;
mod orthopoly;
mod polylog_eta;
mod special09;
mod subtree;
mod zeta_polygamma;
use airy::*;
use bessel::*;
use betainc::*;
use carlson::*;
use celem::*;
use constants::*;
use erf::*;
use expint::*;
use f64_special::*;
use format::*;
use gamma::*;
use incgamma::*;
pub(crate) use integral::*;
use orthopoly::*;
use polylog_eta::*;
use special09::*;
use subtree::*;
use zeta_polygamma::*;

// ═══════════════════════════════════════════════════════════════════════════
// Public entry point
// ═══════════════════════════════════════════════════════════════════════════

/// Evaluate `expr` numerically to `digits` decimal digits of precision.
///
/// Returns the decimal string representation of the result, or an error
/// if the expression contains free symbols, infinities, or NaN.
///
/// A number that is certainly not 0 but lies below the exponent range
/// (`erfc(10⁵)`, `exp(−10¹⁰)`, `2·besselk(0, 10¹⁰)`) is an
/// [`SymplexError::Unevaluable`] error that says so ([`underflow_error`]);
/// before 0.32 it was printed `0`.  Such numbers are held scaled where they
/// can be (`extended`): a sum of them of both signs is decided as any sum
/// (`exp(−10¹⁰) − exp(−2·10¹⁰)` is refused as nonzero, its logarithm is
/// `−10¹⁰`), and one that cannot be decided is refused
/// ([`undecided_error`]); before 0.34 both printed `0`.  A cancellation
/// that is zero to the precision reached at the scale of its terms is still
/// `0` (`exp(−10¹⁰)·(sin²1 + cos²1) − exp(−10¹⁰)`) — unless a term
/// certainly not 0 was lost below its error, or an exact input is not held
/// in full at that precision: then it is pursued to the precision that
/// shows what was lost, and refused beyond the configured maximum or below
/// the exponent range ([`Adaptive::Hidden`]; before 0.35 such values printed
/// `0`: `sin²1 + cos²1 − 1 + exp(−10¹⁰)`).
pub(crate) fn evalf(arena: &Arena, expr: ExprId, digits: u32) -> Result<String, SymplexError> {
    evalf_with(arena, expr, digits, ZeroSearch::Number)
}

/// [`evalf`] with the [`ZeroSearch`] `search` for a value that is zero to
/// the working precision.  A nonzero number below the exponent range
/// ([`Adaptive::Underflow`]) is refused ([`underflow_error`]): 0 is not one
/// of its digits.  (Up to 0.57 `eval_const_f64` printed it `0` here and
/// parsed the string; it reads the value directly since.)
fn evalf_with(
    arena: &Arena,
    expr: ExprId,
    digits: u32,
    search: ZeroSearch,
) -> Result<String, SymplexError> {
    // Decimal digits to binary precision with guard bits, zero digits and a
    // precision beyond the configured maximum refused.
    let prec = checked_precision(arena, digits, "eval_decimal")?;

    debug!(
        digits = digits,
        binary_precision = prec,
        "evalf: starting numerical evaluation"
    );

    // Free symbols make the whole expression unevaluable; report the
    // offending symbol by name up front instead of surfacing an internal
    // cache miss from some parent node later.  Bound variables (a `Sum`
    // index, a `RootOf` polynomial variable, …) are not free and are
    // handled by the sub-tree evaluators below.
    let free = walk::free_symbols(arena, expr);
    if !free.is_empty() {
        let mut names: Vec<&str> = free
            .iter()
            .filter_map(|&id| match arena.node(id) {
                ExprNode::Symbol(sid) => Some(arena.symbol_name(*sid)),
                _ => None,
            })
            .collect();
        names.sort_unstable();
        if let Some(name) = names.first() {
            debug!(symbol = name, "evalf: expression has free symbols");
            return Err(SymplexError::FreeSymbol {
                name: (*name).to_owned(),
            });
        }
    }

    let adaptive = with_eval_consts(digits, |cc| {
        evaluate_adaptive(arena, expr, digits, search, cc)
    })?;
    match adaptive {
        Adaptive::Settled(result, _) => format_complex(&result, digits),
        Adaptive::PartBelow { imaginary, .. } => Err(part_underflow_error(imaginary)),
        Adaptive::Underflow(_) | Adaptive::Beyond(_) => Err(underflow_error()),
        Adaptive::Undecided => Err(undecided_error()),
        Adaptive::Hidden { below_range: true } => Err(hidden_error()),
        Adaptive::Hidden { below_range: false } => Err(hidden_deep_error(digits)),
    }
}

/// [`evalf`] with every definite integral evaluated by the quadrature rule
/// the caller chose: Gauss–Legendre when `gauss_legendre`, tanh-sinh
/// otherwise (`Ex::integrate_numeric_decimal`).
pub(crate) fn evalf_quadrature(
    arena: &Arena,
    expr: ExprId,
    digits: u32,
    gauss_legendre: bool,
) -> Result<String, SymplexError> {
    let _rule = quad::RuleGuard::set(gauss_legendre);
    evalf(arena, expr, digits)
}

/// [`const_value`] with every definite integral evaluated by the quadrature
/// rule the caller chose (as [`evalf_quadrature`]).
pub(crate) fn const_value_quadrature(
    arena: &Arena,
    expr: ExprId,
    digits: u32,
    gauss_legendre: bool,
) -> Result<ConstValue, SymplexError> {
    let _rule = quad::RuleGuard::set(gauss_legendre);
    const_value(arena, expr, digits)
}

/// A bound on what the integrand `body` (in `var`) holds next to the finite
/// ends of a piece `[lo, hi]` of an `f64` quadrature every node of which
/// gave 0 (see `improper::held_at_ends`): 0 where it is an exact 0 there or
/// below the exponent range, `∞` when it cannot be read there.
pub(crate) fn f64_piece_ends_bound(
    arena: &Arena,
    body: ExprId,
    var: ExprId,
    lo: f64,
    hi: f64,
) -> f64 {
    with_eval_consts(F64_DIGITS, |cc| {
        Ok(improper::held_at_ends(arena, body, var, lo, hi, cc))
    })
    .map_or(f64::INFINITY, |held| {
        if accuracy::is_unknown(held) {
            f64::INFINITY
        } else {
            held.exp2()
        }
    })
}

/// Can an `f64` quadrature of `body` (in `var`) from `a` to `b` (`±∞`
/// allowed), which evaluates the integrand by `compiled`, have seen its
/// integral beyond `|x| = 2¹⁶` to within `allowed`?  Always for a finite
/// range; for an infinite one by certified evaluations of the integrand out
/// to `2²⁰⁴⁸` (see `improper::f64_tail_trusted`): an overflow in the
/// compiled body cuts the integrand off (`Ex::integrate_numeric` of
/// `|x|/(x² + 1)` over the line, which diverges, was `709.78` before
/// 0.47).  `false` when the integrand cannot be evaluated there.
pub(crate) fn f64_quadrature_tail_trusted(
    arena: &Arena,
    body: ExprId,
    var: ExprId,
    a: f64,
    b: f64,
    compiled: &dyn Fn(f64) -> f64,
    allowed: f64,
) -> bool {
    let (low, high) = (a.min(b), a.max(b));
    if low.is_finite() && high.is_finite() {
        return true;
    }
    with_eval_consts(F64_DIGITS, |cc| {
        Ok((high != f64::INFINITY
            || improper::f64_tail_trusted(arena, body, var, true, compiled, allowed, cc))
            && (low != f64::NEG_INFINITY
                || improper::f64_tail_trusted(arena, body, var, false, compiled, allowed, cc)))
    })
    .unwrap_or(false)
}

/// The bits behind `digits` decimal digits: `3.4` bits per digit
/// (`log₂ 10 ≈ 3.3219`) plus 64 guard bits, at least 128.  In `u64`, where
/// no digit count overflows (`(2³² − 1)·34 < 2³⁸`).
///
/// Up to 0.60 this was `(digits as usize) * 34 / 10 + 64`, written out in
/// five places: on a target whose `usize` has 32 bits (`wasm32`) the product
/// overflows from `digits = 126,322,568` on — a panic in a debug build, and
/// in a release build a wrapped precision (128 bits for that count) that
/// passed the check against the configured maximum.
const fn working_bits(digits: u32) -> u64 {
    let bits = digits as u64 * 34 / 10 + 64;
    if bits < 128 { 128 } else { bits }
}

/// [`working_bits`] of `digits` when an evaluation may run at it: at most
/// `max_bits` (the configured maximum) and at most an eighth of `usize_max`,
/// the largest `usize` of the target — the precision search doubles the
/// precision and adds guard bits in `usize`, which then cannot overflow
/// (on a 64-bit target the configured maximum, a `u32`, is the only limit).
fn precision_within(digits: u32, max_bits: u32, usize_max: u64) -> Option<u64> {
    let bits = working_bits(digits);
    (bits <= u64::from(max_bits) && bits <= usize_max / 8).then_some(bits)
}

/// The working precision behind `digits` decimal digits ([`working_bits`]).
/// The entry points have refused a count whose precision `usize` cannot
/// hold ([`checked_precision`]); it would saturate here.
fn working_precision(digits: u32) -> usize {
    usize::try_from(working_bits(digits)).unwrap_or(usize::MAX)
}

/// The working precision of an evaluation to `digits` digits, for the
/// entry points of the evaluator (`operation` names the one that reports
/// zero digits).
///
/// # Errors
///
/// * [`SymplexError::InvalidArgument`] for `digits == 0`: no digit is
///   certified, and up to 0.60 every digit of the 128-bit working value was
///   printed (`eval_decimal(0)` of `1/3` gave
///   `0.333333333333333333333333333333333333334`, 39 digits, the last one
///   not a digit of `1/3`).
/// * [`SymplexError::PrecisionExhausted`] when the precision exceeds the
///   configured maximum, or what the target's `usize` allows
///   ([`precision_within`]).
fn checked_precision(
    arena: &Arena,
    digits: u32,
    operation: &'static str,
) -> Result<usize, SymplexError> {
    if digits == 0 {
        return Err(SymplexError::invalid_argument(
            operation,
            "at least one digit must be requested",
        ));
    }
    let max_bits = arena.config.max_evalf_precision;
    let usize_max = u64::try_from(usize::MAX).unwrap_or(u64::MAX);
    precision_within(digits, max_bits, usize_max)
        .and_then(|bits| usize::try_from(bits).ok())
        .ok_or_else(|| SymplexError::PrecisionExhausted {
            requested: digits,
            achieved: u32::try_from((u64::from(max_bits) * 10 / 34).saturating_sub(6))
                .unwrap_or(u32::MAX),
        })
}

/// Evaluate `expr` once at `prec` bits: its value and the exponent of its
/// error bound (see [`accuracy`]).  A failed node other than the root is
/// skipped: parent nodes (`Sum`, `Product`, `Piecewise`) evaluate their own
/// sub-trees with substitution.
fn evaluate_once(
    arena: &Arena,
    expr: ExprId,
    post_order: &[ExprId],
    w: &mut Wp<'_>,
) -> Result<Evaluated, SymplexError> {
    evaluate_tree_full(arena, expr, post_order, None, false, w)
}

/// One evaluation of a whole expression: the value, its bound, and whether
/// some node lacked a value or a bound only for want of precision in its
/// arguments ([`precision_limited`]).
struct Evaluated {
    value: Complex,
    bound: accuracy::Bound,
    precision_limited: bool,
    /// The value is 0 with an underflow bound, and certainly not 0
    /// ([`accuracy::underflow_nonzero`]): what is known of it.
    underflowed_nonzero: Option<accuracy::Nonzero>,
    /// The value lies below the exponent range and is known scaled
    /// (`extended`): `value` is its placeholder.
    beyond: Option<extended::Scaled>,
    /// The parts of the value that are 0 within an underflow bound and
    /// certainly not 0 ([`accuracy::PartsBelow`]).
    parts_below: accuracy::PartsBelow,
    /// The value depends on one that only the scaled form of a complex
    /// route holds below the exponent range ([`held_by_complex_route`]).
    untracked_below: bool,
    /// The nonzero terms some sum lost below its error bound
    /// ([`accuracy::Absorbed`]).
    absorbed: Option<accuracy::Absorbed>,
    /// The bits that hold every exact value of the expression in full
    /// ([`input_bits`]).
    exact_bits: u64,
    /// The bits that hold the second-order parts of the exact values with a
    /// small part ([`second_order_bits`]).
    second_bits: u64,
}

/// Error bounds of nodes held apart from their values: quadrature's
/// integrand (`quad.rs`) keeps the two maps, read as [`Cache::Split`]; the
/// evaluator holds one map of balls ([`Balls`]).
type ErrMap = FxHashMap<ExprId, accuracy::Bound>;

/// A bound variable preset to a value with an error bound.
type Seed = (ExprId, CBall);

/// Evaluate `root` bottom-up over `post_order` (its post-order) at `prec`
/// bits: its value and error bound.  `seed` presets a bound variable (a
/// `Sum` index, a `RootSum` root).  Conditions (relations, `And`, …) have no
/// value and are skipped; `Piecewise` decides them from their operands.  A
/// node that fails is skipped when `strict` is false (a parent `Sum`,
/// `Piecewise`, … may not need it) and fails the evaluation when it is true.
#[allow(clippy::too_many_arguments)]
fn evaluate_tree(
    arena: &Arena,
    root: ExprId,
    post_order: &[ExprId],
    seed: Option<Seed>,
    strict: bool,
    w: &mut Wp<'_>,
) -> Result<CBall, SymplexError> {
    evaluate_tree_full(arena, root, post_order, seed, strict, w)
        .map(|ev| CBall::new(ev.value, ev.bound))
}

/// [`evaluate_tree`], and whether a node was [`precision_limited`].
///
/// A composite node of rationals and `i` takes its exact value
/// ([`exact`]); a node whose value overflows the exponent range, or is the
/// reciprocal of one that underflowed, fails with that reason
/// ([`range_error`]) rather than passing `±∞` up to a
/// `PrecisionExhausted` that no precision would cure.
#[allow(clippy::too_many_arguments)]
fn evaluate_tree_full(
    arena: &Arena,
    root: ExprId,
    post_order: &[ExprId],
    seed: Option<Seed>,
    strict: bool,
    w: &mut Wp<'_>,
) -> Result<Evaluated, SymplexError> {
    let prec = w.bits();
    let mut balls = Balls::default();
    let mut failed: FxHashMap<ExprId, SymplexError> = FxHashMap::default();
    let mut exact_values = exact::ExactMap::default();
    let mut limited = false;
    // The nodes that are, or depend on, a node without a value or a bound
    // for want of precision in its arguments ([`precision_limited`]).
    let mut tainted: rustc_hash::FxHashSet<ExprId> = rustc_hash::FxHashSet::default();
    let mut sigs = conjugate::Sigs::new();
    // The nodes whose value underflowed to 0 but is certainly not 0.
    let mut nonzero: FxHashMap<ExprId, accuracy::Nonzero> = FxHashMap::default();
    // The nodes known below the exponent range (`extended`).
    let mut exts = extended::ExtMap::default();
    // The nodes with a part that is 0 within an underflow bound and
    // certainly not 0 (`accuracy::parts_below`).
    let mut below_parts: FxHashMap<ExprId, accuracy::PartsBelow> = FxHashMap::default();
    // The nodes that are, or depend on, a value held below the range by a
    // complex route alone ([`held_by_complex_route`]).
    let mut untracked: rustc_hash::FxHashSet<ExprId> = rustc_hash::FxHashSet::default();
    // The nonzero terms the sums lost below their error bounds.
    let mut absorbed: Option<accuracy::Absorbed> = None;
    // The bits that hold every exact value in full (`input_bits`), and its
    // second-order part (`second_order_bits`).
    let mut exact_bits = 0u64;
    let mut second_bits = 0u64;
    if let Some((id, CBall { mid: value, err })) = seed {
        sigs.record_leaf(id, &value, err);
        balls.insert(id, CBall::new(value, err));
    }
    for &id in post_order {
        if balls.contains_key(&id) || (id != root && is_condition(arena.node(id))) {
            continue;
        }
        let cache = Cache::Balls(&balls);
        let exact_z = exact::exact_value(arena, id, &exact_values);
        match (&exact_z, arena.node(id)) {
            (Some(z), _) => {
                exact_bits = exact_bits.max(input_bits(&z.re)).max(input_bits(&z.im));
                second_bits = second_bits
                    .max(second_order_bits(&z.re, arena))
                    .max(second_order_bits(&z.im, arena));
            }
            (None, ExprNode::Num(n)) => {
                let q = arena.num(*n);
                exact_bits = exact_bits.max(input_bits(q));
                second_bits = second_bits.max(second_order_bits(q, arena));
            }
            _ => {}
        }
        // Whether the value overflowed the exponent range.
        let mut overflowed = false;
        let mut evaluated = eval_ball(arena, id, cache, w).and_then(|ball| {
            match range_error(arena, id, &ball.mid, cache) {
                Some(err) => {
                    overflowed = true;
                    Err(err)
                }
                None => Ok(ball),
            }
        });
        if evaluated.is_err()
            && (incomplete_gamma_beyond(arena, id, cache) || beta_beyond(arena, id, cache))
        {
            overflowed = true;
        }
        // A value outside the exponent range, or a function of one, held
        // scaled (`extended`).
        let mut extended_value = false;
        if exact_z.is_none() {
            let underflowed = matches!(&evaluated, Ok(b)
                if accuracy::mag(&b.mid).is_none() && accuracy::is_underflow(b.err.joint()));
            let edge = extended::Edge {
                underflowed,
                overflowed,
            };
            if extended::wanted(arena, id, edge, cache, &exts)
                && let Some(outcome) = extended::extend(
                    arena,
                    id,
                    edge,
                    cache,
                    &exts,
                    &exact_values,
                    w,
                    &mut absorbed,
                )
            {
                extended_value = true;
                evaluated = match outcome {
                    extended::Outcome::InRange(ball) => Ok(ball),
                    extended::Outcome::Beyond(s) => {
                        let placeholder = s.placeholder(prec);
                        exts.insert(id, extended::Ext::Beyond(s));
                        Ok(placeholder)
                    }
                    // No ordinary value: the node fails as before (with its
                    // own overflow, or the overflow of the argument it
                    // lacks), and only the nodes with a rule for it see it.
                    extended::Outcome::Above(s) => {
                        let err = match evaluated {
                            Err(e) => e,
                            Ok(_) => extended::overflow_error(&s),
                        };
                        exts.insert(id, extended::Ext::Above(s));
                        Err(err)
                    }
                    extended::Outcome::Near(p, d) => {
                        let ordinary = near_ordinary(evaluated, &p, &d, prec);
                        exts.insert(id, extended::Ext::Near(p, d));
                        Ok(ordinary)
                    }
                };
            }
        }
        let evaluated = match (exact_z, evaluated) {
            (Some(z), evaluated) => {
                let refined = match evaluated {
                    Ok(CBall { mid: value, err: e }) => exact::refine(&z, value, e, prec),
                    Err(_) => exact::to_value(&z, prec),
                };
                exact_values.insert(id, z);
                Ok(refined)
            }
            (None, evaluated) => evaluated,
        };
        let tainted_child = || {
            arena
                .node(id)
                .children()
                .iter()
                .any(|c| tainted.contains(c))
        };
        // A domain error of a function whose argument is ±∞ or 0 only
        // because an argument lacked precision (`loggamma` of `atanh(1 −
        // 2·10⁻⁵⁰)³`, infinite at 128 bits): no value yet, as in
        // `eval_ball` for an inexact zero.
        let evaluated = match evaluated {
            Err(e) if domain_error(&e) && tainted_child() => Ok(CBall::unknown(c_zero(prec))),
            other => other,
        };
        match evaluated {
            Ok(CBall {
                mid: mut value,
                err: mut e,
            }) => {
                if precision_limited(&evaluation_arguments(arena, id), &value, e, cache)
                    || (e.is_unknown() && tainted_child())
                {
                    limited = true;
                    tainted.insert(id);
                }
                // A sum or product of conjugate pairs and real terms is
                // exactly real (see `conjugate`); its computed imaginary
                // part is a rounding residue within its error of 0.
                if let ExprNode::Add(children) | ExprNode::Mul(children) = arena.node(id)
                    && !e.is_unknown()
                    && !accuracy::exact_zero(&value.1, e.im)
                    && accuracy::part_contains_zero(&value.1, e.im)
                    && sigs.conjugates_pair_up(children, cache)
                {
                    value.1 = BigFloat::new(prec);
                    e.im = accuracy::EXACT;
                }
                sigs.record(arena, id, &value, e, cache);
                // A sum known scaled recorded its lost terms itself
                // (`extended`); an exact one lost nothing.
                if let ExprNode::Add(children) = arena.node(id)
                    && !exact_values.contains_key(&id)
                    && !matches!(exts.get(&id), Some(extended::Ext::Beyond(_)))
                {
                    note_absorbed(children, e.joint(), cache, &exts, &nonzero, &mut absorbed);
                }
                if !exts.contains_key(&id) {
                    note_series_tail(arena.node(id), e.joint(), cache, &mut absorbed);
                    note_limit_tail(arena.node(id), e.joint(), cache, &mut absorbed);
                    note_cross_terms(
                        arena,
                        id,
                        &value,
                        e.joint(),
                        cache,
                        &exact_values,
                        &mut absorbed,
                    );
                }
                if !extended_value && !exact_values.contains_key(&id) {
                    note_flattened(arena.node(id), e.joint(), &exts, &nonzero, &mut absorbed);
                }
                if accuracy::mag(&value).is_none() && accuracy::is_underflow(e.joint()) {
                    let status = match exts.get(&id) {
                        Some(extended::Ext::Beyond(s)) => s.nonzero(),
                        _ => accuracy::underflow_nonzero(arena, id, cache, &nonzero),
                    };
                    if let Some(s) = status {
                        nonzero.insert(id, s);
                    }
                }
                // The parts certainly not 0 below the exponent range: only
                // where some value of this evaluation left the range.
                if !(nonzero.is_empty() && exts.is_empty() && below_parts.is_empty()) {
                    let scaled = match (exts.get(&id), arena.node(id)) {
                        (Some(x), _) => extended::parts_below(x, prec),
                        (None, ExprNode::Add(children)) if !exts.is_empty() => {
                            extended::sum_parts_below(children, cache, &exts, prec)
                        }
                        _ => None,
                    };
                    let parts = accuracy::parts_below(
                        arena,
                        id,
                        &value,
                        e,
                        cache,
                        scaled,
                        &nonzero,
                        &below_parts,
                    );
                    if !parts.is_none() {
                        below_parts.insert(id, parts);
                    }
                }
                let underflowed =
                    accuracy::mag(&value).is_none() && accuracy::is_underflow(e.joint());
                if (underflowed && held_by_complex_route(arena.node(id), cache))
                    || (!untracked.is_empty()
                        && arena
                            .node(id)
                            .children()
                            .iter()
                            .any(|c| untracked.contains(c)))
                {
                    untracked.insert(id);
                }
                balls.insert(id, CBall::new(value, e));
            }
            Err(e) => {
                let e = certified_infinity(arena.node(id), e, Cache::Balls(&balls));
                let e = root_cause(arena.node(id), e, &failed);
                // A value above the exponent range held scaled is refused at
                // the root only: its parent may bring it back.
                let above = matches!(exts.get(&id), Some(extended::Ext::Above(_)));
                if (strict && !above) || id == root {
                    return Err(e);
                }
                failed.insert(id, e);
            }
        }
    }
    let CBall { mid: value, err } = balls.remove(&root).ok_or_else(|| {
        SymplexError::NotImplemented("evalf: expression not found in cache".into())
    })?;
    let beyond = match exts.remove(&root) {
        Some(extended::Ext::Beyond(s)) => Some(s),
        _ => None,
    };
    if let Some(s) = &beyond {
        extended::note_hidden(s, prec, &mut absorbed);
    }
    Ok(Evaluated {
        value,
        bound: err,
        precision_limited: limited,
        underflowed_nonzero: nonzero.get(&root).copied(),
        beyond,
        parts_below: below_parts.get(&root).copied().unwrap_or_default(),
        untracked_below: untracked.contains(&root),
        absorbed,
        exact_bits,
        second_bits,
    })
}

/// The ordinary value of a node known as `p + d` (`extended::Ext::Near`,
/// `d` below the exponent range): its own when it is `p` within its bound
/// (`exp`, `cos` or a power of `1 + d` at the placeholder of `d`), otherwise
/// `p` within the underflow bound — a node without an ordinary value, or `ln`
/// of a value outside the range, `ln(e^(−10¹⁰) + e^(−2·10¹⁰)) = −10¹⁰ +
/// e^(−10¹⁰) + …`, whose ordinary value is that of `ln(0 ± underflow)`.
fn near_ordinary(
    evaluated: Result<CBall, SymplexError>,
    p: &exact::Gauss,
    d: &extended::Scaled,
    prec: usize,
) -> CBall {
    let CBall { mid: pv, err: pb } = exact::to_value(p, prec);
    if let Ok(CBall { mid: v, err: b }) = evaluated
        && !b.is_unknown()
        && accuracy::mag(&v).is_some()
    {
        let diff = c_sub(&v, &pv, prec + 8, RM);
        let within = |x: &BigFloat, e: accuracy::ErrExp, f: accuracy::ErrExp| {
            accuracy::part_contains_zero(x, accuracy::shift(accuracy::lsum(e, f), 1.0))
        };
        if within(&diff.0, b.re, pb.re) && within(&diff.1, b.im, pb.im) {
            return CBall::new(v, b);
        }
    }
    let u = accuracy::UNDERFLOW;
    let real = accuracy::exactly_real(&d.m, d.eb);
    let b = accuracy::Bound {
        re: accuracy::lsum(pb.re, u),
        im: if real {
            pb.im
        } else {
            accuracy::lsum(pb.im, u)
        },
    };
    CBall::new(pv, b)
}

/// Record in `slot` the terms of a sum (`children`, with the error bound
/// `2^radius`) that are certainly not 0 and smaller than that bound
/// ([`accuracy::Absorbed`]): a value in range whose ball excludes 0, one
/// known below the exponent range (`extended`), or one that underflowed but
/// is certainly not 0 (its magnitude below the underflow bound).
fn note_absorbed(
    children: &[ExprId],
    radius: accuracy::ErrExp,
    cache: Cache<'_>,
    exts: &extended::ExtMap,
    nonzero: &FxHashMap<ExprId, accuracy::Nonzero>,
    slot: &mut Option<accuracy::Absorbed>,
) {
    if !radius.is_finite() {
        return;
    }
    for c in children {
        let term = match exts.get(c) {
            Some(extended::Ext::Beyond(s)) => {
                if s.nonzero().is_none() {
                    continue;
                }
                s.lg_value()
            }
            _ => {
                let (Some(v), Some(b)) = (cache.get(c), cache.bound(c)) else {
                    continue;
                };
                if !b.is_unknown() && !accuracy::contains_zero(v, b.joint()) {
                    accuracy::lg_abs(v)
                } else if nonzero.contains_key(c) {
                    accuracy::UNDERFLOW
                } else {
                    continue;
                }
            }
        };
        accuracy::Absorbed::note(slot, term, radius);
    }
}

/// Record in `slot` the part below the exponent range, certainly not 0, of
/// an argument of `node` that the node's ordinary value (with the error
/// bound `2^radius`) does not hold ([`accuracy::Absorbed`]): an argument
/// known scaled below the range (`extended`), the part `d` of one that is
/// `p + d`, or one that underflowed but is certainly not 0, taken by a node
/// without a rule of its own for it — its argument is the placeholder `0 ±
/// underflow` or `p ± underflow`, so its value is `f(0)` or `f(p)`.  A
/// cancellation against `f(p)` leaves the lost part: before 0.36
/// `sin(exp(e^(−10¹⁰))) − sin 1` (truly `cos 1·e^(−10¹⁰)`), `besselj(0,
/// e^(−10¹⁰)) − 1`, `erfc(e^(−10¹⁰)) − 1`, `2^(e^(−10¹⁰)) − 1` and
/// `cos(besselk(0, 10¹⁰)) − 1` printed `0` (`besselj(0, T) − 1 = −T²/4 +
/// O(T⁴)` by SymPy's series, for `T = e^(−10¹⁰)` mpmath
/// `-exp(-mpf(10)**10)**2/4` = `-2.15e-8685889639`).
fn note_flattened(
    node: &ExprNode,
    radius: accuracy::ErrExp,
    exts: &extended::ExtMap,
    nonzero: &FxHashMap<ExprId, accuracy::Nonzero>,
    slot: &mut Option<accuracy::Absorbed>,
) {
    if !radius.is_finite() {
        return;
    }
    for c in node.children().iter() {
        let term = match exts.get(c) {
            Some(extended::Ext::Beyond(s)) => {
                if s.nonzero().is_none() {
                    continue;
                }
                s.lg_value()
            }
            Some(extended::Ext::Near(_, d)) => {
                if d.nonzero().is_none() {
                    continue;
                }
                d.lg_value()
            }
            None if nonzero.contains_key(c) => accuracy::UNDERFLOW,
            // (A node with a value above the range has none of its own.)
            Some(extended::Ext::Above(_)) | None => continue,
        };
        accuracy::Absorbed::note(slot, term, radius);
    }
}

/// Record in `slot` the first term of the Taylor series at 0 of an
/// elementary function at a small argument that its value (with the error
/// bound `2^radius`) cannot show ([`accuracy::Absorbed`]): `x³/3` or `x⁵/5`
/// of `atan x`, `x²/2` or `x⁴/24` of `cos x`, `x`, `x²/2` or `x³/6` of
/// `exp x`.  The value is right within its bound, but a cancellation against
/// the leading terms leaves that term: `atan(10⁻⁴⁰⁰)·(sin²1 + cos²1) −
/// 10⁻⁴⁰⁰` is `−10⁻¹²⁰⁰/3` (before 0.35 it printed `0`), and the same
/// `+ 10⁻¹²⁰⁰/3` is `2·10⁻²⁰⁰¹` (before 0.36 it printed `0`: only the
/// first term was recorded).  Only for an argument certainly not 0 below
/// `2⁻⁸`, where the series is dominated by its first terms.
fn note_series_tail(
    node: &ExprNode,
    radius: accuracy::ErrExp,
    cache: Cache<'_>,
    slot: &mut Option<accuracy::Absorbed>,
) {
    // `log₂` of the coefficients of the third and fifth powers of the odd
    // functions (the smallest of each group: a term taken smaller only sends
    // the search deeper).
    const THIRD: f64 = -1.584_962_500_721_156; // log₂(1/3)
    const SIXTH: f64 = -2.584_962_500_721_156; // log₂(1/6)
    const TWO_FIFTEENTHS: f64 = -2.906_890_595_608_519; // log₂(2/15)
    const ONE_120TH: f64 = -6.906_890_595_608_519; // log₂(1/120)
    let (c, terms): (ExprId, &[(f64, f64)]) = match *node {
        ExprNode::Sin(c) | ExprNode::Sinh(c) | ExprNode::Asin(c) | ExprNode::Asinh(c) => {
            (c, &[(SIXTH, 3.0), (ONE_120TH, 5.0)])
        }
        ExprNode::Tan(c) | ExprNode::Tanh(c) | ExprNode::Atan(c) | ExprNode::Atanh(c) => {
            (c, &[(THIRD, 3.0), (TWO_FIFTEENTHS, 5.0)])
        }
        // `2/(3√π)`, `2/(10√π)`.
        ExprNode::Erf(c) => (
            c,
            &[
                (-1.410_710_565_457_315_5, 3.0),
                (-3.147_675_390_659_634, 5.0),
            ],
        ),
        // `1/2`, `1/24`.
        ExprNode::Cos(c) | ExprNode::Cosh(c) => (c, &[(-1.0, 2.0), (-4.584_962_500_721_156, 4.0)]),
        ExprNode::Exp(c) => (c, &[(0.0, 1.0), (-1.0, 2.0), (SIXTH, 3.0)]),
        _ => return,
    };
    if !radius.is_finite() {
        return;
    }
    let (Some(x), Some(b)) = (cache.get(&c), cache.bound(&c)) else {
        return;
    };
    if b.is_unknown() || accuracy::contains_zero(x, b.joint()) {
        return;
    }
    let lx = accuracy::lg_abs(x);
    if lx.is_nan() || lx >= -8.0 {
        return;
    }
    // The largest term below the bound (the series decreases).
    if let Some(t) = terms
        .iter()
        .map(|&(lc, n)| lc + n * lx)
        .find(|&t| t < radius)
    {
        accuracy::Absorbed::note(slot, t, radius);
    }
}

/// Record in `slot` the distance of a function from the value it tends to
/// — its limit at infinity, or the leading term of its asymptotic expansion
/// — when its value (with the error bound `2^radius`) cannot show it
/// ([`accuracy::Absorbed`]): `1 − tanh x ≥ e^(−2x)`, `π/2 − atan x ≥
/// 1/(2x)`, `1 − erf x = erfc(−(−x)) ≥ e^(−x²)/(2x√π)` (`x ≥ 1`), `ζ(s) − 1 ≥
/// 2^(−s)`, `1 − η(s) ≥ 2^(−s−1)` (`s ≥ 2`), `Li_s(z) − z ≥
/// abs(z)²·2^(−s−1)` (`s ≥ 4`, `0 < abs(z) ≤ 1`), `abs(asinh x − sign(x)·ln(2·abs(x)))`
/// and `abs(acosh x − ln 2x) ≥ 1/(8x²)` (`abs(x) ≥ 2`) — lower bounds, at the end
/// of the argument's ball nearer to the limit.  The value is right within
/// its bound, but a cancellation against the limit leaves that distance:
/// before 0.37 `tanh(400) − 1` (truly `−7.34·10⁻³⁴⁸`), `erf(30) − 1`,
/// `atan(10⁴⁰⁰) − π/2`, `zeta(2000) − 1` and `polylog(2000, 1/2) − 1/2`
/// printed `0` (mpmath at 30 digits does too; SymPy's `N(…, maxn=2000)` finds
/// them).
fn note_limit_tail(
    node: &ExprNode,
    radius: accuracy::ErrExp,
    cache: Cache<'_>,
    slot: &mut Option<accuracy::Absorbed>,
) {
    if !radius.is_finite() {
        return;
    }
    // `log₂ abs(x)` of a real argument, and of the end of its ball nearer to
    // infinity (the tails fall with `abs(x)`).
    let real = |c: ExprId| -> Option<(f64, f64, bool)> {
        let (v, b) = (cache.get(&c)?, cache.bound(&c)?);
        if b.is_unknown() || !accuracy::exactly_real(v, b) || v.0.is_zero() {
            return None;
        }
        let lx = accuracy::lg_abs(v);
        let far = accuracy::lsum(lx, b.re + 1.0);
        (lx.is_finite() && far.is_finite()).then_some((lx, far, v.0.is_negative()))
    };
    let e2 = |l: f64| l.exp2();
    const LOG2_E: f64 = std::f64::consts::LOG2_E;
    let tail = || -> Option<f64> {
        Some(match *node {
            ExprNode::Tanh(c) => {
                let (lx, far, _) = real(c)?;
                (lx >= 0.0).then_some(())?;
                -2.0 * e2(far) * LOG2_E
            }
            ExprNode::Atan(c) => {
                let (lx, far, _) = real(c)?;
                (lx >= 0.0).then_some(())?;
                -1.0 - far
            }
            ExprNode::Erf(c) | ExprNode::Erfc(c) => {
                let (lx, far, negative) = real(c)?;
                (lx >= 0.0 && (negative || matches!(node, ExprNode::Erf(_)))).then_some(())?;
                // `log₂(2√π) = 1.8257…`.
                -e2(2.0 * far) * LOG2_E - 1.83 - far
            }
            ExprNode::Asinh(c) | ExprNode::Acosh(c) => {
                let (lx, far, negative) = real(c)?;
                (lx >= 1.0 && !(negative && matches!(node, ExprNode::Acosh(_)))).then_some(())?;
                -3.0 - 2.0 * far
            }
            ExprNode::Zeta(c) => {
                let (lx, far, negative) = real(c)?;
                (!negative && lx >= 1.0).then_some(())?;
                -e2(far)
            }
            ExprNode::Lib(f, ref args) => match (f, args.as_slice()) {
                (LibFn::DirichletEta, &[s]) => {
                    let (ls, far, negative) = real(s)?;
                    (!negative && ls >= 1.0).then_some(())?;
                    -e2(far) - 1.0
                }
                (LibFn::PolyLog, &[s, z]) => {
                    let (ls, _, negative) = real(s)?;
                    (!negative && ls >= 2.0).then_some(())?;
                    polylog_tail(
                        cache.get(&s)?,
                        cache.bound(&s)?,
                        cache.get(&z)?,
                        cache.bound(&z)?,
                        radius,
                    )
                }
                _ => return None,
            },
            _ => return None,
        })
    };
    // A tail beyond `f64` lies below the exponent range.
    if let Some(t) = tail()
        && !t.is_nan()
    {
        accuracy::Absorbed::note(slot, t.max(-1e300), radius);
    }
}

/// `log₂` of a lower bound on `abs(Li_s(z) − z)` over the balls of a real `s ≥
/// 4` and of `z` (`polylog::tail_lower_bound` at their ends nearer to the
/// limit: `s` raised, `abs(z)` lowered, `abs(ln z)` raised by the ball's relative
/// radius, at most `10⁻⁶`), or a size below the exponent range — a
/// cancellation against `z` refused — where none is proved and the leading
/// term `abs(z)²·2^(−s)` is not 64 bits above the value's error `2^radius` (where
/// it is, the value shows the distance and nothing is lost).  Before 0.37 the
/// bound was for a real `abs(z) ≤ 1` only; `polylog(1270, −3) + 3` (truly
/// `9·2⁻¹²⁷⁰`) and `polylog(3598, (3 + 4i)/5) − (3 + 4i)/5`, which the faster
/// large-order expansion of 0.37 evaluates, printed `0` in development.
fn polylog_tail(
    s: &Complex,
    bs: accuracy::Bound,
    z: &Complex,
    bz: accuracy::Bound,
    radius: accuracy::ErrExp,
) -> f64 {
    let (Ok(sf), Ok(x), Ok(y)) = (f64_of(&s.0), f64_of(&z.0), f64_of(&z.1)) else {
        return -1e300;
    };
    let estimate = 2.0 * x.hypot(y).log2() - sf - 1.0;
    match polylog_tail_proved(sf, s, bs, x, y, bz) {
        Some(t) => t,
        None if estimate - 64.0 >= radius => estimate - 64.0,
        None => -1e300,
    }
}

/// [`polylog_tail`]'s bound where it is proved.
fn polylog_tail_proved(
    sf: f64,
    s: &Complex,
    bs: accuracy::Bound,
    x: f64,
    y: f64,
    bz: accuracy::Bound,
) -> Option<f64> {
    const UNKNOWN_TAIL: f64 = -1e300;
    let rz = x.hypot(y);
    // Relative radii, at most 2^−20.
    let rel_s = if bs.is_exact() {
        0.0
    } else {
        bs.joint().exp2() / sf.max(1.0)
    };
    let rel_z = if bz.is_exact() {
        0.0
    } else {
        bz.joint().exp2() / rz
    };
    // (Float evidence: wide radii only withhold the proof.)
    if rel_s.is_nan() || rel_z.is_nan() || rel_s > 1e-6 || rel_z > 1e-6 {
        return None;
    }
    let s_up = sf * (1.0 + rel_s) + rel_s;
    let s_int = s.0.is_int() && bs.is_exact();
    let near_int = {
        let d = (sf - sf.round()).abs();
        d < 1.0 / 256.0 + rel_s * sf.max(1.0)
    };
    // `abs(z)` lowered by its ball (twice); the side of the unit circle must
    // be the center's.
    let shrink = 1.0 - 2.0 * rel_z;
    let grow = 2.0 * rel_z;
    let outside = rz > 1.0;
    if (outside && rz * shrink <= 1.0) || (!outside && rz * (1.0 + grow) > 1.0) {
        return None;
    }
    let near = near_int && !s_int;
    let t = polylog::tail_lower_bound(s_up, s_int, near, x * shrink, y * shrink);
    if t <= UNKNOWN_TAIL {
        return None;
    }
    if !outside {
        return Some(t);
    }
    // `abs(ln z)` grows with the argument's change: the conditions again at the
    // far end.
    let t2 = polylog::tail_lower_bound(
        s_up,
        s_int,
        near,
        x * (1.0 + grow),
        y * (1.0 + grow) + grow * rz,
    );
    (t2 > UNKNOWN_TAIL).then_some(t)
}

/// `log₂` of the ratio of the smallest term of the sum `id` that is
/// certainly not 0 to the sum's value, when the sum is certainly not 0 —
/// or, for an exact real rational `x` that is not a sum (the canonical form
/// folds `1 + 10⁻³⁰⁰` into one rational), the size of its small part
/// ([`rational_small_part`]).
fn small_part(arena: &Arena, id: ExprId, cache: Cache<'_>, exact: &exact::ExactMap) -> Option<f64> {
    let ExprNode::Add(children) = arena.node(id) else {
        // (A small part within 64 bits shows at any working precision.)
        return match exact::operand(arena, id, exact) {
            Some(z) if z.im.is_zero() && input_bits(&z.re) >= 64 => rational_small_part(&z.re),
            _ => None,
        };
    };
    let total = cache.get(&id)?;
    let tb = cache.bound(&id)?;
    if tb.is_unknown() || accuracy::contains_zero(total, tb.joint()) {
        return None;
    }
    let smallest = children
        .iter()
        .filter_map(|c| {
            let (v, b) = (cache.get(c)?, cache.bound(c)?);
            (!b.is_unknown() && !accuracy::contains_zero(v, b.joint())).then(|| accuracy::lg_abs(v))
        })
        .fold(f64::INFINITY, f64::min);
    let ratio = smallest - accuracy::lg_abs_low(total);
    ratio.is_finite().then_some(ratio)
}

/// `log₂` of `abs(n)` (within `2⁻⁵²` relative, or a bit beyond `f64`).
fn lg_int(n: &BigInt) -> f64 {
    use num_traits::{Signed, ToPrimitive};
    match n.abs().to_f64() {
        Some(f) if f.is_finite() && f > 0.0 => f.log2(),
        _ => n.bits() as f64,
    }
}

/// `log₂ abs(h/u)` for an integer `n = u + h` whose leading `k` binary
/// (`k ≤ 64`) or decimal (`k ≤ 8`) digits are `u` and whose rest `h ≠ 0` lies
/// at least 64 bits below the unit of the `k`-th digit: `10⁴⁰⁰ + 1` is `u =
/// 10⁴⁰⁰`, `h = 1`.  Integers below 128 bits have no such part.
fn integer_small_part(n: &BigInt) -> Option<f64> {
    use num_integer::Integer;
    use num_traits::Signed;
    let a = n.abs();
    let bits = a.bits();
    if bits < 128 {
        return None;
    }
    // `u = round(a/g)·g` for the unit `g` of the k-th digit.
    let split = |g: &BigInt| -> Option<Option<f64>> {
        let (quot, rem) = a.div_mod_floor(g);
        let up = &rem + &rem >= *g;
        let u = if up { (quot + 1u32) * g } else { quot * g };
        let h = &a - &u;
        if h.is_zero() {
            // `n` is short: no small part.
            return Some(None);
        }
        (lg_int(&h) < lg_int(g) - 64.0).then(|| Some(lg_int(&h) - lg_int(&u)))
    };
    for k in 1..=64u64 {
        let g = BigInt::from(1u32) << (bits - k);
        if let Some(found) = split(&g) {
            return found;
        }
    }
    let digits = (bits as f64 * std::f64::consts::LOG10_2).floor() as u32 + 1;
    for k in 1..=8u32 {
        let Some(e) = digits.checked_sub(k).filter(|&e| e > 0) else {
            break;
        };
        if let Some(found) = split(&BigInt::from(10u32).pow(e)) {
            return found;
        }
    }
    None
}

/// `log₂ abs(h/u)` for an exact rational `x = u + h` with a small part `h`
/// (the largest found): a small part of its numerator or denominator
/// ([`integer_small_part`]: `(10³⁰⁰ + 1)/10³⁰⁰`), or `u ≠ 0` the last
/// convergent of its continued fraction before a partial quotient `a ≥ 2¹⁶`
/// (among the first 64), so `abs(h) ≈ 1/(b²·a)` for `u = p/b` (within a
/// bit).  `1 + 10⁻³⁰⁰` is `1 + h`, `log₂ abs(h) = −996.6`; `10⁻³⁰⁰` itself
/// (`u = 0`) and `1234567/7654321` have none.
fn rational_small_part(x: &crate::base::numeric::Q) -> Option<f64> {
    use num_integer::Integer;
    if x.is_zero() {
        return None;
    }
    let digits = [x.numer(), x.denom()]
        .into_iter()
        .filter_map(integer_small_part)
        .fold(None, |acc: Option<f64>, r| {
            Some(acc.map_or(r, |a| a.max(r)))
        });
    let lg = lg_int;
    let (mut n, mut d) = (x.numer().clone(), x.denom().clone());
    // The convergents `pₖ/qₖ` and the ones before.
    let (mut p, mut q) = (BigInt::from(1), BigInt::from(0));
    let (mut p_prev, mut q_prev) = (BigInt::from(0), BigInt::from(1));
    for step in 0..64 {
        if d.is_zero() {
            return digits;
        }
        let (a, r) = n.div_mod_floor(&d);
        if step > 0 && a.bits() >= 16 && !p.is_zero() {
            let lg_u = lg(&p) - lg(&q);
            let lg_h = -(2.0 * lg(&q) + lg(&a));
            let cf = lg_h - lg_u;
            return Some(digits.map_or(cf, |d| d.max(cf)));
        }
        let next_p = &a * &p + &p_prev;
        let next_q = &a * &q + &q_prev;
        p_prev = std::mem::replace(&mut p, next_p);
        q_prev = std::mem::replace(&mut q, next_q);
        n = std::mem::replace(&mut d, r);
    }
    digits
}

/// Record in `slot` the second-order term that a power `(a + h)^q` or a
/// product `(a + h)·(b + k)` of sums with small parts loses below its error
/// bound `2^radius` ([`accuracy::Absorbed`]): `q(q − 1)/2·a^q·(h/a)²`, and
/// `ab·(h/a)·(k/b)`.  A cancellation of the leading and first-order terms
/// leaves it: `(√2 + 10⁻⁴⁰⁰)²·(sin²1 + cos²1) − 2 − 2√2·10⁻⁴⁰⁰` is
/// `10⁻⁸⁰⁰` (before 0.35 it printed `0`).  The same for an analytic
/// function at an argument with a small part, `f″(a)·h²/2` (taken as
/// `max(abs(f(a + h)), 1)·(h/a)²`), and for an exact rational `a + h` the
/// canonical form folded into one number ([`rational_small_part`]): before
/// 0.36 `sin(1 + 10⁻³⁰⁰) + sin(1 − 10⁻³⁰⁰) − 2·sin 1` (truly `−sin 1·10⁻⁶⁰⁰`),
/// `log(1 + 10⁻³⁰⁰) + log(1 − 10⁻³⁰⁰)` (`−10⁻⁶⁰⁰`) and `(1 + 10⁻⁴⁰⁰)^(1/2) − 1 −
/// 10⁻⁴⁰⁰/2` (`−10⁻⁸⁰⁰/8`) printed `0`: the inputs are held in full at
/// about 1,000 or 1,330 bits ([`input_bits`]), and the value needs twice as
/// many.
#[allow(clippy::too_many_arguments)]
fn note_cross_terms(
    arena: &Arena,
    id: ExprId,
    value: &Complex,
    radius: accuracy::ErrExp,
    cache: Cache<'_>,
    exact: &exact::ExactMap,
    slot: &mut Option<accuracy::Absorbed>,
) {
    if !radius.is_finite() {
        return;
    }
    let lv = accuracy::lg_abs(value);
    if !lv.is_finite() {
        return;
    }
    let lost = match arena.node(id) {
        ExprNode::Pow(b, e) => {
            let Some(q) = arena
                .as_num(*e)
                .and_then(crate::base::numeric::ratio_to_f64)
            else {
                return;
            };
            let Some(r) = small_part(arena, *b, cache, exact) else {
                return;
            };
            // (A guide for the depth of the search: a missed note is a
            // shallower search, never a value.)
            let c = (q * (q - 1.0) / 2.0).abs();
            if c == 0.0 || r > -8.0 {
                return;
            }
            lv + c.log2() + 2.0 * r
        }
        ExprNode::Mul(children) => {
            let mut parts: Vec<f64> = children
                .iter()
                .filter_map(|&c| small_part(arena, c, cache, exact))
                .filter(|&r| r <= -8.0)
                .collect();
            if parts.len() < 2 {
                return;
            }
            parts.sort_by(f64::total_cmp);
            lv + parts[parts.len() - 1] + parts[parts.len() - 2]
        }
        ExprNode::Sin(c)
        | ExprNode::Cos(c)
        | ExprNode::Tan(c)
        | ExprNode::Exp(c)
        | ExprNode::Ln(c)
        | ExprNode::Asin(c)
        | ExprNode::Acos(c)
        | ExprNode::Atan(c)
        | ExprNode::Sinh(c)
        | ExprNode::Cosh(c)
        | ExprNode::Tanh(c)
        | ExprNode::Asinh(c)
        | ExprNode::Acosh(c)
        | ExprNode::Atanh(c)
        | ExprNode::Gamma(c)
        | ExprNode::LogGamma(c)
        | ExprNode::Digamma(c)
        | ExprNode::Erf(c)
        | ExprNode::Erfc(c)
        | ExprNode::LambertW(c)
        | ExprNode::Si(c)
        | ExprNode::Ci(c)
        | ExprNode::Ei(c)
        | ExprNode::Li(c)
        | ExprNode::Zeta(c)
        | ExprNode::Factorial(c) => {
            let Some(r) = small_part(arena, *c, cache, exact) else {
                return;
            };
            if r > -8.0 {
                return;
            }
            lv.max(0.0) + 2.0 * r
        }
        _ => return,
    };
    accuracy::Absorbed::note(slot, lost, radius);
}

/// Is the node `node`, evaluated to `value ± e`, without a value or a bound
/// only because its arguments' error balls are too wide — its bound unknown
/// or its value not finite while every argument has a finite value with a
/// bound, at least one of them inexact?  Such a node needs more precision in
/// its arguments, not in itself: a quotient of two differences that are 0
/// at the working precision (`(e^ε − 1)/(e^{2ε} − 1)`, `ε = 10⁻⁴⁵`), a
/// function at an argument that cancelled to 0 or onto a pole
/// (`erfc⁻¹(sin(1 + 10⁻¹⁵⁰) − sin 1)`), the side of a branch cut decided
/// by a part within its error of 0, a condition at its threshold.  An
/// argument that underflowed is not: below the exponent range no precision
/// resolves it.  (See [`evaluate_adaptive`], which pursues such a value as
/// it pursues a root that is zero to the working precision.)
fn precision_limited(
    arguments: &[ExprId],
    value: &Complex,
    e: accuracy::Bound,
    cache: Cache<'_>,
) -> bool {
    let finite = |z: &Complex| !(z.0.is_inf() || z.0.is_nan() || z.1.is_inf() || z.1.is_nan());
    if !e.is_unknown() && finite(value) {
        return false;
    }
    let mut inexact = false;
    for &c in arguments {
        let (Some(v), Some(b)) = (cache.get(&c), cache.bound(&c)) else {
            continue;
        };
        if b.is_unknown() || !finite(v) || accuracy::is_underflow(b.joint()) {
            return false;
        }
        inexact |= !b.is_exact();
    }
    inexact
}

/// The arguments whose precision the value of node `id` depends on (see
/// [`precision_limited`]): its children, and for a `Piecewise` the
/// operands of the relations in its conditions, which are not evaluated as
/// nodes of their own.  So a condition at its threshold — decided only to
/// the working precision, the branch value then without a bound — is
/// pursued as the zero of a difference is.  Up to 0.41 it stopped at the
/// cap of the digits: `Piecewise((1, sin(13/5)² + cos(13/5)² ≥ 1 +
/// 10⁻²⁰⁰), (0, True))` was refused at 30 digits (it is 0).
fn evaluation_arguments(arena: &Arena, id: ExprId) -> smallvec::SmallVec<[ExprId; 6]> {
    let node = arena.node(id);
    let ExprNode::Piecewise(pairs) = node else {
        return node.children();
    };
    let mut out: smallvec::SmallVec<[ExprId; 6]> = smallvec::SmallVec::new();
    let mut stack: Vec<ExprId> = Vec::new();
    for &(value, cond) in pairs {
        out.push(value);
        stack.push(cond);
    }
    let mut seen: rustc_hash::FxHashSet<ExprId> = rustc_hash::FxHashSet::default();
    while let Some(c) = stack.pop() {
        if !seen.insert(c) {
            continue;
        }
        match arena.node(c) {
            ExprNode::And(ks) | ExprNode::Or(ks) => stack.extend(ks.iter().copied()),
            ExprNode::Not(k) => stack.push(*k),
            ExprNode::Gt(a, b) | ExprNode::Ge(a, b) | ExprNode::Eq_(a, b) | ExprNode::Ne(a, b) => {
                out.push(*a);
                out.push(*b);
            }
            _ => {}
        }
    }
    out
}

/// An incomplete gamma function `Γ(s, x)` or `γ(s, x)` of a shape
/// `s ≥ 2²⁰` whose ordinary evaluation failed: its value may lie beyond the
/// exponent range (`Γ(s)` leaves it from `s ≈ 7.8·10⁷` on; the series and
/// the fraction form `xˢe⁻ˣ`), where `extended` holds it scaled from
/// [`ln_incomplete_gamma`] — so that `uppergamma(s, x)/Γ(s)` is evaluated.
fn incomplete_gamma_beyond(arena: &Arena, id: ExprId, cache: Cache<'_>) -> bool {
    match arena.node(id) {
        ExprNode::Lib(LibFn::UpperGamma | LibFn::LowerGamma, args) if args.len() == 2 => cache
            .get(&args[0])
            .and_then(|v| f64_of(&v.0).ok())
            .is_some_and(|s| s >= 1_048_576.0),
        _ => false,
    }
}

/// A `B(a, b)` of positive real shapes on the `exp(ln B)` route
/// ([`BETA_VIA_LN_MIN`], [`beta_wp`]) whose ordinary evaluation failed: its
/// value lies beyond the exponent range (`B(10¹², 10¹²) ≈ 2^(−2·10¹²)`),
/// where `extended` holds it scaled from [`ln_beta_wp`] — so that
/// `ln(beta(10¹², 10¹²))` is evaluated (refused up to 0.57).
fn beta_beyond(arena: &Arena, id: ExprId, cache: Cache<'_>) -> bool {
    let ExprNode::Beta(a, b) = arena.node(id) else {
        return false;
    };
    let shape = |c: &ExprId| {
        cache
            .get(c)
            .filter(|v| v.1.is_zero() && bf_strictly_positive(&v.0))
            .and_then(|v| f64_of(&v.0).ok())
    };
    matches!((shape(a), shape(b)), (Some(x), Some(y)) if x + y >= BETA_VIA_LN_MIN)
}

/// The error of a node whose value `value` has an infinite part although
/// its arguments are finite: the value overflows the exponent range of
/// `BigFloat` (`Γ(10²⁰/3)`, `exp(10¹⁶)`, a product of such numbers), or is
/// the reciprocal of an argument that underflowed below it.  `None` when
/// the value is finite, or infinite for another reason (a division by an
/// argument that is only 0 to the working precision, which more precision
/// may resolve, see [`precision_limited`]).  Before 0.31 the infinite value
/// propagated to a `PrecisionExhausted`: `Γ(10²⁰/3)` "achieved 0 digits",
/// while `uppergamma(10²⁰/3, 10²⁰/3)` reported its overflow.
fn range_error(
    arena: &Arena,
    id: ExprId,
    value: &Complex,
    cache: Cache<'_>,
) -> Option<SymplexError> {
    if !(value.0.is_inf() || value.1.is_inf()) {
        return None;
    }
    let node = arena.node(id);
    let what = match node {
        ExprNode::Exp(_) => "exp",
        ExprNode::Pow(..) => "a power",
        ExprNode::Mul(_) => "a product",
        ExprNode::Add(_) => "a sum",
        ExprNode::Gamma(_) => "Gamma",
        ExprNode::Factorial(_) => "factorial",
        ExprNode::Binomial(..) => "binomial",
        ExprNode::Beta(..) => "beta",
        ExprNode::Sinh(_) => "sinh",
        ExprNode::Cosh(_) => "cosh",
        ExprNode::Ei(_) => "Ei",
        // The functions that grow like an exponential or a factorial, and
        // have no pole at a finite nonzero argument.
        ExprNode::Lib(f, _) => match f {
            LibFn::Erfi => "erfi",
            LibFn::Shi => "Shi",
            LibFn::Chi => "Chi",
            LibFn::BesselI => "besseli",
            LibFn::AiryBi => "airybi",
            LibFn::AiryBiPrime => "airybiprime",
            LibFn::RisingFactorial => "rising_factorial",
            LibFn::Subfactorial => "subfactorial",
            // `xˢe⁻ˣ` of the series or the fraction overflowed (since
            // 0.57: an infinity went on, and `uppergamma(s, x)/Γ(s)` was
            // `PrecisionExhausted` for `s ≥ 10⁸`, `x > s + 1`).
            LibFn::UpperGamma => "uppergamma",
            LibFn::LowerGamma => "lowergamma",
            _ => return None,
        },
        _ => return None,
    };
    let mut lg_sum = 0.0f64;
    for c in node.children() {
        let Some(v) = cache.get(&c) else {
            continue;
        };
        if v.0.is_inf() || v.0.is_nan() || v.1.is_inf() || v.1.is_nan() {
            return None;
        }
        if accuracy::mag(v).is_none() {
            let b = cache.err(&c);
            // An exact 0 is a parameter (`besseli(0, x)`); an inexact one
            // may be a division by a difference that cancelled, which more
            // precision resolves (`precision_limited`).
            if b.is_exact() {
                continue;
            }
            let underflow = accuracy::is_underflow(b.joint());
            return underflow.then(|| SymplexError::Unevaluable {
                reason: format!(
                    "{what}: an argument underflows the arbitrary-precision exponent range \
                     (below 2^{:e}), and its reciprocal overflows it",
                    f64::from(astro_float::EXPONENT_MIN)
                ),
            });
        }
        lg_sum += accuracy::lg_abs(v);
    }
    // The magnitude where it is cheap to estimate.
    let lg = match node {
        ExprNode::Exp(a) => cache
            .get(a)
            .and_then(|v| f64_of(&v.0).ok())
            .map(|x| x * std::f64::consts::LOG2_E),
        ExprNode::Gamma(a) | ExprNode::Factorial(a) => cache
            .get(a)
            .and_then(|v| f64_of(&v.0).ok())
            .filter(|x| *x > 1.0)
            .map(|x| {
                let x = if matches!(node, ExprNode::Factorial(_)) {
                    x + 1.0
                } else {
                    x
                };
                ((x - 0.5) * x.ln() - x) * std::f64::consts::LOG2_E
            }),
        ExprNode::Mul(_) => Some(lg_sum),
        _ => None,
    }
    .filter(|lg| lg.is_finite() && *lg > f64::from(astro_float::EXPONENT_MAX) / 2.0);
    Some(SymplexError::Unevaluable {
        reason: match lg {
            Some(lg) => format!(
                "{what}: the value (about 2^{lg:.3e}) overflows the arbitrary-precision exponent range"
            ),
            None => format!("{what}: the value overflows the arbitrary-precision exponent range"),
        },
    })
}

/// The error of node `node`, which failed with `e`: when `e` is only the
/// absence of a child's value (a child that failed and was skipped, see
/// [`evaluate_tree`]), the child's own error.  Before 0.30 the parent
/// reported the cache miss: `harmonic(7/2) + 1` was "sub-expression e22 not
/// in cache (likely contains free symbols)".
fn root_cause(
    node: &ExprNode,
    e: SymplexError,
    failed: &FxHashMap<ExprId, SymplexError>,
) -> SymplexError {
    if let SymplexError::Unevaluable { reason } = &e {
        for c in node.children() {
            if let Some(child_error) = failed.get(&c)
                && *reason == not_evaluated_reason(c)
            {
                return child_error.clone();
            }
        }
    }
    e
}

/// Is `node` a condition (a truth value, not a number)?
fn is_condition(node: &ExprNode) -> bool {
    matches!(
        node,
        ExprNode::BoolTrue
            | ExprNode::BoolFalse
            | ExprNode::Gt(_, _)
            | ExprNode::Ge(_, _)
            | ExprNode::Eq_(_, _)
            | ExprNode::Ne(_, _)
            | ExprNode::And(_)
            | ExprNode::Or(_)
            | ExprNode::Not(_)
    )
}

/// [`range_error`] over values and bounds held in two maps (quadrature's
/// integrand, `quad.rs`).
fn exponent_range_error(
    arena: &Arena,
    id: ExprId,
    value: &Complex,
    cache: &FxHashMap<ExprId, Complex>,
    errs: &ErrMap,
) -> Option<SymplexError> {
    range_error(arena, id, value, Cache::Split(cache, Some(errs)))
}

/// [`eval_ball`] over values and bounds held in two maps (quadrature's
/// integrand, `quad.rs`).
fn eval_node_with_error(
    arena: &Arena,
    id: ExprId,
    cache: &FxHashMap<ExprId, Complex>,
    errs: &ErrMap,
    w: &mut Wp<'_>,
) -> Result<CBall, SymplexError> {
    eval_ball(arena, id, Cache::Split(cache, Some(errs)), w)
}

/// Evaluate node `id` from its children's values and error bounds
/// (`cache`): its value and error bound.
///
/// Most nodes are evaluated by [`eval_node`] and bounded from their
/// children by [`accuracy::node_error`].  `+` and `·` are evaluated with
/// their per-part bounds in one pass ([`accuracy::add_with_bound`],
/// [`accuracy::mul_with_bound`]: the roundings of the partial results are
/// part of the bound).  The nodes whose evaluator knows more than the
/// children's values report their own bound: sums and products (their
/// terms' bounds; an infinite sum's tail), `RootOf` and `RootSum` (certified
/// inclusion disks of the roots), `Piecewise` (whether its conditions were
/// decided with certainty), physical constants (their value's bound) and
/// definite integrals (the quadrature's error estimate).
fn eval_ball(
    arena: &Arena,
    id: ExprId,
    cache: Cache<'_>,
    w: &mut Wp<'_>,
) -> Result<CBall, SymplexError> {
    let prec = w.bits();
    // What a route charges to its value is this node's (`wp.rs`, "A route's
    // own error").
    w.clear_own();
    // `erfinv`, `erfcinv` next to the ends of their domains, from the exact
    // complement of the argument (`cspecial.rs`; refused before 0.45).
    if let Some(r) = cspecial::inverse_erf_end(arena, arena.node(id), cache, w) {
        let CBall { mid: value, err } = r?;
        let err = accuracy::reported(&value, err, prec);
        return Ok(CBall::new(value, err));
    }
    let CBall { mid: value, err } = match arena.node(id) {
        ExprNode::Add(children) | ExprNode::Mul(children) => {
            let mut parts: Vec<(&Complex, accuracy::Bound)> = Vec::with_capacity(children.len());
            // `q·erf(u) + q·erfc(u)` is `q` exactly ([`erf_complements`]).
            let folded = match arena.node(id) {
                ExprNode::Add(_) => erf_complements(arena, &children[..], prec),
                _ => None,
            };
            for (i, &c) in children.iter().enumerate() {
                // (A paired term must have a value too: the identity holds
                // where both sides are defined.)
                let v = get_cached(cache, c)?;
                if folded.as_ref().is_some_and(|f| f.paired[i]) {
                    continue;
                }
                parts.push((v, cache.err(&c)));
            }
            if let Some(f) = &folded {
                parts.push((&f.constant.mid, f.constant.err));
            }
            return Ok(if matches!(arena.node(id), ExprNode::Add(_)) {
                accuracy::add_with_bound(&parts, prec)
            } else {
                accuracy::mul_with_bound(&parts, prec)
            });
        }
        ExprNode::Sum(body, var, lo, hi) => eval_sum(arena, *body, *var, *lo, *hi, cache, w)?,
        ExprNode::Product_(body, var, lo, hi) => {
            eval_product(arena, *body, *var, *lo, *hi, cache, w)?
        }
        ExprNode::Piecewise(pairs) => eval_piecewise(arena, pairs, cache, w)?,
        ExprNode::RootOf(poly, var, index) => eval_rootof(arena, *poly, *var, *index, prec)?,
        ExprNode::RootSum(poly, body, sumvar) => eval_rootsum(arena, *poly, *body, *sumvar, w)?,
        ExprNode::DefiniteIntegral(body, var, lo, hi) => {
            eval_definite_integral(arena, *body, *var, *lo, *hi, cache, w)?
        }
        node if inverse_identity(arena, node, cache, prec).is_some() => {
            let u = inverse_identity(arena, node, cache, prec).unwrap_or(id);
            return Ok(CBall::new(get_cached(cache, u)?.clone(), cache.err(&u)));
        }
        // Lambert W reports its own bound (complex arguments, every branch).
        node if lambertw::lambert_parts(arena, node).is_some() => {
            let value = match eval_node(arena, id, cache, w) {
                Ok(value) => value,
                // `W_k(0)` for `k ≠ 0` at an argument only 0 to the working
                // precision: no value yet (as in the arm below).
                Err(e) if domain_error(&e) && inexact_zero_child(node, cache) => {
                    return Ok(CBall::new(c_zero(prec), accuracy::Bound::UNKNOWN));
                }
                Err(e) => return Err(e),
            };
            let err = match lambertw::lambert_parts(arena, node) {
                Some(Ok((x, k))) => {
                    let bz = cache.err(&x);
                    lambertw::error_bound(get_cached(cache, x)?, bz, &value, k, prec)
                }
                _ => accuracy::Bound::UNKNOWN,
            };
            CBall::new(value, err)
        }
        // `loggamma` of a non-real argument reports its own bound (the
        // sensitivity rules are for real arguments).
        ExprNode::LogGamma(c) if cache.get(c).is_some_and(|v| !v.1.is_zero()) => {
            let value = eval_node(arena, id, cache, w)?;
            let bz = cache.err(c);
            let err = loggamma::error_bound(get_cached(cache, *c)?, bz, &value, prec);
            CBall::new(value, err)
        }
        // `ζ` of a rational literal next to a trivial zero, from the exact
        // rational ([`zeta_next_to_trivial_zero`]): no error of the argument.
        ExprNode::Zeta(c) if trivial_zero_offset(arena, *c).is_some() => {
            let Some(offset) = trivial_zero_offset(arena, *c) else {
                return Err(unevaluable("zeta lost its argument"));
            };
            let s = &get_cached(cache, *c)?.0;
            let v = zeta_next_to_trivial_zero(s, &offset, w)?;
            let err = accuracy::Bound::real(accuracy::part_lg(&v) + 2.0 - prec as f64);
            CBall::new((v, BigFloat::new(prec)), err)
        }
        // `ζ`, `η`, `ψ⁽ᵐ⁾`, `ψ`, `Γ` and `x!` of a non-real argument report their
        // own bounds (`zeta.rs`; the sensitivity rules are for real
        // arguments).
        node if complex_zeta_family(node, cache).is_some() => {
            let (kind, c) = complex_zeta_family(node, cache).unwrap_or((ZetaFamily::Zeta, id));
            let value = eval_node(arena, id, cache, w)?;
            let z = get_cached(cache, c)?;
            let bz = cache.err(&c);
            let err = match kind {
                ZetaFamily::Zeta => zeta::zeta_bound(z, bz, &value, prec),
                ZetaFamily::Eta => zeta::eta_bound(z, bz, &value, prec),
                ZetaFamily::Gamma => zeta::gamma_bound(z, bz, &value, prec),
                ZetaFamily::Factorial => {
                    let z1 = (
                        z.0.add(&BigFloat::from_i32(1, 64), exact_bits(&z.0, prec), RM),
                        z.1.clone(),
                    );
                    zeta::gamma_bound(&z1, bz, &value, prec)
                }
                ZetaFamily::Polygamma(order) => {
                    let bn = cache.err(&order);
                    let m = get_cached(cache, order)
                        .ok()
                        .and_then(|v| f64_of(&v.0).ok())
                        .map_or(0.0, f64::round);
                    // The order is an integer only when it is exact (an
                    // inexact one may be any number of its ball).
                    if !bn.is_exact() {
                        accuracy::Bound::UNKNOWN
                    } else {
                        zeta::polygamma_bound(m as u32, z, bz, &value, prec)
                    }
                }
                ZetaFamily::Digamma => zeta::polygamma_bound(0, z, bz, &value, prec),
                ZetaFamily::Harmonic => {
                    let z1 = (
                        z.0.add(&BigFloat::from_i32(1, 64), exact_bits(&z.0, prec), RM),
                        z.1.clone(),
                    );
                    zeta::polygamma_bound(0, &z1, bz, &value, prec)
                }
                ZetaFamily::Erf(kind) => cerf::erf_bound(kind, z, bz, &value, prec),
            };
            CBall::new(value, err)
        }
        // The special functions of `cspecial.rs` at non-real arguments (or
        // real ones where the value is not real) report their own bounds.
        node if cspecial::classify(node, cache).is_some() => {
            let Some((f, args)) = cspecial::classify(node, cache) else {
                return Err(unevaluable("special function lost its arguments"));
            };
            let vals = args
                .iter()
                .map(|a| get_cached(cache, *a))
                .collect::<Result<Vec<_>, _>>()?;
            let routed = match cspecial::eval_routed(f, &vals, w) {
                Ok(routed) => routed,
                // A domain error at 0 for an argument only 0 to the working
                // precision (as in the arm below).
                Err(e) if domain_error(&e) && inexact_zero_child(node, cache) => {
                    return Ok(CBall::new(c_zero(prec), accuracy::Bound::UNKNOWN));
                }
                Err(e) => return Err(e),
            };
            let err = cspecial::bound(arena, id, f, &args, &routed, cache, w);
            CBall::new(routed.value, err)
        }
        // Elliptic integrals of a non-real parameter, or with a complex value
        // at real arguments, report their own bound (`elliptic.rs`: each part
        // to its own precision; the sensitivity rules are for real arguments
        // and the rounding of `finish` is at the value's magnitude).
        ExprNode::Lib(f, args) if elliptic::takes(*f, args, cache) => {
            let value = eval_node(arena, id, cache, w)?;
            let err = elliptic::bound(arena, id, args, &value, cache, w);
            CBall::new(value, err)
        }
        // `hermite`, `laguerre`, `assoc_laguerre`, `rising_factorial`,
        // `falling_factorial` of a non-real argument (refused before 0.56).
        node @ ExprNode::Lib(f, args)
            if hypergeom::nonreal_args(*f, args, cache)
                || factorials::nonreal_args(*f, args, cache) =>
        {
            let value = match eval_node(arena, id, cache, w) {
                Ok(value) => value,
                Err(e) if domain_error(&e) && rounded_onto_pole(node, cache) => {
                    return Ok(CBall::new(c_zero(prec), accuracy::Bound::UNKNOWN));
                }
                Err(e) => return Err(e),
            };
            let cap = Some(sensitivity::gap_cap(prec));
            let err = if factorials::exact_product(*f, args, cache, prec) {
                accuracy::Bound::EXACT
            } else {
                sensitivity::complex_route_bound(arena, id, args, &value, cache, cap, w)
            };
            CBall::new(value, err)
        }
        ExprNode::Beta(a, b)
            if cache.get(a).is_some_and(|v| !v.1.is_zero())
                || cache.get(b).is_some_and(|v| !v.1.is_zero()) =>
        {
            let value = eval_node(arena, id, cache, w)?;
            let bound = |c: &ExprId| cache.err(c);
            let err = zeta::beta_bound(
                get_cached(cache, *a)?,
                bound(a),
                get_cached(cache, *b)?,
                bound(b),
                &value,
                prec,
            );
            CBall::new(value, err)
        }
        // `polylog` off the real interval [−1, 1] reports its own bound (the
        // sensitivity rules are for real arguments; `polylog.rs`).
        node @ ExprNode::Lib(_, args) if polylog::is_general(node, cache) => {
            let value = match eval_node(arena, id, cache, w) {
                Ok(value) => value,
                // An argument that cancelled to 0 or rounded onto 1 (as in
                // the arm below).
                Err(e)
                    if domain_error(&e)
                        && (inexact_zero_child(node, cache) || rounded_onto_pole(node, cache)) =>
                {
                    return Ok(CBall::new(c_zero(prec), accuracy::Bound::UNKNOWN));
                }
                Err(e) => return Err(e),
            };
            let bound = |c: ExprId| cache.err(&c);
            let err = polylog::error_bound(
                &get_cached(cache, args[0])?.0,
                bound(args[0]),
                get_cached(cache, args[1])?,
                bound(args[1]),
                &value,
                w,
            );
            CBall::new(value, err)
        }
        // `polylog` of a complex order reports its own bound.
        node @ ExprNode::Lib(_, args) if polylog::is_complex_order(node, cache) => {
            let value = match eval_node(arena, id, cache, w) {
                Ok(value) => value,
                // An argument that rounded onto 1 (as in the arm above).
                Err(e) if domain_error(&e) && rounded_onto_pole(node, cache) => {
                    return Ok(CBall::new(c_zero(prec), accuracy::Bound::UNKNOWN));
                }
                Err(e) => return Err(e),
            };
            let bound = |c: ExprId| cache.err(&c);
            let err = polylog::complex_order_bound(
                get_cached(cache, args[0])?,
                bound(args[0]),
                get_cached(cache, args[1])?,
                bound(args[1]),
                &value,
                w,
            );
            CBall::new(value, err)
        }
        // `sin`, `cos`, `tan` of a real inexact argument beyond `2⁶⁴`
        // (`sin(e·10¹⁰⁰)`, a quadrature node at `x ≈ 3.7·10²²`), which
        // `eval_node` refuses at every precision: the function of the
        // argument's value taken as exact, reduced modulo 2π in full, its
        // error the argument's own bound (`node_error`) — and no value yet
        // (`precision_limited`, so a higher precision follows) while that
        // bound is not well below 1.  Before 0.47 `sin(e·10¹⁰⁰)` was
        // refused (SymPy: `0.409160531722613`), and every oscillating
        // integrand on an infinite range lacked a bound at its far nodes.
        ExprNode::Sin(c) | ExprNode::Cos(c) | ExprNode::Tan(c)
            if large_inexact_trig_argument(arena, *c, cache) =>
        {
            let b = cache.err(c);
            if b.is_unknown() || b.joint() > -4.0 {
                return Ok(CBall::new(c_zero(prec), accuracy::Bound::UNKNOWN));
            }
            let arg = &get_cached(cache, *c)?.0;
            let value = match arena.node(id) {
                ExprNode::Sin(_) => arg.sin(prec, RM, w.cc),
                ExprNode::Cos(_) => arg.cos(prec, RM, w.cc),
                _ => arg.tan(prec, RM, w.cc),
            };
            let value = (value, BigFloat::new(prec));
            let err = accuracy::node_error(arena, id, &value, cache, prec, || {
                sensitivity::special_error(arena, id, &value, cache, w)
            });
            return Ok(CBall::new(value, err));
        }
        ExprNode::PhysicalConstant(_, value_id) => match cache.get(value_id) {
            // The constant is an atom to every tree walk, so its stored
            // value (`h/(2π)` for ħ) was never visited: evaluate it as a
            // subtree of its own unless it is a single already-cached node.
            Some(v) => CBall::new(v.clone(), cache.err(value_id)),
            None => {
                let order = walk::post_order_ids(arena, *value_id);
                evaluate_tree(arena, *value_id, &order, None, true, w)?
            }
        },
        _ => {
            let value = match eval_node(arena, id, cache, w) {
                Ok(value) => value,
                // A domain error at 0 (`Ci(0)`, `arg(0)`) for an argument
                // that is only 0 to the working precision says nothing
                // about the true argument: no value is known yet, and the
                // evaluation is repeated at a higher precision.  Before 0.29
                // `Ci(−(exp(10⁻³⁰) − 1 − 10⁻³⁰))` was `Unevaluable: Ci(0) is
                // −∞`.  Nor does a pole met by an inexact argument that
                // rounded onto it (`ζ(1 + 10⁻⁴⁵)`: `1 + 10⁻⁴⁵` is 1 at 128
                // bits).
                Err(e)
                    if domain_error(&e)
                        && (inexact_zero_child(arena.node(id), cache)
                            || rounded_onto_pole(arena.node(id), cache)) =>
                {
                    return Ok(CBall::new(c_zero(prec), accuracy::Bound::UNKNOWN));
                }
                Err(e) => return Err(e),
            };
            let err = accuracy::node_error(arena, id, &value, cache, prec, || {
                sensitivity::special_error(arena, id, &value, cache, w)
            });
            // `polygamma`'s order is an integer only when it is exact (the
            // sensitivity rules read an order within `2⁻⁴⁰` of an integer
            // as that integer): up to 0.53 `polygamma(1 + 10⁻⁵⁰, 2)`, whose
            // order rounds onto 1, was `ψ′(2)`.
            let err = match arena.node(id) {
                ExprNode::Polygamma(n, _) if !cache.err(n).is_exact() => accuracy::Bound::UNKNOWN,
                _ => err,
            };
            return Ok(CBall::new(value, err));
        }
    };
    let err = accuracy::reported(&value, err, prec);
    Ok(CBall::new(value, err))
}

/// The pairs `q·erf(u) + q·erfc(u)` among the terms of a sum, folded to
/// their exact value `q` (`erf + erfc = 1` identically, also off the real
/// axis).
struct ErfComplements {
    /// The terms that belong to a pair.
    paired: Vec<bool>,
    /// The sum of the `q` of the pairs.
    constant: CBall,
}

/// The terms `q·erf(u)`, `q·erfc(u)` (`q` a rational literal, 1 when absent)
/// of a sum that pair up with the same `q` and the same `u`
/// ([`ErfComplements`]), or `None` when there is no such pair.  Evaluated as
/// two rounded values their sum is `q` only within its error, a ball that
/// holds every number near `q`: up to 0.58 `erfc(√2) − 1 + erf(√2)` was a
/// zero ball at every precision, and `floor` of it — 0, a step of `floor` —
/// was refused after a search to the last precision (0.2 s; 1 s for the five
/// evaluations of the fuzzer's check).
fn erf_complements(arena: &Arena, children: &[ExprId], prec: usize) -> Option<ErfComplements> {
    // (coefficient, argument, is `erfc`) of a term `q·erf(u)` or `q·erfc(u)`.
    let term = |c: ExprId| -> Option<(Option<&crate::base::numeric::Q>, ExprId, bool)> {
        let function = |f: ExprId| match arena.node(f) {
            ExprNode::Erf(u) => Some((*u, false)),
            ExprNode::Erfc(u) => Some((*u, true)),
            _ => None,
        };
        if let Some((u, complement)) = function(c) {
            return Some((None, u, complement));
        }
        match arena.node(c) {
            ExprNode::Mul(factors) => match factors[..] {
                [q, f] => {
                    let (u, complement) = function(f)?;
                    Some((Some(arena.as_num(q)?), u, complement))
                }
                _ => None,
            },
            _ => None,
        }
    };
    // Most sums hold no `erfc`: one pass over the node kinds decides.
    let terms: Vec<_> = children.iter().map(|&c| term(c)).collect();
    if !terms.iter().flatten().any(|t| t.2) {
        return None;
    }
    let mut paired = vec![false; children.len()];
    let mut constant = crate::base::numeric::Q::zero();
    for i in 0..terms.len() {
        let Some((q, u, false)) = terms[i] else {
            continue;
        };
        let partner = (0..terms.len())
            .find(|&j| !paired[j] && matches!(terms[j], Some((p, v, true)) if p == q && v == u));
        if let Some(j) = partner {
            paired[i] = true;
            paired[j] = true;
            constant += q
                .cloned()
                .unwrap_or_else(|| crate::base::numeric::Q::from_integer(BigInt::from(1)));
        }
    }
    if !paired.iter().any(|p| *p) {
        return None;
    }
    Some(ErfComplements {
        paired,
        constant: exact::to_value(&exact::Gauss::real(constant), prec),
    })
}

/// The functions of `zeta.rs` and `cerf.rs` at a non-real argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ZetaFamily {
    Zeta,
    Eta,
    Gamma,
    Factorial,
    Digamma,
    Harmonic,
    /// `polygamma(m, z)`, with the order's node.
    Polygamma(ExprId),
    Erf(cerf::Kind),
}

/// Is `node` one of the functions of `zeta.rs` (`ζ`, `η`, `Γ`, `x!`, `ψ`,
/// `ψ⁽ᵐ⁾`, `harmonic`) or `cerf.rs` (`erf`, `erfc`, `erfi`) whose (last)
/// argument has a non-real value in `cache`?  Which one, and that argument.
fn complex_zeta_family(node: &ExprNode, cache: Cache<'_>) -> Option<(ZetaFamily, ExprId)> {
    let (kind, c) = match node {
        ExprNode::Zeta(c) => (ZetaFamily::Zeta, *c),
        ExprNode::Gamma(c) => (ZetaFamily::Gamma, *c),
        ExprNode::Factorial(c) => (ZetaFamily::Factorial, *c),
        ExprNode::Digamma(c) => (ZetaFamily::Digamma, *c),
        ExprNode::Polygamma(n, c) => (ZetaFamily::Polygamma(*n), *c),
        ExprNode::Erf(c) => (ZetaFamily::Erf(cerf::Kind::Erf), *c),
        ExprNode::Erfc(c) => (ZetaFamily::Erf(cerf::Kind::Erfc), *c),
        ExprNode::Lib(LibFn::Erfi, args) if args.len() == 1 => {
            (ZetaFamily::Erf(cerf::Kind::Erfi), args[0])
        }
        ExprNode::Lib(LibFn::DirichletEta, args) if args.len() == 1 => (ZetaFamily::Eta, args[0]),
        ExprNode::Lib(LibFn::Harmonic, args) if args.len() == 1 => (ZetaFamily::Harmonic, args[0]),
        _ => return None,
    };
    cache
        .get(&c)
        .is_some_and(|v| !v.1.is_zero())
        .then_some((kind, c))
}

/// `f(g(u)) = u` for a function `f` and the principal branch `g` of its
/// inverse — `sin∘asin`, `cos∘acos`, `tan∘atan`, `sinh∘asinh`, `cosh∘acosh`,
/// `tanh∘atanh`, `exp∘ln` — an identity on all of ℂ (where `g(u)` is defined:
/// the inner node must have evaluated): `u`, when `node` is such a
/// composition.  The value is then `u`'s, with `u`'s bound.  Numerically
/// `sin(asin(7/5·√2/√(24/25)))` is `sin(π/2 − 1.29i)`, whose imaginary part
/// `cos(π/2)·sinh(−1.29)` is 0 only to within its error, and the square root
/// of `1 − ½·sin²(…)` (a negative number) then had an undecidable side of its
/// cut (Rubi's antiderivatives with `elliptic_f(asin(…), m)`).
///
/// Where the inner function has a singular point the identity needs `u`'s
/// error ball to exclude it: `ln 0`, `atan(±i)` and `atanh(±1)` are
/// infinite, and `exp(ln 0) = exp(zoo)` has no value.  Before 0.30
/// `exp(ln((1 + √2)² − 3 − 2√2))` (a zero the canonical form does not see)
/// took the value of the zero ball and printed `0`.
fn inverse_identity(
    arena: &Arena,
    node: &ExprNode,
    cache: Cache<'_>,
    prec: usize,
) -> Option<ExprId> {
    let (outer, inner) = match node {
        ExprNode::Sin(a)
        | ExprNode::Cos(a)
        | ExprNode::Tan(a)
        | ExprNode::Sinh(a)
        | ExprNode::Cosh(a)
        | ExprNode::Tanh(a)
        | ExprNode::Exp(a) => (node, *a),
        _ => return None,
    };
    let u = match (outer, arena.node(inner)) {
        (ExprNode::Sin(_), ExprNode::Asin(u))
        | (ExprNode::Cos(_), ExprNode::Acos(u))
        | (ExprNode::Tan(_), ExprNode::Atan(u))
        | (ExprNode::Sinh(_), ExprNode::Asinh(u))
        | (ExprNode::Cosh(_), ExprNode::Acosh(u))
        | (ExprNode::Tanh(_), ExprNode::Atanh(u))
        | (ExprNode::Exp(_), ExprNode::Ln(u)) => *u,
        _ => return None,
    };
    if !cache.contains_key(&inner) {
        return None;
    }
    let uv = cache.get(&u)?;
    let singular: &[(i32, i32)] = match outer {
        ExprNode::Exp(_) => &[(0, 0)],
        ExprNode::Tan(_) => &[(0, 1), (0, -1)],
        ExprNode::Tanh(_) => &[(1, 0), (-1, 0)],
        _ => &[],
    };
    if !singular.is_empty() {
        let err = cache.err(&u).joint();
        if accuracy::is_unknown(err) {
            return None;
        }
        for &(a, b) in singular {
            let d = (
                uv.0.sub(&BigFloat::from_i32(a, prec), prec, RM),
                uv.1.sub(&BigFloat::from_i32(b, prec), prec, RM),
            );
            // One bit for the rounding of the difference.
            if accuracy::contains_zero(&d, accuracy::shift(err, 1.0)) {
                return None;
            }
        }
    }
    Some(u)
}

/// Is `node` a function with poles at integers (`Γ`, `ζ`, …) whose
/// argument is an integer at the working precision without being exact?
/// Or one with a singular point elsewhere — `Li_s(1)`, `K(1)`, `Π(1, m)`,
/// `Π(n, 1)`, `erfc⁻¹(2)`, `erf⁻¹(±1)` — met by an argument that rounded
/// onto it (before 0.31 `polylog(1/3, 1 − 10⁻⁴⁰)` was "`polylog(s, 1)`
/// diverges" at 16 digits: `1 − 10⁻⁴⁰` is 1 at 128 bits)?
fn rounded_onto_pole(node: &ExprNode, cache: Cache<'_>) -> bool {
    if let ExprNode::Lib(f, _) = node {
        let args = node.children();
        // Is argument `k` an inexact real value equal to one of `points`?
        let at = |k: usize, points: &[i32]| -> bool {
            args.get(k).is_some_and(|c| {
                let inexact = !cache.err(c).is_exact();
                inexact
                    && cache.get(c).is_some_and(|v| {
                        v.1.is_zero()
                            && points
                                .iter()
                                .any(|&p| v.0.cmp(&BigFloat::from_i32(p, 64)) == Some(0))
                    })
            })
        };
        match f {
            LibFn::PolyLog => return at(1, &[1]),
            LibFn::EllipticK => return at(0, &[1]),
            LibFn::EllipticPi => return at(0, &[1]) || at(1, &[1]),
            LibFn::ErfcInv => return at(0, &[2]),
            LibFn::ErfInv => return at(0, &[-1, 1]),
            // `E_n(0)` is finite for `n > 1` only: an order that rounded
            // onto 1 (`expint(1 + 10⁻⁴⁰, 0)` = 10⁴⁰) is decided at a
            // higher precision.
            LibFn::ExpInt => {
                let zero =
                    |c: &ExprId| cache.get(c).is_some_and(|v| v.0.is_zero() && v.1.is_zero());
                return at(0, &[1]) && args.get(1).is_some_and(zero);
            }
            _ => {}
        }
    }
    let poles = match node {
        ExprNode::Gamma(_)
        | ExprNode::LogGamma(_)
        | ExprNode::Digamma(_)
        | ExprNode::Polygamma(_, _)
        | ExprNode::Zeta(_)
        | ExprNode::Factorial(_)
        | ExprNode::Beta(_, _)
        | ExprNode::Binomial(_, _) => true,
        ExprNode::Lib(f, _) => matches!(
            f,
            LibFn::RisingFactorial | LibFn::FallingFactorial | LibFn::Harmonic
        ),
        _ => false,
    };
    poles
        && node.children().into_iter().any(|c| {
            cache.get(&c).is_some_and(|v| v.1.is_zero() && v.0.is_int())
                && !cache.err(&c).is_exact()
        })
}

/// Does `node` have a child whose value is 0 without being exactly 0?
fn inexact_zero_child(node: &ExprNode, cache: Cache<'_>) -> bool {
    node.children().into_iter().any(|c| {
        cache
            .get(&c)
            .is_some_and(|v| accuracy::mag(v).is_none() && !cache.err(&c).is_exact())
    })
}

/// How far [`evaluate_adaptive`] pursues a value that is zero to the working
/// precision (its error ball contains 0 and shrinks with the precision, or
/// the value itself falls with every increase in precision) before it is
/// returned as 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ZeroSearch {
    /// To the cap of every evaluation, twice the working precision plus 256
    /// bits: the internal callers of [`evalf_complex`] and
    /// [`eval_const_f64`], which compare the value against a tolerance (the
    /// integrator's `F′ − f` check) and meet true zeros all the time.
    Cap,
    /// [`ZERO_SEARCH_BITS`] beyond the working precision: the zero and sign
    /// tests of the crate that go beyond the cap (`solve`, `polysys`,
    /// `poly::algebraic`), whose candidates are exact algebraic numbers.
    Deep,
    /// As [`Deep`](ZeroSearch::Deep), and at least to the precision that
    /// holds every exact rational of the expression in full (see
    /// [`input_bits`]): results read as numbers ([`evalf`], `Ex::eval_f64`,
    /// `Ex::eval_complex64`), where a 0 that is really `2.5·10⁻¹⁴⁹` is a
    /// wrong answer.
    Number,
}

/// The bits that hold the rational `p/q` in full: `max(bits(p), bits(q))`,
/// or the bits of the odd part of `p` over a power of two (a float of that
/// many bits holds it exactly).  A zero to a precision below the largest of
/// these over the exact values of an expression may be the rounding of an
/// input: `sin(2/3 + 10⁻²⁰⁰⁰) − sin(2/3)` is the difference of two equal
/// floats below 6,646 bits (before 0.35 it printed `0`).
fn input_bits(q: &crate::base::numeric::Q) -> u64 {
    let (p, d) = (q.numer(), q.denom());
    if d.bits() == d.trailing_zeros().unwrap_or(0) + 1 {
        p.bits() - p.trailing_zeros().unwrap_or(0)
    } else {
        p.bits().max(d.bits())
    }
}

/// The bits that hold the second-order part of an exact rational `u + h`
/// with a small part ([`rational_small_part`]): `2·log₂ abs(u/h)`.  Held to
/// [`input_bits`] only, `u·(1 + h/u + O((h/u)²))` loses its `(h/u)²` term,
/// which a cancellation of the first two leaves: before 0.36 `(1 +
/// 10⁻⁷⁰⁰)⁻¹·((√2 + √3)² − 2√6)/5 − 1 + 10⁻⁷⁰⁰` (truly `10⁻¹⁴⁰⁰`; the
/// canonical form holds `10⁷⁰⁰/(5·(10⁷⁰⁰ + 1))` and `1 − 10⁻⁷⁰⁰`, in full
/// at 2,328 bits, agreeing to 4,651) printed `0`.  0 for a rational without
/// one, below 256 bits, or beyond the configured maximum precision (refused
/// as it is).
fn second_order_bits(q: &crate::base::numeric::Q, arena: &Arena) -> u64 {
    // Below 256 bits the second-order part lies within the zero search of
    // every evaluation ([`ZERO_SEARCH_BITS`]).
    let bits = input_bits(q);
    if !(256..=u64::from(arena.config.max_evalf_precision)).contains(&bits) {
        return 0;
    }
    rational_small_part(q).map_or(0, |r| (-2.0 * r).ceil().max(0.0) as u64)
}

/// Bits beyond the working precision to which [`ZeroSearch::Deep`] pursues
/// a value that is zero to the precision reached (≈ 308 decimal digits of
/// cancellation).  A tiny value hidden by the cancellation of larger terms
/// has no digits until the precision exceeds the depth of the
/// cancellation, `mag(terms) − mag(value)`, and nothing before that
/// distinguishes it from a true zero; the search doubles the precision.
/// A true zero — met all the time, by every `solve` candidate check — pays
/// one evaluation at about this many bits, which bounds the budget: 3,072
/// bits made one `solve_general` fixture of the SymPy oracle (17 true-zero
/// checks) take 6 s in a debug build.  Deeper cancellations are avoided
/// at their source (`eval` keeps `polygamma(n, p/q)` symbolic when its
/// closed form would cancel more than 200 bits).  Before 0.30 the search
/// stopped at the cap of the digits: `1/2 − erf(13√2)/2` (`2.476·10⁻¹⁴⁹`,
/// 493 bits of cancellation) was `0.0` from `eval_f64`.
pub(crate) const ZERO_SEARCH_BITS: usize = 1024;

/// How a value returned by [`evaluate_adaptive`] was settled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Settled {
    /// Every part the value shows carries the requested digits (an exact
    /// zero included).
    Certified,
    /// Zero to the precision reached: the true value is within the error
    /// bound of the last evaluation, and not known to be 0.
    ZeroToPrecision,
}

/// How [`evaluate_adaptive`] ended.
enum Adaptive {
    /// A value, and how it was settled.
    Settled(Complex, Settled),
    /// A number that is certainly not 0 came out 0 below the exponent
    /// range of `BigFloat` (`besselk(0, 10¹⁰) ≈ 1.2·10^(−4342944824)`):
    /// no digit of it is known ([`underflow_error`]), at most its sign.
    Underflow(accuracy::Nonzero),
    /// A number below the exponent range known to the requested digits
    /// scaled (`extended`): `m·2^k` (`erfc(10⁵)`, `exp(−10¹⁰) −
    /// exp(−2·10¹⁰)`).  Not printed ([`underflow_error`]), but its sign
    /// and direction are known.
    Beyond(extended::Scaled),
    /// 0 within the underflow bound and not known to be nonzero: a sum of
    /// such numbers of both signs without a scaled form (`besselk(0, 10¹⁰)
    /// − airyai(10⁷)`) — no precision decides it ([`undecided_error`]).
    Undecided,
    /// Zero to the precision reached, but a sum lost a term that is
    /// certainly not 0 below its error bound ([`accuracy::Absorbed`]) and no
    /// precision within the limit keeps it: `below_range` — the term lies
    /// below the exponent range (`sin²1 + cos²1 − 1 + e^(−10¹⁰)`,
    /// [`hidden_error`]); otherwise it needs more than the configured maximum
    /// precision.  Not known to be 0: if the rest cancels exactly, the value is
    /// the lost term.
    Hidden { below_range: bool },
    /// A value in range with its digits, one part of which is certainly not
    /// 0 but lies below the exponent range ([`accuracy::PartsBelow`]): `1 +
    /// i·e^(−10¹⁰)`.  The value holds that part as 0; a result read as a
    /// number is refused ([`part_underflow_error`]): a certified nonzero part
    /// is never dropped (decision D3), and this one cannot be printed.  Only
    /// for [`ZeroSearch::Number`].
    PartBelow {
        /// The value, with the part below the range set to 0.
        value: Complex,
        /// The part below the range is the imaginary one (else the real one).
        imaginary: bool,
    },
}

/// The error of a decimal result one part of which is certainly not 0 but
/// lies below the exponent range, next to a part within it
/// ([`Adaptive::PartBelow`]): the same kind as a whole value below the range
/// ([`underflow_error`]).  Up to 0.58 the part was dropped like one whose
/// ball contains 0: `1 + I*exp(-10^10)` printed `1`, `lowergamma(1, 10^10 +
/// I)` (`1 − e^(−10¹⁰ − i)`) printed `1` (mpmath, with unbounded exponents:
/// `1 + 9.27858442032487e-4342944820j`).
fn part_underflow_error(imaginary: bool) -> SymplexError {
    let (part, other) = if imaginary {
        ("imaginary", "a real")
    } else {
        ("real", "an imaginary")
    };
    SymplexError::Unevaluable {
        reason: format!(
            "the {part} part is not 0 but underflows the arbitrary-precision exponent range \
             (its magnitude is below 2^{:e}) next to {other} part within it: it cannot be \
             printed, and is not dropped",
            f64::from(astro_float::EXPONENT_MIN)
        ),
    }
}

/// The error of a decimal result that is zero to the precision reached
/// while a nonzero term below the exponent range was lost in it
/// ([`Adaptive::Hidden`]).  Before 0.35 it printed `0`:
/// `sin²1 + cos²1 − 1 + e^(−10¹⁰)` (truly `e^(−10¹⁰)`).
fn hidden_error() -> SymplexError {
    SymplexError::Unevaluable {
        reason: format!(
            "the value is 0 to the precision reached, but a term that is not 0 and lies below the \
             arbitrary-precision exponent range (2^{:e}) was lost in a cancellation: not known \
             to be 0",
            f64::from(astro_float::EXPONENT_MIN)
        ),
    }
}

/// The error of a decimal result that is zero to the precision reached
/// while a nonzero term was lost below its error bound deeper than the
/// configured maximum precision reaches ([`Adaptive::Hidden`]).
fn hidden_deep_error(digits: u32) -> SymplexError {
    SymplexError::PrecisionExhausted {
        requested: digits,
        achieved: 0,
    }
}

/// The error of a decimal result that is 0 within the underflow bound and
/// not known to be anything ([`Adaptive::Undecided`]).  Before 0.34 it was
/// printed `0`, and so was a sum of scaled values of both signs:
/// `exp(−10¹⁰) − exp(−2·10¹⁰)`.
fn undecided_error() -> SymplexError {
    SymplexError::Unevaluable {
        reason: format!(
            "the value underflows the arbitrary-precision exponent range (its magnitude is \
             below 2^{:e}) and is not known to be 0 or not: a sum of such values of both signs",
            f64::from(astro_float::EXPONENT_MIN)
        ),
    }
}

/// The error of a decimal result that is a nonzero number below the
/// exponent range ([`Adaptive::Underflow`]).  mpmath, with unbounded
/// exponents, prints `erfc(10⁵)` as `5.2e-4342944825`; before 0.32 `evalf`
/// printed `0`, a digit that is not correct.  (The `f64` routes return
/// `0.0`, the correctly rounded `f64` of such a number.)
fn underflow_error() -> SymplexError {
    SymplexError::Unevaluable {
        reason: format!(
            "the value is not 0 but underflows the arbitrary-precision exponent range \
             (its magnitude is below 2^{:e})",
            f64::from(astro_float::EXPONENT_MIN)
        ),
    }
}

/// Is `node` `Γ(a, z)`, `E_n(z)` or `erfc(z)` with a non-real argument — the
/// functions whose value below the exponent range only the scaled form of
/// their complex route holds (`cexpint::below_range`; a node with such a value
/// is the caller's test)?  Up to 0.58 such a node failed (`PrecisionExhausted`)
/// and with it every expression that needs it.  Since 0.59 its parts are
/// certified through sums, products and the functions of
/// `accuracy::parts_below`; where that certificate does not reach — a product
/// of two factors with such parts of different signs, a function without a
/// rule — a result read as a number that would drop a part is refused as
/// before ([`untracked_part_error`]) rather than printed without it.
fn held_by_complex_route(node: &ExprNode, cache: Cache<'_>) -> bool {
    let nonreal = |c: &ExprId| cache.get(c).is_some_and(|v| !v.1.is_zero());
    match node {
        ExprNode::Erfc(c) => nonreal(c),
        ExprNode::Lib(LibFn::UpperGamma | LibFn::ExpInt, args) => args.iter().any(nonreal),
        _ => false,
    }
}

/// The error of a decimal result one part of which is 0 within its error
/// while the value depends on a part below the exponent range that the
/// certificate of [`Adaptive::PartBelow`] does not follow through the
/// expression ([`held_by_complex_route`]): the part may be 0 or not, and is
/// not dropped.  (`(π + π·E₂(10¹⁰ − i))·(−3 + (1 + i)·K₀(10¹⁰))`: mpmath gives the
/// imaginary part `3.65e-4342944824`.)
fn untracked_part_error(imaginary: bool) -> SymplexError {
    let part = if imaginary { "imaginary" } else { "real" };
    SymplexError::Unevaluable {
        reason: format!(
            "the {part} part is 0 within its error, and the value depends on a part below the \
             arbitrary-precision exponent range (2^{:e}) that is not followed through this \
             expression: not known to be 0, and not dropped",
            f64::from(astro_float::EXPONENT_MIN)
        ),
    }
}

/// Evaluate `expr` to `digits` correct significant digits.
///
/// The value is computed at [`working_precision`] with an error bound
/// ([`accuracy`]).  It is accepted when the bound covers the requested
/// digits of every part it shows ([`settle`]); otherwise — catastrophic
/// cancellation, amplification by `exp` or a division by a small
/// difference, or just a pessimistic bound — the whole expression is
/// re-evaluated at a higher precision (Ziv's strategy), up to twice the
/// initial precision plus 256 bits (and the configured maximum), as SymPy's
/// `evalf` retries an `Add` that lost its accuracy.  Before 0.26 the first
/// value was returned whatever its accuracy: `exp(10⁻³⁰) − 1` came out `0`,
/// and a quotient by a difference that is zero to the working precision
/// came out as noise.
///
/// Only the bound certifies digits.  Before 0.30 two successive nonzero
/// values that agreed to the requested digits were accepted too, and
/// agreement is no evidence when an input rounds to the same value at both
/// precisions: `(1 + 10⁻¹⁵⁰)^(10¹⁵⁰)` (truly `e`) was `1` at 16 and 30
/// digits (`1 + 10⁻¹⁵⁰` is 1 at 128 and at 256 bits), and
/// `exp(11/15·10⁻⁴⁰) − (1 + 11/15·10⁻⁴⁰) + I₅₀(1)` lost its first term
/// (`2.7·10⁻⁸¹`, below both precisions) and printed `2.934…·10⁻⁸⁰` for
/// `3.204…·10⁻⁸⁰`.  The nodes that are more accurate than their propagated
/// bound report their own (`sin` of an exact huge rational, reduced from
/// its exact value: see `accuracy::node_error`).
///
/// A value that is zero to the precision reached — its error ball contains
/// 0 and shrank with the precision, or the value itself shrank with every
/// increase in precision — is pursued further, doubling the precision up
/// to the limit of `search` (see [`ZeroSearch`], [`ZERO_SEARCH_BITS`]):
/// a tiny value hidden by cancellation is resolved there.  So is a value
/// whose bound shrinks with the precision without covering the digits yet
/// (`(1 + 10⁻¹⁵⁰)^(10¹⁵⁰)` needs about 560 bits at 16 digits), unless the
/// bound says the digits need more than the limit (refused at once).  At the limit a
/// ball that still contains 0 is returned as 0 ([`Settled::ZeroToPrecision`];
/// SymPy returns a float with no significant digits).  Any other value
/// lacking the digits is refused with [`SymplexError::PrecisionExhausted`]:
/// a value whose magnitude fell with the precision but whose ball excludes
/// 0 (before 0.30 it was 0 too), or an infinite value — from finite inputs
/// it can only come from a division by a difference that cancelled to 0,
/// whose true value is unknown.
///
/// The zero search is not only for the root.  A value without a bound, or
/// without a finite value (NaN, `±∞`), because some node's *arguments* are
/// too imprecise ([`precision_limited`]: a quotient of two differences that
/// are both 0 at the working precision, a function at an argument that
/// cancelled to 0 or onto a pole, the side of a cut decided by a part within
/// its error of 0) is pursued the same way, to the same limit: the
/// arguments' balls shrink with the precision.  Before 0.31 such a value
/// stopped at the cap (NaN at once): `(e^ε − 1)/(e^{2ε} − 1)` at `ε = 10⁻⁴⁵`
/// was refused at 16 digits and `erfc⁻¹(sin(1 + 10⁻¹⁵⁰) − sin 1)` at 16 and
/// 30 digits.  The internal tolerance checks ([`ZeroSearch::Cap`]) are not
/// affected (their search ends at the cap), and an infinite value that no
/// precision cures — beyond the exponent range — fails where it arises
/// ([`range_error`]).
fn evaluate_adaptive(
    arena: &Arena,
    expr: ExprId,
    digits: u32,
    search: ZeroSearch,
    cc: &mut Consts,
) -> Result<Adaptive, SymplexError> {
    let prec0 = working_precision(digits);
    let max_prec = arena.config.max_evalf_precision as usize;
    let cap = (2 * prec0).max(prec0 + 256).min(max_prec).max(prec0);
    let post_order = walk::post_order_ids(arena, expr);
    // A definite integral is a quadrature at every precision: its search
    // ends at the cap, and a zero ball within the digits is 0 at once (see
    // below).
    let integral = post_order
        .iter()
        .any(|&id| matches!(arena.node(id), ExprNode::DefiniteIntegral(..)));
    let deep = match search {
        ZeroSearch::Cap => cap,
        _ if integral => cap,
        ZeroSearch::Deep | ZeroSearch::Number => (prec0 + ZERO_SEARCH_BITS).min(max_prec).max(cap),
    };
    let needed = i64::from(digits) * 3322 / 1000 + 4;
    let mut prec = prec0;
    // The previous evaluation: the magnitude of its value and its error
    // bound, both absolute (`log₂`), and its precision.
    let mut previous: Option<(Option<i64>, usize, accuracy::ErrExp)> = None;
    // Pursuits of a zero ball that lost a term (each to a higher precision,
    // within the configured maximum; the count is only a safeguard).
    let mut hidden_pursuits = 0u32;
    loop {
        let Evaluated {
            value,
            bound,
            precision_limited,
            underflowed_nonzero,
            beyond,
            parts_below,
            untracked_below,
            absorbed,
            exact_bits,
            second_bits,
        } = evaluate_once(arena, expr, &post_order, &mut Wp::new(prec, cc))?;
        // The precision that holds every exact value in full, then their
        // second-order parts, for a result read as a number.
        let (inputs, second) = match search {
            ZeroSearch::Number => (
                usize::try_from(exact_bits).unwrap_or(usize::MAX),
                usize::try_from(second_bits).unwrap_or(usize::MAX),
            ),
            ZeroSearch::Cap | ZeroSearch::Deep => (0, 0),
        };
        // A value below the exponent range known scaled is settled as its
        // mantissa, the absolute tests shifted by its scale.
        let (value, bound, scale) = match &beyond {
            Some(s) => (s.m.clone(), s.eb, s.k),
            None => (value, bound, 0),
        };
        let err = bound.joint();
        let below =
            beyond.is_none() && accuracy::mag(&value).is_none() && accuracy::is_underflow(err);
        // A number that is not 0 but lies below the exponent range: no
        // precision brings it back, and 0 is not one of its digits.
        if below && let Some(s) = underflowed_nonzero {
            debug!(prec, err, "evalf: a nonzero value underflowed");
            return Ok(Adaptive::Underflow(s));
        }
        // 0 within the underflow bound, of unknown sign or zeroness: no
        // precision decides it.  Before 0.34 it was a zero to the precision.
        if below {
            debug!(prec, err, "evalf: an underflow of unknown sign");
            return Ok(Adaptive::Undecided);
        }
        let nan = value.0.is_nan() || value.1.is_nan();
        let finite = !(nan || value.0.is_inf() || value.1.is_inf());
        // A value without a bound (or without a finite value) because some
        // node's arguments are too imprecise — not because the root is zero
        // to the working precision — is pursued as such a root is, to the end
        // of the zero search: the arguments' balls shrink with the precision
        // (`precision_limited`).  Before 0.31 it stopped at the cap: `(e^ε −
        // 1)/(e^{2ε} − 1)` at `ε = 10⁻⁴⁵` (`0/0` at 128 bits, NaN) was refused
        // at once at 16 digits, and `erfc⁻¹(sin(1 + 10⁻¹⁵⁰) − sin 1)` (its
        // argument needs about 600 bits) at 16 and 30 digits.
        let limited = precision_limited && (!finite || accuracy::is_unknown(err));
        if nan && !limited {
            return Err(SymplexError::PrecisionExhausted {
                requested: digits,
                achieved: 0,
            });
        }
        if finite && let Some((v, zeroed)) = settle(&value, bound, digits, needed, prec, search) {
            // A part set to 0 that is certainly not 0 and lies below the
            // exponent range: no precision brings it back, and a result read
            // as a number does not drop it (D3).  Up to 0.58 it was dropped
            // as a part whose ball contains 0 is: `1 + i·e^(−10¹⁰)` printed
            // `1`.  A part 0 within the underflow bound and not known to be
            // nonzero is dropped as before.
            if search == ZeroSearch::Number && beyond.is_none() {
                let below_im = zeroed.im && parts_below.im.is_some();
                if below_im || (zeroed.re && parts_below.re.is_some()) {
                    debug!(prec, ?parts_below, "evalf: a nonzero part underflowed");
                    return Ok(Adaptive::PartBelow {
                        value: v,
                        imaginary: below_im,
                    });
                }
            }
            let zero_part = zeroed.re || zeroed.im;
            // A part that is 0 within its ball next to a part with its
            // digits, in a result read as a number, is pursued to the cap
            // (one evaluation there) before it is dropped: its ball holds
            // the value's rounding and the error the rest of the expression
            // carries into it, which the working precision may leave far
            // above a part that is not 0.  Up to 0.54 it was dropped at
            // once: `exp(i(π/2 + 10⁻⁴⁵))` printed `i` at 10 digits (its
            // real part `−10⁻⁴⁵` needs about 190 bits), `Ci(−69·10³⁰π)`
            // `3.141592654*i` (`−2.13·10⁻⁶⁵`, 250 bits).  Not to the depth
            // of the zero search of a whole value: a value whose ball still
            // contains 0 is printed `0`, wrong in every digit if it is not,
            // while a part dropped next to a part with its digits leaves
            // the number right to those digits (D3).
            if zero_part
                && search == ZeroSearch::Number
                && beyond.is_none()
                && !integral
                && prec < cap
            {
                debug!(
                    prec,
                    cap, "evalf: a part is zero to the working precision; pursued"
                );
                previous = Some((accuracy::mag(&value), prec, err));
                prec = cap;
                continue;
            }
            if let Some(s) = beyond {
                debug!(prec, scale, "evalf: a value below the exponent range");
                return Ok(Adaptive::Beyond(extended::Scaled { m: v, ..s }));
            }
            // A part about to be dropped without a certificate, in a number
            // that depends on a value only a complex route's scaled form holds
            // below the range: refused, as the whole expression was while
            // that route failed ([`untracked_part_error`]).
            if zero_part && search == ZeroSearch::Number && untracked_below {
                debug!(prec, "evalf: an untracked part below the exponent range");
                return Err(untracked_part_error(zeroed.im));
            }
            return Ok(Adaptive::Settled(v, Settled::Certified));
        }
        let lift = |e: accuracy::ErrExp| accuracy::shift(e, scale as f64);
        let acc = accuracy::accurate_bits(&value, err);
        // Only with a bound: `sign` of a value that cancelled to 0 is 0 at
        // every precision, but its true value may be ±1.  And only a ball
        // that shrinks with the precision: a true zero's error falls about
        // a bit per bit gained, while near a pole it grows with the value —
        // `tan(π(1/2 − 10⁻³⁰⁰))` (truly `3.2·10²⁹⁹`) had a ball of `2³⁹⁶`
        // around `10⁴⁰` noise, which contains 0, and came out `0`.
        //
        // And only a value that is itself noise: 0, or falling with the
        // precision as the rounding residue of a cancellation does.  A
        // value that stays put inside a ball that contains 0 has a
        // pessimistic bound, not a zero: `zeta(720·polygamma(1669, E))` is
        // 1 at every precision, its argument (`≈ 10⁴⁰⁰⁰`) is known to its
        // relative error, the bound of `ζ′` was at least 1, so the ball held
        // 0 — and the value was `0` before 0.30.
        let known = finite && !accuracy::is_unknown(err);
        let (ball_shrinking, value_shrinking) = match &previous {
            None => (true, accuracy::mag(&value).is_none()),
            Some((prev_mag, prev_prec, prev_err)) => {
                let gained = i64::try_from(prec - prev_prec).unwrap_or(0);
                // (An underflow's ball is the bottom of the exponent range.)
                let half = (gained / 2) as f64;
                let ball = accuracy::is_exact(err)
                    || accuracy::is_underflow(err)
                    || lift(err) <= *prev_err - half;
                // Against the previous value, or its ball when it was 0.
                let value = match (accuracy::mag(&value).map(|m| m + scale), prev_mag) {
                    (None, _) => true,
                    (Some(m), Some(pm)) => m <= pm - gained / 2,
                    (Some(m), None) => m as f64 <= *prev_err - half,
                };
                (ball, value)
            }
        };
        let noise = known && value_shrinking;
        let zero_ball = noise && ball_shrinking && accuracy::contains_zero(&value, err);
        // A zero ball is returned as 0 only when its radius is below the
        // requested digits in absolute terms, `2^err ≤ 2^−needed`: the 0
        // loses the radius, and only then is it right to those digits
        // whatever the true value.  A wider ball says nothing (it holds
        // numbers of every size up to its radius): the cancellation is
        // deeper than the search so far.  A result read as a number, and
        // the deep tests of the crate ([`ZeroSearch::Number`],
        // [`ZeroSearch::Deep`]), pursue it to the configured maximum
        // precision; the internal tolerance checks ([`ZeroSearch::Cap`]) stop
        // at their cap.  Then it is refused.  Before 0.30 any zero ball at
        // the end of the search was 0: `(i·sin s + cos s)·e^(cos 2x)` with
        // `s = sin 2x` at `x = 1/3 + 4i` (the trigonometric form of
        // `exp(exp(2ix))`, truly `1.000263649 + 0.000207495i`) cancels about
        // 3,400 bits and was `0` with a ball of radius `2²²²⁷` at 1,152 bits.
        let small_zero = zero_ball && lift(err) <= -(needed as f64);
        // An expression with a definite integral whose ball contains 0 and is
        // within the requested digits in absolute terms is 0 to them at once:
        // its value lies within `2^err ≤ 2^−needed` of 0, and a deeper search
        // is a quadrature at every step.  Before 0.43 an integral equal to 0
        // (`∫₀^{2π} −(1 − 2 sin t)·cos(2t + 4 cos t) dt`) was refused.
        if integral
            && known
            && beyond.is_none()
            && accuracy::contains_zero(&value, err)
            && err <= -(needed as f64)
        {
            debug!(prec, err, "evalf: an integral is zero to the digits");
            return Ok(Adaptive::Settled(c_zero(prec0), Settled::ZeroToPrecision));
        }
        let wide_zero = zero_ball && !small_zero;
        // Past the cap only a value that is noise, or whose bound falls with
        // the precision (the precision it needs is then predictable), is
        // pursued, to the end of the search — not an underflow: below the
        // exponent range no precision resolves it.  (`(1 + 10⁻¹⁵⁰)^(10¹⁵⁰)`
        // needs about 560 bits at 16 digits, past the cap of 384.)
        let converging = known && ball_shrinking && previous.is_some();
        let pursue = (noise || converging || limited) && !accuracy::is_underflow(err);
        let limit = if wide_zero && search != ZeroSearch::Cap && !integral {
            max_prec.max(deep)
        } else if pursue || prec > cap {
            deep
        } else {
            cap
        };
        // A bound that says the digits need more than the search allows is
        // refused now rather than after an evaluation at its limit.
        if let Some(a) = acc
            && known
            && !noise
            && prec < limit
            && prec as f64 + (needed as f64 - a) > deep as f64
        {
            debug!(
                prec,
                err,
                ?acc,
                "evalf: the bound needs more than the search allows"
            );
            return Err(SymplexError::PrecisionExhausted {
                requested: digits,
                achieved: achieved_digits(Some(a), digits),
            });
        }
        // A zero ball that lost a nonzero term below its error is not taken
        // for 0 by a result read as a number: if the rest cancels exactly,
        // the value is that term.  It is pursued to the precision that keeps
        // the nearest such term (its `depth` bits more, and a margin) — and
        // from there, as long as some term is lost — and refused when that
        // is beyond the configured maximum or the term lies below the
        // exponent range.  Before 0.35 it was 0: `sin²1 + cos²1 − 1 +
        // e^(−10¹⁰)`, `sin²1 + cos²1 − 1 + sin(10⁻²⁰⁰⁰)` (`10⁻²⁰⁰⁰`, 6,644
        // bits below the terms, beyond the 1,024-bit search).
        //
        // A zero below the precision that holds the exact inputs in full
        // (`input_bits`) may be the rounding of an input, and a result read
        // as a number pursues it there too: before 0.35 `sin(2/3 + 10⁻⁴⁰⁰)
        // − sin(2/3)` printed `0` (truly `7.86·10⁻⁴⁰¹`).  And from there to
        // the precision of the second-order part of an input with a small
        // part (`second_order_bits`; refused beyond the maximum).
        let margin = usize::try_from(needed).unwrap_or(0) + 32;
        let lost_target = absorbed.and_then(|a| a.nearest).map(|nearest| {
            prec.saturating_add(nearest.ceil() as usize)
                .saturating_add(margin)
        });
        let input_target = if inputs > prec {
            Some(inputs.saturating_add(margin))
        } else {
            (second > prec).then(|| second.saturating_add(margin))
        };
        // A part certainly not 0 below the exponent range next to a part
        // that is zero to the precision reached (`Ei(−10¹⁰ + i) − iπ`) is
        // such a lost term too.
        let lost_below_range =
            absorbed.is_some_and(|a| a.below_range) || (beyond.is_none() && !parts_below.is_none());
        let hidden = if !small_zero || search == ZeroSearch::Cap {
            None
        } else if lost_below_range {
            Some(None)
        } else {
            lost_target
                .into_iter()
                .chain(input_target)
                .min()
                .map(|target| (target <= max_prec && hidden_pursuits < 16).then_some(target))
        };

        if prec >= limit {
            if small_zero {
                match hidden {
                    None => {
                        debug!(prec, err, "evalf: zero to the working precision");
                        return Ok(Adaptive::Settled(c_zero(prec0), Settled::ZeroToPrecision));
                    }
                    Some(Some(target)) => {
                        debug!(prec, target, "evalf: a zero ball lost a term; pursued");
                        previous = Some((None, prec, lift(err)));
                        prec = target;
                        hidden_pursuits += 1;
                        continue;
                    }
                    Some(None) => {
                        debug!(prec, err, ?absorbed, "evalf: a zero ball lost a term");
                        return Ok(Adaptive::Hidden {
                            below_range: lost_below_range,
                        });
                    }
                }
            }
            debug!(prec, err, ?acc, "evalf: precision exhausted");
            return Err(SymplexError::PrecisionExhausted {
                requested: digits,
                achieved: achieved_digits(acc, digits),
            });
        }
        let step = match acc {
            Some(a) if known => (needed as f64 - a).clamp(0.0, prec as f64).ceil() as usize + 32,
            // A value that is 0 to the working precision says nothing about
            // the precision its first bit needs: go to the cap, and from
            // there double.
            None if known && accuracy::mag(&value).is_none() && prec < cap => cap - prec,
            _ => prec,
        };
        previous = Some((accuracy::mag(&value).map(|m| m + scale), prec, lift(err)));
        prec = (prec + step.max(64)).min(limit);
        // Within a quarter of the limit, the limit: the search for a zero
        // would otherwise end with two evaluations of nearly the same cost
        // (3,072 and 3,200 bits at 16 digits).
        if prec > cap && 4 * (limit - prec) < limit {
            prec = limit;
        }
        debug!(prec, ?acc, "evalf: re-evaluating at a higher precision");
    }
}

/// The decimal digits `acc` accurate bits stand for, at most `digits`.
fn achieved_digits(acc: Option<f64>, digits: u32) -> u32 {
    acc.map_or(0.0, |a| (a.max(0.0) * 0.301).min(f64::from(digits))) as u32
}

/// `value ± bound` as a result certified to `digits` digits (`needed`
/// bits), or `None` when it is not yet.
///
/// The printer shows each part of a complex value with `digits`
/// significant digits of its own unless it is negligible next to the
/// other part ([`is_negligible_part`]), so a part that is shown must carry
/// `needed` bits relative to its own magnitude, not only relative to `|z|`.
/// A part whose error ball contains 0 is below the precision of the result
/// (the joint bound holds) and is set to 0: its digits are noise.  Before
/// 0.30 only the joint accuracy was required, and the smaller part printed
/// uncertified digits: `ln(W(−1/e + 10⁻³⁵))` was `−7.373338…·10⁻¹⁸ + πi`
/// at 16 digits (truly `−7.373306…·10⁻¹⁸`), `Ci(x) + i·(e^(10⁻¹⁶) − 1 −
/// 10⁻¹⁶)` printed an imaginary part `5.00000000000000016666666671757·10⁻³³`
/// at 30 digits (truly `…6666666666667·10⁻³³`).
///
/// A result read as a number ([`ZeroSearch::Number`]: [`evalf`],
/// `Ex::eval_f64`, `Ex::eval_complex64`) shows every part whose ball
/// excludes 0, however small next to the other, so such a part needs its
/// digits too; for the other searches a part negligible next to the other
/// is kept as computed (its digits are not read).  Up to 0.52 the printer
/// dropped a negligible part even when it was certified nonzero:
/// `Chi(−60)` printed `9.68091107e23` at 10 digits (it is `9.68091107·10²³ +
/// 3.141592654i`), `10³⁰ + i` printed `1e30`.
///
/// The flags say which parts were set to 0 although they are not exactly 0
/// ([`Zeroed`]: [`evaluate_adaptive`] pursues such a part of a number to the
/// cap, and refuses the number when the part is certainly not 0 below the
/// exponent range).
fn settle(
    value: &Complex,
    bound: accuracy::Bound,
    digits: u32,
    needed: i64,
    prec: usize,
    search: ZeroSearch,
) -> Option<(Complex, Zeroed)> {
    if accuracy::mag(value).is_none() {
        return bound.is_exact().then(|| (value.clone(), Zeroed::default()));
    }
    if accuracy::accurate_bits(value, bound.joint())? < needed as f64 {
        return None;
    }
    // `Some(true)`: set the part to 0; `Some(false)`: keep it; `None`: it
    // lacks digits.
    let part = |p: &BigFloat, e: accuracy::ErrExp, other: &BigFloat| -> Option<bool> {
        if accuracy::exact_zero(p, e) {
            return Some(false);
        }
        if accuracy::part_contains_zero(p, e) {
            return Some(true);
        }
        if accuracy::is_exact(e)
            || (search != ZeroSearch::Number && is_negligible_part(p, other, digits))
        {
            return Some(false);
        }
        (accuracy::part_lg_low(p) - e >= needed as f64).then_some(false)
    };
    let zero_re = part(&value.0, bound.re, &value.1)?;
    let zero_im = part(&value.1, bound.im, &value.0)?;
    let mut out = value.clone();
    if zero_re {
        out.0 = BigFloat::new(prec);
    }
    if zero_im {
        out.1 = BigFloat::new(prec);
    }
    Some((
        out,
        Zeroed {
            re: zero_re,
            im: zero_im,
        },
    ))
}

/// The parts [`settle`] set to 0 because their balls contain 0 (not the
/// parts that are exactly 0).
#[derive(Clone, Copy, Debug, Default)]
struct Zeroed {
    re: bool,
    im: bool,
}

// ═══════════════════════════════════════════════════════════════════════════
// Direct `f64` / `Complex64` results (no decimal string in between)
// ═══════════════════════════════════════════════════════════════════════════

/// Decimal digits requested by `Ex::eval_f64` and `Ex::eval_complex64`:
/// 16 digits map to a 128-bit working precision (see [`evalf`]), the same
/// precision [`evalf_f64`] and [`evalf_complex64`] evaluate at.
pub(crate) const F64_DIGITS: u32 = 16;

/// Run `f` with this thread's constants cache (`π`, `e`, `ln 2`, `ln 10`)
/// for an evaluation to `digits` digits: the one cache every route of the
/// evaluator shares — `eval_f64`, `eval_decimal` and the crate's own zero
/// and sign tests.
///
/// `Consts::new()` alone costs ~170 µs and a cold cache recomputes `π` and
/// `ln 2` on every `sin`/`exp` (`exp(2)` at 128 bits: 475 µs cold, 2.5 µs
/// warm), which dominated every `eval_f64` until that route kept its
/// cache; `eval_decimal` created a new one per call until 0.60.1, so `π`
/// to a million digits took 3.3 s every time it was asked for (issue #8).
/// `crate::base::bigfloat::with_consts` says why a warm cache returns the
/// same bits as a cold one, how the caches are kept by band of working
/// precision (a 16-digit evaluation never copies constants a
/// million-digit one left behind), what a nested evaluation finds, and
/// what memory stays in use (`Context::clear_numeric_cache` frees it).
fn with_eval_consts<T>(
    digits: u32,
    f: impl FnOnce(&mut Consts) -> Result<T, SymplexError>,
) -> Result<T, SymplexError> {
    crate::base::bigfloat::with_consts(working_precision(digits), f).map_err(|e| {
        SymplexError::NotImplemented(format!("astro-float constants init failed: {e:?}"))
    })?
}

/// Evaluate `expr` to an arbitrary-precision complex value at the working
/// precision [`evalf`] uses for `digits` decimal digits — the evaluation
/// half of [`evalf`], without the decimal formatting — and how it was
/// settled (`search`: see [`ZeroSearch`]).
///
/// Errors as [`evalf`]: [`SymplexError::PrecisionExhausted`] when the
/// precision exceeds the configured maximum or the result is NaN,
/// [`SymplexError::FreeSymbol`] for an unbound symbol.
fn evalf_value(
    arena: &Arena,
    expr: ExprId,
    digits: u32,
    search: ZeroSearch,
) -> Result<(Complex, Settled), SymplexError> {
    let prec = working_precision(digits);
    // A number below the exponent range is 0 within its bound, and not
    // known to be 0 by that bound: zero to the precision, for the tolerance
    // checks of the crate and for the `f64` routes, whose correctly rounded
    // value it is.
    evalf_adaptive_value(arena, expr, digits, search).map(|a| match a {
        Adaptive::Settled(z, settled) => (z, settled),
        // The part below the range is 0, its correctly rounded `f64`
        // (`evalf_f64` refuses an imaginary one itself).
        Adaptive::PartBelow { value, .. } => (value, Settled::Certified),
        Adaptive::Underflow(_)
        | Adaptive::Beyond(_)
        | Adaptive::Undecided
        | Adaptive::Hidden { .. } => (c_zero(prec), Settled::ZeroToPrecision),
    })
}

/// [`evaluate_adaptive`] for the callers inside the crate, after the checks
/// of [`evalf`] (the configured maximum precision, free symbols).
fn evalf_adaptive_value(
    arena: &Arena,
    expr: ExprId,
    digits: u32,
    search: ZeroSearch,
) -> Result<Adaptive, SymplexError> {
    checked_precision(arena, digits, "evalf")?;

    let free = walk::free_symbols(arena, expr);
    if !free.is_empty() {
        let mut names: Vec<&str> = free
            .iter()
            .filter_map(|&id| match arena.node(id) {
                ExprNode::Symbol(sid) => Some(arena.symbol_name(*sid)),
                _ => None,
            })
            .collect();
        names.sort_unstable();
        if let Some(name) = names.first() {
            debug!(symbol = name, "evalf_value: expression has free symbols");
            return Err(SymplexError::FreeSymbol {
                name: (*name).to_owned(),
            });
        }
    }

    with_eval_consts(digits, |cc| {
        evaluate_adaptive(arena, expr, digits, search, cc)
    })
}

/// [`evalf_value`] for the zero and sign tests of the rest of the crate
/// (`poly::algebraic::is_zero_checked`): the value of the variable-free
/// `expr` to `digits` digits with the [`ZeroSearch`] `search`, and whether
/// it is certified or only zero to the precision reached — a distinction
/// [`evalf_complex`] does not make.
///
/// A number below the exponent range that is certainly not 0 is
/// [`Settled::Certified`] with a stand-in at the bottom of the range that
/// has its sign (its direction, when it is known scaled): its digits are
/// not the value's, its zeroness, sign and realness are.  One of unknown
/// direction, or 0 within the underflow bound and not known to be nonzero,
/// is an error: neither is 0 to a precision the search chose.  Before 0.34
/// both were [`Settled::ZeroToPrecision`], which `solve` (and `polysys`)
/// read as 0: `solve(exp(−10¹⁰), x)` was an identity, every `x` a
/// solution.
///
/// # Errors
///
/// As [`evalf`].
pub(crate) fn evalf_settled(
    arena: &Arena,
    expr: ExprId,
    digits: u32,
    search: ZeroSearch,
) -> Result<(Complex, Settled), SymplexError> {
    let prec = working_precision(digits);
    match evalf_adaptive_value(arena, expr, digits, search)? {
        Adaptive::Settled(z, settled) => Ok((z, settled)),
        // The part within the range has its digits and its sign; the other
        // is read as 0 here (below the precision of the value, as `settle`
        // leaves a part whose ball contains 0).
        Adaptive::PartBelow { value, .. } => Ok((value, Settled::Certified)),
        Adaptive::Beyond(s) => s
            .representative(prec)
            .map(|z| (z, Settled::Certified))
            .ok_or_else(underflow_error),
        Adaptive::Underflow(accuracy::Nonzero::Positive) => Ok((
            (extended::tiny(false, prec), BigFloat::new(prec)),
            Settled::Certified,
        )),
        Adaptive::Underflow(accuracy::Nonzero::Negative) => Ok((
            (extended::tiny(true, prec), BigFloat::new(prec)),
            Settled::Certified,
        )),
        Adaptive::Underflow(accuracy::Nonzero::Unsigned) => Err(underflow_error()),
        Adaptive::Undecided => Err(undecided_error()),
        Adaptive::Hidden { below_range: true } => Err(hidden_error()),
        Adaptive::Hidden { below_range: false } => Err(hidden_deep_error(digits)),
    }
}

/// [`evalf_settled`] with its evidence as a [`Verdict`]: the value is
/// `Certified` when every part it shows carries the digits (an exact zero
/// included: a certified value with both parts 0 is exactly 0), and only
/// `Approximate` when it is zero to the precision reached
/// ([`Settled::ZeroToPrecision`]).  [`evalf_complex`] drops that
/// distinction; a sign, zero or realness that justifies a claim must not
/// (read the verdict with [`Verdict::proven`]).
///
/// A certified part that is 0 next to a nonzero other part is only below
/// the precision of the whole value (see [`settle`]): realness needs
/// `im(x)` evaluated on its own.
///
/// [`Verdict`]: crate::base::truth::Verdict
/// [`Verdict::proven`]: crate::base::truth::Verdict::proven
///
/// # Errors
///
/// As [`evalf_settled`].
pub(crate) fn evalf_verdict(
    arena: &Arena,
    expr: ExprId,
    digits: u32,
    search: ZeroSearch,
) -> Result<crate::base::truth::Verdict<Complex>, SymplexError> {
    use crate::base::truth::{Evidence, Verdict};
    let (z, settled) = evalf_settled(arena, expr, digits, search)?;
    let evidence = match settled {
        Settled::Certified => Evidence::Certified,
        Settled::ZeroToPrecision => Evidence::Approximate,
    };
    Ok(Verdict::new(z, evidence))
}

/// The numeric value of a symbol-free constant read for a decision: the
/// value [`evalf`] prints at `digits` digits, held as its settled parts —
/// no decimal string is formatted and read back.
///
/// Each part is 0 when its error ball contained 0 and otherwise carries
/// `digits` certified significant digits, however small next to the other
/// part ([`settle`] with [`ZeroSearch::Number`]), so the sign of a part that
/// is not 0 is certified ([`re_sign`](Self::re_sign),
/// [`im_sign`](Self::im_sign)).  Its views:
///
/// * the decimal digits of the real part, exactly those [`evalf`] prints,
///   kept unscaled ([`Decimal`]: the value of `exp(1000³)` has a decimal
///   exponent of 4.3·10⁸, whose power of ten cannot be formed);
/// * the `f64` of a real value, correctly rounded from its real part
///   ([`to_f64`](Self::to_f64)) — for display and guides: it carries no
///   error bound.
///
/// Up to 0.57 the crate read the decimal string back (`parse::<f64>`, an
/// exact rational of the decimal), which rounded twice (16 digits, then an
/// `f64`: an ulp away from the value) and made every caller re-derive the
/// imaginary part from the text.  `tests/unit/test_decimal_roundtrip.rs`
/// keeps such round trips from coming back.
#[derive(Clone, Debug)]
pub(crate) struct ConstValue {
    z: Complex,
    settled: Settled,
    digits: u32,
}

impl ConstValue {
    /// The certified sign of the real part: `POSITIVE` or `NEGATIVE` when it
    /// is not 0 (its ball excludes 0), `ZERO` when the whole value is
    /// certified 0, and `REAL` (undecided) for a part that is 0 within its
    /// ball — next to a nonzero imaginary part, or in a value zero only to
    /// the precision reached.
    pub(crate) fn re_sign(&self) -> crate::base::truth::SignSet {
        self.part_sign(&self.z.0)
    }

    /// The certified sign of the imaginary part (as
    /// [`re_sign`](Self::re_sign)).  A part that is 0 next to a nonzero real
    /// part is only below the precision of the value: realness is proved by
    /// `im(x)` evaluated on its own (`realness::imaginary_part_sign`).
    pub(crate) fn im_sign(&self) -> crate::base::truth::SignSet {
        self.part_sign(&self.z.1)
    }

    fn part_sign(&self, part: &BigFloat) -> crate::base::truth::SignSet {
        use crate::base::truth::SignSet;
        if !part.is_zero() {
            return if is_strictly_positive(part) {
                SignSet::POSITIVE
            } else {
                SignSet::NEGATIVE
            };
        }
        if self.settled == Settled::Certified && self.z.0.is_zero() && self.z.1.is_zero() {
            SignSet::ZERO
        } else {
            SignSet::REAL
        }
    }

    /// The real part to the digits [`evalf`] prints, as an unscaled
    /// [`Decimal`]; `None` only for a part without a finite value.
    pub(crate) fn re_decimal(&self) -> Option<Decimal> {
        Decimal::of(&self.z.0, self.digits)
    }

    /// The value as an `f64`, correctly rounded from its real part (`±∞`
    /// beyond the `f64` range, `0.0` below it), for display and guides;
    /// `None` when the value has a nonzero imaginary part (however small:
    /// it is certified) or no finite value.
    pub(crate) fn to_f64(&self) -> Option<f64> {
        if !self.z.1.is_zero() {
            return None;
        }
        finite_part_to_f64(&self.z.0).ok()
    }
}

/// The value [`evalf`] prints for the variable-free `expr` at `digits`
/// digits, as a [`ConstValue`] — the API for reading the numeric value of a
/// constant inside the crate.  The same evaluation as [`evalf`]: a value
/// that is zero to the precision reached is pursued as for a result read as
/// a number ([`ZeroSearch::Number`]) and is then 0, and a number below the
/// exponent range is refused (its sign, when known, is in
/// [`evalf_settled`]).
///
/// # Errors
///
/// As [`evalf`].
pub(crate) fn const_value(
    arena: &Arena,
    expr: ExprId,
    digits: u32,
) -> Result<ConstValue, SymplexError> {
    match evalf_adaptive_value(arena, expr, digits, ZeroSearch::Number)? {
        Adaptive::Settled(z, settled) => Ok(ConstValue { z, settled, digits }),
        Adaptive::PartBelow { imaginary, .. } => Err(part_underflow_error(imaginary)),
        Adaptive::Underflow(_) | Adaptive::Beyond(_) => Err(underflow_error()),
        Adaptive::Undecided => Err(undecided_error()),
        Adaptive::Hidden { below_range: true } => Err(hidden_error()),
        Adaptive::Hidden { below_range: false } => Err(hidden_deep_error(digits)),
    }
}

/// `ln x` for a positive real constant `x`, from a 16-digit evaluation;
/// `None` when `x` has free symbols, does not evaluate, or is not a
/// positive real number.  Accurate far beyond `f64` range (`x = 10⁻⁴⁰⁰`
/// is fine), for callers that need a magnitude estimate.
pub(crate) fn ln_of_positive_constant(arena: &Arena, x: ExprId) -> Option<f64> {
    // A value that is only zero to the precision reached has no logarithm
    // (astro-float's `is_positive` is the sign bit, true for `+0`).  One
    // below the exponent range has, when it is known scaled.
    let z = match evalf_adaptive_value(arena, x, F64_DIGITS, ZeroSearch::Cap).ok()? {
        Adaptive::Settled(z, _) => z,
        Adaptive::Beyond(s) => {
            let real_positive = s.m.1.is_zero() && bf_strictly_positive(&s.m.0);
            return real_positive.then(|| s.ln_abs()).flatten();
        }
        Adaptive::Underflow(_)
        | Adaptive::Undecided
        | Adaptive::Hidden { .. }
        | Adaptive::PartBelow { .. } => return None,
    };
    if !bf_strictly_positive(&z.0) || !is_real_to_digits(&z, F64_DIGITS) {
        return None;
    }
    let e = z.0.exponent()?;
    let mut m = z.0.clone();
    m.set_exponent(0); // m ∈ [1/2, 1)
    let m = f64_of(&m).ok()?;
    Some(m.ln() + f64::from(e) * std::f64::consts::LN_2)
}

/// [`evalf_value`] for the rest of the crate: the variable-free `expr`
/// evaluated to `digits` correct significant digits, as an
/// arbitrary-precision complex value.  An `Ok` value is finite.  A value
/// that is zero to the precision is pursued only to the cap of the digits
/// ([`ZeroSearch::Cap`]): the callers compare against a tolerance and meet
/// true zeros all the time (the integrator's `F′ − f`).
///
/// # Errors
///
/// As [`evalf`].
pub(crate) fn evalf_complex(
    arena: &Arena,
    expr: ExprId,
    digits: u32,
) -> Result<Complex, SymplexError> {
    evalf_value(arena, expr, digits, ZeroSearch::Cap).map(|(z, _)| z)
}

/// Does the evaluation of the variable-free `expr` meet a value proved
/// infinite ([`SymplexError::Infinite`]: a function at a pole of exact
/// arguments, `∞`, `zoo`) — `expr` itself (`elliptic_pi(π, 1)`) or a part
/// its value needs (`1/elliptic_pi(π, 1)`, `√elliptic_pi(π, 1)`)?  `false`
/// when it evaluates, and when it cannot be evaluated for another reason
/// (`erfinv(9/4)`): that is no evidence of an infinity.
pub(crate) fn meets_infinity(arena: &Arena, expr: ExprId) -> bool {
    matches!(
        evalf_settled(arena, expr, F64_DIGITS, ZeroSearch::Cap),
        Err(SymplexError::Infinite { .. })
    )
}

/// `|z|` rounded to an `f64`: `+∞` beyond the `f64` range, `0.0` below it;
/// `None` for NaN.
pub(crate) fn abs_to_f64(z: &Complex) -> Option<f64> {
    f64_of(&crate::base::bigcomplex::c_abs(z, 128, RM)).ok()
}

/// `|a − b|` rounded to an `f64` (see [`abs_to_f64`]), the difference taken
/// exactly enough that two values agreeing to their last digit give a
/// difference at that digit, not 0 or noise from a coarser rounding.
pub(crate) fn distance_to_f64(a: &Complex, b: &Complex) -> Option<f64> {
    let prec = [&a.0, &a.1, &b.0, &b.1]
        .iter()
        .filter_map(|x| x.mantissa_max_bit_len())
        .max()
        .unwrap_or(128)
        .max(128)
        + 64;
    abs_to_f64(&crate::base::bigcomplex::c_sub(a, b, prec, RM))
}

/// Is `z` real to `digits` digits — its imaginary part zero or negligible
/// next to its real part ([`is_negligible_part`])?  An approximate reading
/// for the heuristics of the crate: since 0.53 [`evalf`] prints an
/// imaginary part whenever its ball excludes 0, however small.
pub(crate) fn is_real_to_digits(z: &Complex, digits: u32) -> bool {
    is_negligible_part(&z.1, &z.0, digits)
}

/// One part of a settled result ([`settle`] with [`ZeroSearch::Number`])
/// as an `f64`: `0.0` when it is 0 (its ball contained 0), otherwise the
/// correctly rounded value, however small next to the other part (the
/// parts `format_complex` prints); an infinite part is refused, as the
/// decimal route refuses to parse `oo`.  Up to 0.52 a part negligible
/// next to the other was `0.0` even when certified nonzero:
/// `eval_complex64(Chi(−60))` had the imaginary part `0.0` (it is `π`).
fn finite_part_to_f64(part: &BigFloat) -> Result<f64, SymplexError> {
    if part.is_zero() {
        return Ok(0.0);
    }
    if part.is_inf() {
        return Err(SymplexError::Unevaluable {
            reason: "the value overflowed the arbitrary-precision exponent range".into(),
        });
    }
    f64_of(part)
}

/// Evaluate `expr` to a [`Complex64`] at the precision behind
/// `Ex::eval_complex64` (16 digits, 128 working bits), rounding each part
/// straight from the `BigFloat` result — the string-free counterpart of
/// `evalf(arena, expr, 16)` followed by a parse.  A part whose ball
/// contains 0 is `0.0`, exactly as it is absent from the printed number;
/// a part certified nonzero has its own correctly rounded value, however
/// small next to the other (since 0.53; `Chi(−60)` had the imaginary
/// part `0.0`).
///
/// The direct rounding is correct to the last bit; the decimal route
/// rounded to 16 significant digits first, which can land one ulp away.
///
/// An expression containing a definite integral is served when the `f64`
/// quadrature certifies [`QUADRATURE_F64_DIGITS`] digits (it seldom
/// certifies 16); the `f64` returned is then the quadrature's estimate.
///
/// # Errors
///
/// As [`evalf`]; additionally [`SymplexError::Unevaluable`] when a part is
/// infinite.
pub(crate) fn evalf_complex64(arena: &Arena, expr: ExprId) -> Result<Complex64, SymplexError> {
    let digits = if contains_definite_integral(arena, expr) {
        QUADRATURE_F64_DIGITS
    } else {
        F64_DIGITS
    };
    let (z, _) = evalf_value(arena, expr, digits, ZeroSearch::Number)?;
    let re = finite_part_to_f64(&z.0)?;
    let im = finite_part_to_f64(&z.1)?;
    Ok(Complex64::new(re, im))
}

/// Evaluate `expr` to an `f64` at the precision behind `Ex::eval_f64`
/// (see [`evalf_complex64`]): the real value `evalf(expr, 16)` prints.
///
/// # Errors
///
/// As [`evalf_complex64`]; [`SymplexError::ComputationFailed`] when the
/// value is not real: its imaginary part is certified nonzero (its ball
/// excludes 0), however small it is next to the real part.  Before 0.30 an
/// imaginary part up to `1e-15` in absolute value was dropped:
/// `sqrt(−10⁻⁴⁰)` (`10⁻²⁰·i`) was `0.0`, and `ln(1 + 6·10⁻¹⁶)·(1 +
/// 2·10⁻¹²·i)` lost an imaginary part of `2·10⁻¹²` of its value; up to
/// 0.52 one negligible next to the real part (below its 16th digit) was
/// dropped: `Chi(−60)` (`9.68·10²³ + πi`) was `9.68·10²³`.
pub(crate) fn evalf_f64(arena: &Arena, expr: ExprId) -> Result<f64, SymplexError> {
    let digits = if contains_definite_integral(arena, expr) {
        QUADRATURE_F64_DIGITS
    } else {
        F64_DIGITS
    };
    // An imaginary part certainly not 0 below the exponent range is a
    // nonzero imaginary part like any other (up to 0.58 `1 + i·e^(−10¹⁰)`
    // was `1.0`); a real part below it next to an imaginary part in range is
    // refused for that part below.
    let z = match evalf_adaptive_value(arena, expr, digits, ZeroSearch::Number)? {
        Adaptive::PartBelow {
            imaginary: true, ..
        } => {
            return Err(SymplexError::ComputationFailed {
                operation: "eval_f64",
                reason: "expression has nonzero imaginary part (not 0, below the \
                         arbitrary-precision exponent range); use eval_complex64() for complex \
                         results"
                    .into(),
            });
        }
        Adaptive::Settled(z, _) | Adaptive::PartBelow { value: z, .. } => z,
        Adaptive::Underflow(_)
        | Adaptive::Beyond(_)
        | Adaptive::Undecided
        | Adaptive::Hidden { .. } => c_zero(working_precision(digits)),
    };
    if !z.1.is_zero() {
        // The `f64` of the part, or its decimal where that is 0 or infinite
        // (an imaginary part certified nonzero below or beyond `f64`).
        let im = match f64_of(&z.1) {
            Ok(v)
                if matches!(
                    v.classify(),
                    std::num::FpCategory::Normal | std::num::FpCategory::Subnormal
                ) =>
            {
                v.to_string()
            }
            _ => format_decimal(&z.1, 6).unwrap_or_else(|_| "?".to_owned()),
        };
        return Err(SymplexError::ComputationFailed {
            operation: "eval_f64",
            reason: format!(
                "expression has nonzero imaginary part (im={im}); use eval_complex64() for complex results"
            ),
        });
    }
    finite_part_to_f64(&z.0)
}

/// Evaluate a constant (variable-free) arena expression to `f64`.
///
/// Returns `None` if the expression contains free symbols, produces
/// a complex result, or evaluation fails for any reason.
///
/// The expression is first simplified via [`eval`](crate::transforms::eval::eval)
/// to reduce exact values (e.g., `sin(π) → 0`, `Γ(5) → 24`) before
/// numerical evaluation.
pub(crate) fn eval_const_f64(arena: &mut Arena, expr: ExprId) -> Option<f64> {
    // Simplify exact values first.
    let evaled = crate::transforms::eval::eval(arena, expr);

    // Fast path: if the result is already a rational number, convert directly
    // without invoking the expensive arbitrary-precision machinery.
    if let Some(r) = arena.as_num(evaled) {
        let result = crate::base::numeric::ratio_to_f64(r)?;
        tracing::trace!(result, "eval_const_f64: rational fast path");
        return Some(result);
    }

    // Fall back to arbitrary-precision evaluation to 16 decimal digits, the
    // `f64` rounded straight from the real part (up to 0.57 the 16-digit
    // decimal was printed and parsed, which can land an ulp away).  A value
    // below the exponent range, or zero to the precision reached, is `0.0`,
    // its correctly rounded `f64`; a nonzero imaginary part refuses.
    match evalf_value(arena, evaled, 16, ZeroSearch::Cap) {
        Ok((z, _)) => {
            let result = if z.1.is_zero() {
                finite_part_to_f64(&z.0).ok()
            } else {
                None
            };
            tracing::trace!(?result, "eval_const_f64: evalf path");
            result
        }
        Err(e) => {
            tracing::debug!(?e, "eval_const_f64: evalf failed");
            None
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Node evaluation
// ═══════════════════════════════════════════════════════════════════════════

/// Evaluate a single node given its children's values in the cache.
fn eval_node(
    arena: &Arena,
    id: ExprId,
    cache: Cache<'_>,
    w: &mut Wp<'_>,
) -> Result<Complex, SymplexError> {
    let prec = w.bits();
    let node = arena.node(id);

    // Special functions at non-real arguments, and at real ones where the
    // value is not real (`cspecial.rs`; refused before 0.45).
    if let Some((f, args)) = cspecial::classify(node, cache) {
        let vals = args
            .iter()
            .map(|a| get_cached(cache, *a))
            .collect::<Result<Vec<_>, _>>()?;
        return cspecial::eval(f, &vals, w);
    }

    match node {
        // ── Atoms ──────────────────────────────────────────────────
        ExprNode::Num(nid) => {
            let r = arena.num(*nid);
            Ok((ratio_to_bigfloat(r, prec), BigFloat::new(prec)))
        }

        ExprNode::Pi => Ok((w.cc.pi(prec, RM).clone(), BigFloat::new(prec))),

        ExprNode::E => Ok((w.cc.e(prec, RM).clone(), BigFloat::new(prec))),

        ExprNode::ImaginaryUnit => Ok((BigFloat::new(prec), BigFloat::from_i32(1, prec))),

        // ── Named constants (arbitrary precision) ───────────────────────────
        ExprNode::EulerGamma => {
            debug!(prec, "evalf: EulerGamma via Brent–McMillan");
            let g = arb_euler_gamma(w)?;
            Ok((g, BigFloat::new(prec)))
        }
        ExprNode::Catalan => {
            debug!(prec, "evalf: Catalan via Ramanujan series");
            let g = arb_catalan(w)?;
            Ok((g, BigFloat::new(prec)))
        }
        ExprNode::GoldenRatio => {
            // φ = (1 + √5)/2, exactly via a single square root.
            let wp = prec + 16;
            let five = BigFloat::from_i32(5, wp);
            let sqrt5 = five.sqrt(wp, RM);
            let one = BigFloat::from_i32(1, wp);
            let two = BigFloat::from_i32(2, wp);
            let phi = one.add(&sqrt5, wp, RM).div(&two, prec, RM);
            Ok((phi, BigFloat::new(prec)))
        }

        // ── Complex analysis ────────────────────────────────────────────
        ExprNode::Re(inner) => {
            let val = get_cached(cache, *inner)?;
            Ok((val.0.clone(), BigFloat::new(prec)))
        }
        ExprNode::Im(inner) => {
            let val = get_cached(cache, *inner)?;
            Ok((val.1.clone(), BigFloat::new(prec)))
        }
        ExprNode::Conjugate(inner) => {
            let val = get_cached(cache, *inner)?;
            Ok((val.0.clone(), val.1.neg()))
        }
        ExprNode::Arg(inner) => {
            let val = get_cached(cache, *inner)?;
            if val.0.is_zero() && val.1.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "arg(0) is undefined".into(),
                });
            }
            Ok((atan2_bf(&val.1, &val.0, w), BigFloat::new(prec)))
        }

        // ── Special functions (0.2) ──────────────────────────────────────
        ExprNode::Si(inner) => {
            let val = get_cached(cache, *inner)?;
            if !val.1.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "Si of complex argument not yet supported in evalf".into(),
                });
            }
            debug!(prec, "evalf: Si via series/asymptotic");
            let (si, _ci) = arb_si_ci(&val.0, true, w)?;
            Ok((si, BigFloat::new(prec)))
        }
        ExprNode::Ci(inner) => {
            let val = get_cached(cache, *inner)?;
            if !val.1.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "Ci of complex argument not yet supported in evalf".into(),
                });
            }
            if val.0.is_zero() {
                return Err(infinite("Ci(0) is zoo"));
            }
            debug!(prec, "evalf: Ci via series/asymptotic");
            let x_abs = val.0.abs();
            let (_si, ci) = arb_si_ci(&x_abs, false, w)?;
            if val.0.is_negative() {
                // Ci(-x) = Ci(x) + iπ (principal branch of the logarithm).
                Ok((ci, w.cc.pi(prec, RM).clone()))
            } else {
                Ok((ci, BigFloat::new(prec)))
            }
        }
        ExprNode::Ei(inner) => {
            let val = get_cached(cache, *inner)?;
            if !val.1.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "Ei of complex argument not yet supported in evalf".into(),
                });
            }
            if val.0.is_zero() {
                return Err(infinite("Ei(0) is -∞"));
            }
            debug!(prec, "evalf: Ei via series/asymptotic");
            let r = arb_ei(&val.0, w)?;
            Ok((r, BigFloat::new(prec)))
        }
        ExprNode::Li(inner) => {
            let val = get_cached(cache, *inner)?;
            if !val.1.is_zero() || val.0.is_negative() {
                return Err(SymplexError::Unevaluable {
                    reason: "li of complex or negative argument not yet supported in evalf".into(),
                });
            }
            if val.0.is_zero() {
                return Ok(c_zero(prec));
            }
            let wp = prec + 32;
            let ln_x = val.0.ln(wp, RM, w.cc);
            if ln_x.is_zero() {
                return Err(infinite("li(1) is -∞"));
            }
            debug!(prec, "evalf: li via Ei(ln x)");
            let r = arb_ei(&ln_x, w)?;
            // `ln x` is rounded at `wp` bits and `Ei′(ln x) = x/ln x`: the
            // value is within `x·2^(1 − wp)` of `li(x)`, whatever its size
            // (next to its zero `1.4513…`, where up to 0.59 the 166-bit
            // dyadic nearest the zero was wrong from the 28th digit of 30).
            w.charge(accuracy::part_lg(&val.0) + 1.0 - wp as f64);
            Ok((r, BigFloat::new(prec)))
        }
        ExprNode::Zeta(inner) => {
            let val = get_cached(cache, *inner)?;
            if !val.1.is_zero() {
                return zeta::zeta_complex(val, w);
            }
            debug!(prec, "evalf: zeta via Borwein / functional equation");
            let r = arb_zeta(&val.0, w)?;
            Ok((r, BigFloat::new(prec)))
        }
        ExprNode::Polygamma(n_id, x_id) => {
            let n_val = get_cached(cache, *n_id)?;
            let x_val = get_cached(cache, *x_id)?;
            if !n_val.1.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "polygamma order must be a non-negative integer".into(),
                });
            }
            // The order exactly an integer (its bound is exact too, see
            // `eval_ball`): up to 0.53 an order within `10⁻¹²` of one was
            // taken for it, `polygamma(1 + 10⁻²⁰, 2)` was `ψ′(2)` while
            // `polygamma(1/2, 2)` was refused.
            if n_val.0.is_negative() || !(n_val.0.is_zero() || n_val.0.is_int()) {
                return Err(SymplexError::Unevaluable {
                    reason: "polygamma order must be a non-negative integer".into(),
                });
            }
            let n_round = f64_of(&n_val.0)?;
            // A non-real argument: the Hurwitz zeta function (`zeta.rs`;
            // refused before 0.44).
            if !x_val.1.is_zero() {
                if n_round > f64::from(zeta::MAX_POLYGAMMA_ORDER) {
                    return Err(unevaluable(
                        "polygamma of an order beyond 10^4 at a complex argument not supported in evalf",
                    ));
                }
                return zeta::polygamma_complex(n_round as u32, x_val, w);
            }
            // A large order: `(−1)^{n+1}·n!·ζ(n + 1, x)` in log space (before
            // 0.30 refused beyond 10⁴, "order must be a non-negative integer",
            // although `eval` leaves `polygamma(10⁵, 1/2)` symbolic).
            if n_round > 10_000.0 {
                debug!(
                    prec,
                    n = n_round,
                    "evalf: polygamma of large order via Hurwitz zeta"
                );
                let n_exact = bigfloat::from_f64(n_round, 64);
                let r = factorials::polygamma_large(&n_exact, &x_val.0, w)?;
                return Ok((r, BigFloat::new(prec)));
            }
            let n = n_round as u32;
            if n == 0 {
                let r = arb_digamma(&x_val.0, w)?;
                return Ok((r, BigFloat::new(prec)));
            }
            debug!(
                prec,
                n, "evalf: polygamma via recurrence + asymptotic series"
            );
            let r = arb_polygamma(n, &x_val.0, prec, w.cc)?;
            Ok((r, BigFloat::new(prec)))
        }
        ExprNode::KroneckerDelta(i_id, j_id) => {
            let iv = get_cached(cache, *i_id)?;
            let jv = get_cached(cache, *j_id)?;
            let d_re = iv.0.sub(&jv.0, prec, RM);
            let d_im = iv.1.sub(&jv.1, prec, RM);
            if d_re.is_zero() && d_im.is_zero() {
                Ok(c_one(prec))
            } else {
                Ok(c_zero(prec))
            }
        }

        // Evaluated with their own error bounds by `eval_ball`, which never
        // hands them here; without the children's bounds a `Piecewise`
        // decision is never certain.
        ExprNode::PhysicalConstant(..)
        | ExprNode::Sum(..)
        | ExprNode::Product_(..)
        | ExprNode::Piecewise(_)
        | ExprNode::RootOf(..)
        | ExprNode::RootSum(..)
        | ExprNode::DefiniteIntegral(..) => {
            eval_ball(arena, id, cache.values_only(), w).map(|CBall { mid: v, err: _ }| v)
        }

        ExprNode::Infinity | ExprNode::NegInfinity | ExprNode::ComplexInfinity => {
            Err(infinite("cannot evaluate infinity to finite precision"))
        }

        // x! = Γ(x + 1) for every real x off the poles (x = −1, −2, …);
        // `eval()` already reduces non-negative integer literals exactly.
        ExprNode::Factorial(x_id) => {
            let x_val = get_cached(cache, *x_id)?;
            if !x_val.1.is_zero() {
                let x1 = (
                    x_val
                        .0
                        .add(&BigFloat::from_i32(1, 64), exact_bits(&x_val.0, prec), RM),
                    x_val.1.clone(),
                );
                return zeta::gamma_complex(&x1, w);
            }
            debug!(
                prec,
                "evalf: factorial via arbitrary-precision Gamma(x + 1)"
            );
            let x_plus_1 = x_val
                .0
                .add(&BigFloat::from_i32(1, 64), exact_bits(&x_val.0, prec), RM);
            let r = arb_gamma_real(&x_plus_1, w)?;
            Ok((r, BigFloat::new(prec)))
        }

        ExprNode::Binomial(n_id, k_id) => {
            debug!(prec, "evalf: Binomial via arbitrary-precision Gamma");
            let n_val = get_cached(cache, *n_id)?;
            let k_val = get_cached(cache, *k_id)?;
            if !n_val.1.is_zero() || !k_val.1.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "Binomial of complex arguments not yet supported in evalf".into(),
                });
            }
            // Integer `0 ≤ n < k`: C(n, k) = 0 (the Γ formula below would
            // hit the pole of Γ(n − k + 1)).  Integers exactly: before 0.29
            // `n` within 10⁻¹² of one counted, and `binomial(2 + 10⁻¹³, 3)`
            // was 0 (truly 3.3·10⁻¹⁴).
            if n_val.0.is_int() && k_val.0.is_int() && !n_val.0.is_negative() && k_val.0 > n_val.0 {
                return Ok((BigFloat::new(prec), BigFloat::new(prec)));
            }
            // Where the Γ formula meets a pole, its limit (mpmath's
            // `gammaprod`): before 0.30 `binomial(−7, 2500)` was refused,
            // "Gamma at non-positive integer pole", although it is
            // `C(2506, 6)`.
            if factorials::binomial_meets_pole(&n_val.0, &k_val.0, prec) {
                let r = factorials::binomial(&n_val.0, &k_val.0, w)?;
                return Ok((r, BigFloat::new(prec)));
            }
            // C(n, k) = Gamma(n+1) / (Gamma(k+1) * Gamma(n-k+1)), the shifted
            // arguments exact.
            let one_bf = BigFloat::from_i32(1, 64);
            let p = exact_bits(&n_val.0, prec).max(exact_bits(&k_val.0, prec)) + 64;
            let n_plus_1 = n_val.0.add(&one_bf, p, RM);
            let k_plus_1 = k_val.0.add(&one_bf, p, RM);
            let n_minus_k = n_val.0.sub(&k_val.0, p, RM);
            let n_minus_k_plus_1 = n_minus_k.add(&one_bf, p, RM);

            let g_numer = arb_gamma_real(&n_plus_1, w)?;
            let g_k = arb_gamma_real(&k_plus_1, w)?;
            let g_nmk = arb_gamma_real(&n_minus_k_plus_1, w)?;
            let g_denom = g_k.mul(&g_nmk, prec, RM);
            if g_denom.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "Binomial denominator is zero (pole in Gamma)".into(),
                });
            }
            let result = g_numer.div(&g_denom, prec, RM);
            // `Γ(n + 1)` and `Γ(n − k + 1)` beyond the exponent range for a
            // huge `n` (their quotient NaN, and before 0.31
            // `lambertw(binomial(10⁵⁰, 2), −1)` was refused): for a small
            // integer `k ≥ 0`, `n(n − 1)⋯(n − k + 1)/k!` instead.
            if (result.is_nan() || result.is_inf())
                && (k_val.0.is_zero() || k_val.0.is_int())
                && !k_val.0.is_negative()
                && let Some(k) = f64_of(&k_val.0).ok().filter(|k| *k <= 64.0)
            {
                let k = k as i32;
                let wp = prec + 16;
                let mut product = BigFloat::from_i32(1, wp);
                for j in 0..k {
                    let factor = n_val.0.sub(&BigFloat::from_i32(j, 64), p, RM);
                    product = product.mul(&factor, wp, RM);
                }
                let k_factorial = (1..=k).fold(BigFloat::from_i32(1, wp), |acc, j| {
                    acc.mul(&BigFloat::from_i32(j, 64), wp, RM)
                });
                return Ok((product.div(&k_factorial, prec, RM), BigFloat::new(prec)));
            }
            // Any other quotient of Gamma values beyond the range: in log
            // space ([`factorials::gamma_ratio`]).  Up to 0.56 the NaN went
            // on and was refused (`PrecisionExhausted`): the mass function
            // `C(k + r − 1, k)·pʳ(1 − p)ᵏ` of `NegativeBinomial(10⁷/999999,
            // 10⁻⁶)` at `k = 1000000100`, `1.398·10⁻⁴¹⁹`.
            if result.is_nan() || result.is_inf() {
                let r = factorials::gamma_ratio(&[n_plus_1], &[k_plus_1, n_minus_k_plus_1], w)?;
                return Ok((r, BigFloat::new(prec)));
            }
            Ok((result, BigFloat::new(prec)))
        }

        ExprNode::Gamma(inner) => {
            let val = get_cached(cache, *inner)?;
            if !val.1.is_zero() {
                debug!(prec, "evalf: Gamma of a complex argument");
                return zeta::gamma_complex(val, w);
            }
            debug!(prec, "evalf: Gamma via Stirling series");
            let result = arb_gamma_real(&val.0, w)?;
            Ok((result, BigFloat::new(prec)))
        }

        // The analytic continuation of ln Γ (SymPy's `loggamma`): off the
        // real axis by `loggamma`, on it by `arb_log_gamma` (real for
        // x > 0, `ln|Γ(x)| − iπ⌈−x⌉` on the cut).  Before 0.30 a complex
        // argument was refused and the value left of 0 was `ln|Γ(x)|`.
        ExprNode::LogGamma(inner) => {
            let val = get_cached(cache, *inner)?;
            if !val.1.is_zero() {
                debug!(prec, "evalf: LogGamma of a complex argument");
                return loggamma::loggamma_complex(val, w);
            }
            debug!(prec, "evalf: LogGamma via Stirling series");
            arb_log_gamma(&val.0, w)
        }

        ExprNode::Digamma(inner) => {
            let val = get_cached(cache, *inner)?;
            if !val.1.is_zero() {
                return zeta::polygamma_complex(0, val, w);
            }
            let result = arb_digamma(&val.0, w)?;
            Ok((result, BigFloat::new(prec)))
        }

        ExprNode::Erf(inner) => {
            let val = get_cached(cache, *inner)?;
            if !val.1.is_zero() {
                return cerf::erf_family(cerf::Kind::Erf, val, w);
            }
            debug!(prec, "evalf: erf via arbitrary-precision series");
            let result = arb_erf(&val.0, w)?;
            Ok((result, BigFloat::new(prec)))
        }

        ExprNode::Erfc(inner) => {
            let val = get_cached(cache, *inner)?;
            if !val.1.is_zero() {
                return cerf::erf_family(cerf::Kind::Erfc, val, w);
            }
            debug!(prec, "evalf: erfc via arbitrary-precision series");
            let result = arb_erfc(&val.0, w)?;
            Ok((result, BigFloat::new(prec)))
        }

        ExprNode::LambertW(_) => eval_lambertw_node(arena, node, cache, w),

        ExprNode::Beta(a_id, b_id) => {
            let a_val = get_cached(cache, *a_id)?;
            let b_val = get_cached(cache, *b_id)?;
            if !a_val.1.is_zero() || !b_val.1.is_zero() {
                debug!(prec, "evalf: Beta of complex arguments");
                return zeta::beta_complex(a_val, b_val, w);
            }
            // Positive shapes whose Gamma values may leave the exponent range
            // (`Γ(x)` overflows from `x ≈ 7.8·10⁷` on): `exp(ln B)`, the
            // `ln Γ` differences carrying the bits of their magnitude.  Up to
            // 0.56 `B(2.694, 6.062·10¹⁸) = 3.858·10⁻⁵¹` was refused
            // (`Γ(b)/Γ(a + b)` of two overflowing values), and with it the
            // density of a Student t with `ν ≥ 10⁸` degrees of freedom, and `B(7/20, 1.25·10⁹)`, the
            // constant of the `F(7/10, 2.5·10⁹)` density.
            if bf_strictly_positive(&a_val.0)
                && bf_strictly_positive(&b_val.0)
                && f64_of(&a_val.0.add(&b_val.0, 64, RM))? >= BETA_VIA_LN_MIN
            {
                debug!(prec, "evalf: Beta via ln Beta");
                return Ok((beta_wp(&a_val.0, &b_val.0, w)?, BigFloat::new(prec)));
            }
            // Beta(a,b) = Gamma(a)*Gamma(b)/Gamma(a+b) at arbitrary precision.
            debug!(prec, "evalf: Beta via arbitrary-precision Gamma");
            let ga = arb_gamma_real(&a_val.0, w)?;
            let gb = arb_gamma_real(&b_val.0, w)?;
            let p = exact_bits(&a_val.0, prec).max(exact_bits(&b_val.0, prec)) + 64;
            let a_plus_b = a_val.0.add(&b_val.0, p, RM);
            let gab = arb_gamma_real(&a_plus_b, w)?;
            if gab.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "Beta function: Gamma(a+b) is zero".into(),
                });
            }
            let numer = ga.mul(&gb, prec, RM);
            let result = numer.div(&gab, prec, RM);
            Ok((result, BigFloat::new(prec)))
        }

        ExprNode::NaN => Err(SymplexError::Unevaluable {
            reason: "cannot evaluate NaN".into(),
        }),

        ExprNode::Symbol(sid) => {
            let name = arena.symbol_name(*sid);
            Err(SymplexError::FreeSymbol {
                name: name.to_owned(),
            })
        }

        // ── Add ────────────────────────────────────────────────────
        ExprNode::Add(children) => {
            let mut sum = c_zero(prec);
            for &child in children.iter() {
                let val = get_cached(cache, child)?;
                sum = c_add(&sum, val, prec, RM);
            }
            Ok(sum)
        }

        // ── Mul ────────────────────────────────────────────────────
        ExprNode::Mul(children) => {
            let mut product = c_one(prec);
            for &child in children.iter() {
                let val = get_cached(cache, child)?;
                product = c_mul(&product, val, prec, RM);
            }
            Ok(product)
        }

        // ── Pow ────────────────────────────────────────────────────
        ExprNode::Pow(base, exp) => {
            let b = get_cached(cache, *base)?;
            let e = get_cached(cache, *exp)?;

            let b_is_real = b.1.is_zero();
            let e_is_real = e.1.is_zero();

            // Optimization: use sqrt for x^(1/2) with non-negative real base.
            if b_is_real
                && !b.0.is_negative()
                && let ExprNode::Num(nid) = arena.node(*exp)
            {
                let r = arena.num(*nid);
                if *r == Ratio::new(BigInt::from(1), BigInt::from(2)) {
                    return Ok((b.0.sqrt(prec, RM), BigFloat::new(prec)));
                }
            }

            // Optimization: small integer exponents.
            if let Some(n) = try_as_small_int(arena, *exp) {
                if b_is_real {
                    // Real base with integer exponent: use real powi.
                    if n >= 0 {
                        return Ok((b.0.powi(n as usize, prec, RM), BigFloat::new(prec)));
                    } else {
                        // `unsigned_abs`: `-n` overflows at `n = i32::MIN`
                        // (`2^(−2³¹)`, the tail of `Exponential(ln 2)` at
                        // `2³¹`; a panic with overflow checks up to 0.56).
                        let pow_pos = b.0.powi(n.unsigned_abs() as usize, prec, RM);
                        let one_bf = BigFloat::from_i32(1, prec);
                        return Ok((one_bf.div(&pow_pos, prec, RM), BigFloat::new(prec)));
                    }
                } else {
                    // Complex base with integer exponent: use c_powi.
                    if n >= 0 {
                        return Ok(c_powi(b, n as usize, prec, RM));
                    } else {
                        let pow_pos = c_powi(b, (-n) as usize, prec, RM);
                        let one_c = c_one(prec);
                        return Ok(c_div(&one_c, &pow_pos, prec, RM));
                    }
                }
            }

            // 0^e: 0 for a real e > 0, infinite for a real e < 0, no value
            // for a non-real e (SymPy's `0**I`, mpmath's `mpc(0)**mpc(1,1)`:
            // nan), as `canon_pow` folds it.  Before 0.30 `0^i` and
            // `0^(−1 + i)` printed `0` (and `0^π` was refused).
            if b.0.is_zero() && b.1.is_zero() && !(e.0.is_zero() && e.1.is_zero()) {
                if e.1.is_zero() && is_strictly_positive(&e.0) {
                    return Ok(c_zero(prec));
                }
                // `0^e` for a negative `e` is `zoo`.
                return Err(if e.1.is_zero() {
                    infinite("cannot evaluate infinity to finite precision")
                } else {
                    unevaluable("0 to a non-real power has no value")
                });
            }

            // Real positive base with real exponent: exp(e·ln b).
            // (`BigFloat::pow` is avoided — it hangs on exactly
            // representable results such as 4^(1/2); see `bf_pow`.)
            if b_is_real && e_is_real && is_sign_positive(&b.0) {
                return Ok((bf_pow(&b.0, &e.0, w), BigFloat::new(prec)));
            }

            // Negative real base, exact integer exponent beyond `i32`:
            // `(−r)^n = (−1)ⁿ·rⁿ`, real, as the bound of this node says (an
            // exact imaginary part: `accuracy::power_bound`).  Through
            // `exp(n·Log b)` the imaginary part was the rounding residue of
            // `n·π` under that exact bound, a "certified" nonzero part, which
            // the printer shows since 0.53: `(−(1 + 2⁻⁵²))^(2⁵³ + 1)` came out
            // `−7.389056099 + 2.814264205e-22*i` (up to 0.52 the part was
            // dropped as negligible next to the real part).
            if b_is_real
                && b.0.is_negative()
                && e_is_real
                && e.0.is_int()
                && cache.err(exp).is_exact()
            {
                let magnitude = bf_pow(&b.0.abs(), &e.0, w);
                let mut half = e.0.clone();
                let even = match half.exponent() {
                    Some(x) => {
                        half.set_exponent(x - 1);
                        half.is_int()
                    }
                    None => true,
                };
                return Ok((
                    if even { magnitude } else { magnitude.neg() },
                    BigFloat::new(prec),
                ));
            }

            // Negative real base, half-integer exponent n/2:
            // (−r)^(n/2) = r^(n/2)·iⁿ exactly (n odd, so ±i).  Through
            // exp(½·(ln r + iπ)) the real part is r^(n/2)·cos(nπ/2), a rounding
            // residue whose sign then picks the side of the next branch cut.
            if b_is_real
                && b.0.is_negative()
                && let ExprNode::Num(nid) = arena.node(*exp)
                && *arena.num(*nid).denom() == BigInt::from(2)
            {
                let n = arena.num(*nid).numer();
                let r = b.0.abs();
                let magnitude = bf_pow(&r, &e.0, w);
                let i_power_is_plus_i =
                    num_integer::Integer::mod_floor(n, &BigInt::from(4)) == BigInt::from(1);
                let im = if i_power_is_plus_i {
                    magnitude
                } else {
                    magnitude.neg()
                };
                return Ok((BigFloat::new(prec), im));
            }

            // General complex power: b^e = exp(e * ln(b)), with the angle
            // `Im(e·ln b)` checked as `exp` checks its argument's: reducing
            // a huge inexact angle mod 2π needs π to as many bits, and
            // `(−2)^(−exp(10⁶))` (angle `≈ 10⁴³⁴²⁹⁴`) never finished.
            if (b.0.is_zero() && b.1.is_zero()) || (e.0.is_zero() && e.1.is_zero()) {
                return Ok(c_pow(b, e, w));
            }
            let product = c_mul(e, &c_ln(b, w), prec, RM);
            check_trig_arg(&product.1, 0, prec, arena)?;
            Ok(c_exp(&product, w))
        }

        // ── Neg ────────────────────────────────────────────────────
        ExprNode::Neg(inner) => {
            let val = get_cached(cache, *inner)?;
            Ok(c_neg(val))
        }

        // ── Trig ───────────────────────────────────────────────────
        ExprNode::Sin(inner) => {
            let val = get_cached(cache, *inner)?;
            let (arg, extra) = trig_arg(arena, *inner, val, prec);
            check_trig_arg(&arg, extra, prec, arena)?;
            if val.1.is_zero() {
                Ok((arg.sin(prec, RM, w.cc), BigFloat::new(prec)))
            } else {
                Ok(c_sin(val, w))
            }
        }

        ExprNode::Cos(inner) => {
            let val = get_cached(cache, *inner)?;
            let (arg, extra) = trig_arg(arena, *inner, val, prec);
            check_trig_arg(&arg, extra, prec, arena)?;
            if val.1.is_zero() {
                Ok((arg.cos(prec, RM, w.cc), BigFloat::new(prec)))
            } else {
                Ok(c_cos(val, w))
            }
        }

        ExprNode::Tan(inner) => {
            let val = get_cached(cache, *inner)?;
            let (arg, extra) = trig_arg(arena, *inner, val, prec);
            check_trig_arg(&arg, extra, prec, arena)?;
            if val.1.is_zero() {
                Ok((arg.tan(prec, RM, w.cc), BigFloat::new(prec)))
            } else {
                let s = c_sin(val, w);
                let c = c_cos(val, w);
                Ok(c_div(&s, &c, prec, RM))
            }
        }

        // ── Exp / Ln ───────────────────────────────────────────────
        ExprNode::Exp(inner) => {
            let val = get_cached(cache, *inner)?;
            if val.1.is_zero() {
                Ok((val.0.exp(prec, RM, w.cc), BigFloat::new(prec)))
            } else {
                check_trig_arg(&val.1, 0, prec, arena)?;
                Ok(c_exp(val, w))
            }
        }

        ExprNode::Ln(inner) => {
            let val = get_cached(cache, *inner)?;
            if val.1.is_zero() && is_sign_positive(&val.0) {
                Ok((val.0.ln(prec, RM, w.cc), BigFloat::new(prec)))
            } else {
                Ok(c_ln(val, w))
            }
        }

        // ── Abs ────────────────────────────────────────────────────
        ExprNode::Abs(inner) => {
            let val = get_cached(cache, *inner)?;
            if val.1.is_zero() {
                Ok((val.0.abs(), BigFloat::new(prec)))
            } else {
                Ok((c_abs(val, prec, RM), BigFloat::new(prec)))
            }
        }

        // ── Inverse trig ──────────────────────────────────────────
        // A real argument outside the real domain (`asin 2`, `acosh ½`,
        // `atanh 2`) takes the complex formula, which is the principal value
        // (mpmath's, and so SymPy's `N`); the real routines would give NaN.
        ExprNode::Asin(inner) => {
            let val = get_cached(cache, *inner)?;
            if val.1.is_zero() {
                if abs_le_one(&val.0, prec) {
                    return Ok((val.0.asin(prec, RM, w.cc), BigFloat::new(prec)));
                }
                // |x| > 1: sign(x)·π/2 − i·sign(x)·acosh|x|.
                let half_pi = half_pi_bf(w);
                let ach = val.0.abs().acosh(prec, RM, w.cc);
                return Ok(if val.0.is_negative() {
                    (half_pi.neg(), ach)
                } else {
                    (half_pi, ach.neg())
                });
            }
            Ok(c_asin(val, w))
        }

        ExprNode::Acos(inner) => {
            let val = get_cached(cache, *inner)?;
            if val.1.is_zero() {
                if abs_le_one(&val.0, prec) {
                    return Ok((val.0.acos(prec, RM, w.cc), BigFloat::new(prec)));
                }
                // x > 1: i·acosh x;  x < −1: π − i·acosh|x|.
                let ach = val.0.abs().acosh(prec, RM, w.cc);
                return Ok(if val.0.is_negative() {
                    (w.cc.pi(prec, RM).clone(), ach.neg())
                } else {
                    (BigFloat::new(prec), ach)
                });
            }
            Ok(c_acos(val, w))
        }

        ExprNode::Atan(inner) => {
            let val = get_cached(cache, *inner)?;
            if val.1.is_zero() {
                Ok((val.0.atan(prec, RM, w.cc), BigFloat::new(prec)))
            } else {
                Ok(c_atan(val, w))
            }
        }

        // ── Atan2 ─────────────────────────────────────────────────
        ExprNode::Atan2(y_id, x_id) => {
            let yv = get_cached(cache, *y_id)?;
            let xv = get_cached(cache, *x_id)?;
            // Both args must be real for a real atan2 result.
            if yv.1.is_zero() && xv.1.is_zero() {
                Ok((atan2_bf(&yv.0, &xv.0, w), BigFloat::new(prec)))
            } else {
                // Complex fallback: atan2 is not standard for complex args;
                // compute as -i * ln((x + iy) / sqrt(x² + y²))
                // but for now just use the real parts as a best-effort.
                Err(SymplexError::Unevaluable {
                    reason: "atan2 with complex arguments is not supported".into(),
                })
            }
        }

        // ── Hyperbolic ────────────────────────────────────────────
        ExprNode::Sinh(inner) => {
            let val = get_cached(cache, *inner)?;
            if val.1.is_zero() {
                Ok((val.0.sinh(prec, RM, w.cc), BigFloat::new(prec)))
            } else {
                check_trig_arg(&val.1, 0, prec, arena)?;
                Ok(c_sinh(val, w))
            }
        }

        ExprNode::Cosh(inner) => {
            let val = get_cached(cache, *inner)?;
            if val.1.is_zero() {
                Ok((val.0.cosh(prec, RM, w.cc), BigFloat::new(prec)))
            } else {
                check_trig_arg(&val.1, 0, prec, arena)?;
                Ok(c_cosh(val, w))
            }
        }

        ExprNode::Tanh(inner) => {
            let val = get_cached(cache, *inner)?;
            if val.1.is_zero() {
                Ok((val.0.tanh(prec, RM, w.cc), BigFloat::new(prec)))
            } else {
                check_trig_arg(&val.1, 0, prec, arena)?;
                let s = c_sinh(val, w);
                let c = c_cosh(val, w);
                Ok(c_div(&s, &c, prec, RM))
            }
        }

        // ── Inverse hyperbolic ────────────────────────────────────
        ExprNode::Asinh(inner) => {
            let val = get_cached(cache, *inner)?;
            if val.1.is_zero() {
                Ok((val.0.asinh(prec, RM, w.cc), BigFloat::new(prec)))
            } else {
                Ok(c_asinh(val, w))
            }
        }

        ExprNode::Acosh(inner) => {
            let val = get_cached(cache, *inner)?;
            if val.1.is_zero() {
                let one = BigFloat::from_i32(1, prec);
                if val.0 >= one {
                    return Ok((val.0.acosh(prec, RM, w.cc), BigFloat::new(prec)));
                }
                // −1 ≤ x < 1: i·acos x;  x < −1: acosh|x| + iπ.
                return Ok(if val.0 >= one.neg() {
                    (BigFloat::new(prec), val.0.acos(prec, RM, w.cc))
                } else {
                    (val.0.abs().acosh(prec, RM, w.cc), w.cc.pi(prec, RM).clone())
                });
            }
            Ok(c_acosh(val, w))
        }

        ExprNode::Atanh(inner) => {
            let val = get_cached(cache, *inner)?;
            if val.1.is_zero() {
                if abs_le_one(&val.0, prec) {
                    return Ok((val.0.atanh(prec, RM, w.cc), BigFloat::new(prec)));
                }
                // |x| > 1: atanh(1/x) − sign(x)·iπ/2.
                let recip = BigFloat::from_i32(1, prec).div(&val.0, prec, RM);
                let half_pi = half_pi_bf(w);
                let im = if val.0.is_negative() {
                    half_pi
                } else {
                    half_pi.neg()
                };
                return Ok((recip.atanh(prec, RM, w.cc), im));
            }
            Ok(c_atanh(val, w))
        }

        // ── Sign ───────────────────────────────────────────────────
        ExprNode::Sign(inner) => {
            let val = get_cached(cache, *inner)?;
            if val.1.is_zero() {
                // Real case: sign returns -1, 0, or 1
                if val.0.is_zero() {
                    Ok(c_zero(prec))
                } else if val.0.is_negative() {
                    Ok((BigFloat::from_i32(-1, prec), BigFloat::new(prec)))
                } else {
                    Ok((BigFloat::from_i32(1, prec), BigFloat::new(prec)))
                }
            } else {
                // Complex case: sign(z) = z / |z|
                let modulus = c_abs(val, prec, RM);
                if modulus.is_zero() {
                    Ok(c_zero(prec))
                } else {
                    let denom = (modulus, BigFloat::new(prec));
                    Ok(c_div(val, &denom, prec, RM))
                }
            }
        }

        // ── Floor ──────────────────────────────────────────────────
        ExprNode::Floor(inner) => {
            let val = get_cached(cache, *inner)?;
            if !val.1.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "floor of complex number is not defined".into(),
                });
            }
            // BigFloat::int() returns the integer part (truncation toward zero).
            // floor(x) = x if x is integer, else truncate toward -infinity.
            let truncated = val.0.int();
            let result = if val.0.is_negative() && truncated != val.0 {
                // For negative non-integers, floor = trunc - 1
                truncated.sub(&BigFloat::from_i32(1, prec), prec, RM)
            } else {
                truncated
            };
            Ok((result, BigFloat::new(prec)))
        }

        // ── Ceiling ────────────────────────────────────────────────
        ExprNode::Ceiling(inner) => {
            let val = get_cached(cache, *inner)?;
            if !val.1.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "ceiling of complex number is not defined".into(),
                });
            }
            let truncated = val.0.int();
            let result = if is_strictly_positive(&val.0) && truncated != val.0 {
                // For positive non-integers, ceil = trunc + 1
                truncated.add(&BigFloat::from_i32(1, prec), prec, RM)
            } else {
                truncated
            };
            Ok((result, BigFloat::new(prec)))
        }

        // ── Min ────────────────────────────────────────────────────
        ExprNode::Min(children) => {
            if children.is_empty() {
                return Err(SymplexError::Unevaluable {
                    reason: "min of empty set".into(),
                });
            }
            let mut best = get_cached(cache, children[0])?.clone();
            if !best.1.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "min requires real arguments".into(),
                });
            }
            for &child in &children[1..] {
                let val = get_cached(cache, child)?;
                if !val.1.is_zero() {
                    return Err(SymplexError::Unevaluable {
                        reason: "min requires real arguments".into(),
                    });
                }
                if val.0.sub(&best.0, prec, RM).is_negative() {
                    best = val.clone();
                }
            }
            Ok(best)
        }

        // ── Max ────────────────────────────────────────────────────
        ExprNode::Max(children) => {
            if children.is_empty() {
                return Err(SymplexError::Unevaluable {
                    reason: "max of empty set".into(),
                });
            }
            let mut best = get_cached(cache, children[0])?.clone();
            if !best.1.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "max requires real arguments".into(),
                });
            }
            for &child in &children[1..] {
                let val = get_cached(cache, child)?;
                if !val.1.is_zero() {
                    return Err(SymplexError::Unevaluable {
                        reason: "max requires real arguments".into(),
                    });
                }
                if is_strictly_positive(&val.0.sub(&best.0, prec, RM)) {
                    best = val.clone();
                }
            }
            Ok(best)
        }

        // ── Heaviside ──────────────────────────────────────────────
        ExprNode::Heaviside(inner) => {
            let val = get_cached(cache, *inner)?;
            if !val.1.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "Heaviside of complex argument not supported in evalf".into(),
                });
            }
            if val.0.is_zero() {
                // H(0) = 0.5
                Ok((BigFloat::from_f64(0.5, prec), BigFloat::new(prec)))
            } else if val.0.is_negative() {
                Ok(c_zero(prec))
            } else {
                Ok((BigFloat::from_i32(1, prec), BigFloat::new(prec)))
            }
        }

        // ── DiracDelta ─────────────────────────────────────────────
        // A distribution on the real line: `δ(x) = 0` for every real
        // `x ≠ 0`, no value at 0 and none off the reals (SymPy 1.14 rejects
        // `DiracDelta(I)`; the limit engine and `is_zero` leave it
        // undecided).  A 0 or non-real argument is refused, unless it is
        // only so to the working precision (`eval_ball` then has no value
        // yet and asks for more precision); an argument within its error
        // of 0, or whose imaginary part is not an exact 0, has no bound
        // (`node_error`).  Up to 0.49 `δ(0)` was a certified `0`, up to
        // 0.50 `δ(i)` (and `compile`, which folds constants by `evalf`).
        ExprNode::DiracDelta(inner) => {
            let val = get_cached(cache, *inner)?;
            // An imaginary part nonzero beyond its error (without a bound:
            // nonzero).
            let non_real = !val.1.is_zero()
                && cache
                    .bound(inner)
                    .is_none_or(|b| !accuracy::part_contains_zero(&val.1, b.im));
            if non_real {
                return Err(unevaluable(
                    "DiracDelta has no value at a non-real argument",
                ));
            }
            if val.0.is_zero() && val.1.is_zero() {
                return Err(unevaluable("DiracDelta(0) has no value"));
            }
            Ok(c_zero(prec))
        }

        // ── Library special functions ────────────────────────────────
        ExprNode::Lib(f, args) => match f {
            LibFn::LambertW => eval_lambertw_node(arena, node, cache, w),
            f if f.arity().accepts(args.len()) => eval_lib_fn(*f, args, cache, w),
            _ => Err(unevaluable(format!(
                "{f} called with {} argument(s)",
                args.len()
            ))),
        },
        // A user function has no numeric meaning.
        ExprNode::Apply(sid, _) => {
            let name = arena.symbol_name(*sid);
            Err(unevaluable(format!("cannot evaluate function '{name}'")))
        }
        // Nor does an internal function of the limit engine.
        ExprNode::Internal(f, _) => Err(unevaluable(format!(
            "cannot evaluate function '{}'",
            f.name()
        ))),

        ExprNode::Derivative(_, _) => Err(SymplexError::Unevaluable {
            reason: "cannot evaluate unevaluated derivative".into(),
        }),

        ExprNode::Integral(_, _) => Err(SymplexError::Unevaluable {
            reason: "cannot evaluate unevaluated integral".into(),
        }),

        ExprNode::BoolTrue
        | ExprNode::BoolFalse
        | ExprNode::Gt(_, _)
        | ExprNode::Ge(_, _)
        | ExprNode::Eq_(_, _)
        | ExprNode::Ne(_, _)
        | ExprNode::And(_)
        | ExprNode::Or(_)
        | ExprNode::Not(_) => Err(SymplexError::Unevaluable {
            reason: "boolean expression".into(),
        }),

        ExprNode::EmptySet
        | ExprNode::UniversalSet
        | ExprNode::Interval(_, _, _)
        | ExprNode::FiniteSet(_)
        | ExprNode::SetUnion(_)
        | ExprNode::SetIntersection(_)
        | ExprNode::SetComplement(_, _)
        | ExprNode::ImageSet(_, _) => Err(SymplexError::Unevaluable {
            reason: "set-valued expressions cannot be numerically evaluated".into(),
        }),

        ExprNode::Limit(_, _, _) => Err(SymplexError::Unevaluable {
            reason: "cannot numerically evaluate unevaluated Limit".into(),
        }),

        ExprNode::Series(_, _, _, _) => Err(SymplexError::Unevaluable {
            reason: "cannot numerically evaluate unevaluated Series".into(),
        }),

        ExprNode::LaplaceTransform(_, _, _) => Err(SymplexError::Unevaluable {
            reason: "cannot numerically evaluate unevaluated LaplaceTransform".into(),
        }),

        ExprNode::InverseLaplaceTransform(_, _, _) => Err(SymplexError::Unevaluable {
            reason: "cannot numerically evaluate unevaluated InverseLaplaceTransform".into(),
        }),

        ExprNode::Residue(_, _, _) => Err(SymplexError::Unevaluable {
            reason: "cannot numerically evaluate unevaluated Residue".into(),
        }),

        ExprNode::DSolve(_, _, _) => Err(SymplexError::Unevaluable {
            reason: "cannot numerically evaluate unevaluated DSolve".into(),
        }),

        ExprNode::ConditionSet(_, _) => Err(SymplexError::Unevaluable {
            reason: "cannot numerically evaluate ConditionSet".into(),
        }),

        ExprNode::Subs(_, _, _) => Err(SymplexError::Unevaluable {
            reason: "cannot numerically evaluate unevaluated Subs".into(),
        }),
    }
}

// A long range whose terms peak far from its ends: a window around the
// mode with a proved bound on the rest of each side.
mod windowsum;
// The field operations live in `base::bigcomplex` (shared with the root
// finders and algebraic-number verification); the transcendental ones
// below need this module's constant cache.
pub(crate) use crate::base::bigcomplex::{
    c_abs, c_add, c_div, c_from_real, c_i, c_mul, c_neg, c_one, c_powi, c_sub, c_zero,
};
// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

// The same sum far below the mean of a huge shape: an expansion in
// `s/(s − x)²` whose error is at most its first omitted term.
mod gammatail;
// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::arena::Arena;

    /// Helper: evaluate and assert the result starts with expected prefix.
    fn assert_evalf_starts_with(arena: &Arena, expr: ExprId, digits: u32, prefix: &str) {
        let result = evalf(arena, expr, digits).unwrap();
        assert!(
            result.starts_with(prefix),
            "expected to start with '{prefix}', got: '{result}'"
        );
    }

    /// Helper: evaluate and assert exact string match.
    fn assert_evalf_eq(arena: &Arena, expr: ExprId, digits: u32, expected: &str) {
        let result = evalf(arena, expr, digits).unwrap();
        assert_eq!(result, expected, "evalf mismatch");
    }

    // ── Integers ────────────────────────────────────────────────────

    #[test]
    fn integer_zero() {
        let a = Arena::new();
        assert_evalf_eq(&a, a.zero, 10, "0");
    }

    #[test]
    fn integer_one() {
        let a = Arena::new();
        assert_evalf_eq(&a, a.one, 10, "1");
    }

    #[test]
    fn integer_negative_one() {
        let a = Arena::new();
        assert_evalf_eq(&a, a.neg_one, 10, "-1");
    }

    #[test]
    fn integer_five() {
        let mut a = Arena::new();
        let five = a.int(5);
        assert_evalf_eq(&a, five, 10, "5");
    }

    #[test]
    fn integer_large() {
        let mut a = Arena::new();
        let n = a.int(123456);
        assert_evalf_eq(&a, n, 10, "123456");
    }

    // ── Rationals ───────────────────────────────────────────────────

    #[test]
    fn rational_half() {
        let mut a = Arena::new();
        let half = a.rational(1, 2);
        assert_evalf_eq(&a, half, 10, "0.5");
    }

    #[test]
    fn rational_one_third() {
        let mut a = Arena::new();
        let third = a.rational(1, 3);
        assert_evalf_starts_with(&a, third, 10, "0.333333333");
    }

    #[test]
    fn rational_negative() {
        let mut a = Arena::new();
        let r = a.rational(-3, 4);
        assert_evalf_starts_with(&a, r, 10, "-0.75");
    }

    // ── Constants ───────────────────────────────────────────────────

    #[test]
    fn constant_pi_15_digits() {
        let a = Arena::new();
        assert_evalf_starts_with(&a, a.pi, 15, "3.14159265358979");
    }

    #[test]
    fn constant_pi_50_digits() {
        let a = Arena::new();
        let result = evalf(&a, a.pi, 50).unwrap();
        assert!(
            result.starts_with("3.1415926535897932384626433832795"),
            "pi to 50 digits: {result}"
        );
    }

    #[test]
    fn constant_e_15_digits() {
        let a = Arena::new();
        assert_evalf_starts_with(&a, a.e_const, 15, "2.71828182845905");
    }

    // ── Arithmetic ──────────────────────────────────────────────────

    #[test]
    fn add_integers() {
        let mut a = Arena::new();
        let two = a.int(2);
        let three = a.int(3);
        let expr = a.add(&[two, three]);
        assert_evalf_eq(&a, expr, 10, "5");
    }

    #[test]
    fn mul_integers() {
        let mut a = Arena::new();
        let two = a.int(2);
        let three = a.int(3);
        let expr = a.mul(&[two, three]);
        assert_evalf_eq(&a, expr, 10, "6");
    }

    #[test]
    fn pow_integer() {
        let mut a = Arena::new();
        let two = a.int(2);
        let ten = a.int(10);
        let expr = a.pow(two, ten);
        assert_evalf_eq(&a, expr, 10, "1024");
    }

    #[test]
    fn pow_negative_exponent() {
        let mut a = Arena::new();
        let two = a.int(2);
        let neg_one = a.int(-1);
        let expr = a.pow(two, neg_one);
        assert_evalf_eq(&a, expr, 10, "0.5");
    }

    #[test]
    fn pi_plus_one() {
        let mut a = Arena::new();
        let pi = a.pi;
        let one = a.one;
        let expr = a.add(&[pi, one]);
        assert_evalf_starts_with(&a, expr, 15, "4.14159265358979");
    }

    // ── Transcendental functions ─────────────────────────────────────

    #[test]
    fn sin_zero() {
        let mut a = Arena::new();
        let zero = a.zero;
        let expr = a.sin(zero);
        assert_evalf_eq(&a, expr, 10, "0");
    }

    #[test]
    fn cos_zero() {
        let mut a = Arena::new();
        let zero = a.zero;
        let expr = a.cos(zero);
        assert_evalf_eq(&a, expr, 10, "1");
    }

    #[test]
    fn exp_zero() {
        let mut a = Arena::new();
        let zero = a.zero;
        let expr = a.exp(zero);
        assert_evalf_eq(&a, expr, 10, "1");
    }

    #[test]
    fn exp_one() {
        let mut a = Arena::new();
        let one = a.one;
        let expr = a.exp(one);
        assert_evalf_starts_with(&a, expr, 15, "2.71828182845905");
    }

    #[test]
    fn ln_one() {
        let mut a = Arena::new();
        let one = a.one;
        let expr = a.ln(one);
        assert_evalf_eq(&a, expr, 10, "0");
    }

    #[test]
    fn ln_e() {
        let mut a = Arena::new();
        let e = a.e_const;
        let expr = a.ln(e);
        // ln(e) should be 1, or very close.
        let result = evalf(&a, expr, 15).unwrap();
        let val: f64 = result.parse().unwrap_or(999.0);
        assert!(
            (val - 1.0).abs() < 1e-10,
            "ln(e) should be ~1, got: {result}"
        );
    }

    #[test]
    fn sqrt_four() {
        let mut a = Arena::new();
        let four = a.int(4);
        let expr = a.sqrt(four);
        assert_evalf_eq(&a, expr, 10, "2");
    }

    #[test]
    fn sqrt_two_30_digits() {
        let mut a = Arena::new();
        let two = a.int(2);
        let expr = a.sqrt(two);
        assert_evalf_starts_with(&a, expr, 30, "1.4142135623730950488");
    }

    #[test]
    fn sin_pi_is_near_zero() {
        let mut a = Arena::new();
        let pi = a.pi;
        let expr = a.sin(pi);
        let result = evalf(&a, expr, 20).unwrap();
        // sin(pi) is mathematically 0, numerically very small.
        let val: f64 = result.parse().unwrap_or(999.0);
        assert!(val.abs() < 1e-15, "sin(pi) should be ~0, got: {result}");
    }

    // ── Neg and Abs ─────────────────────────────────────────────────

    #[test]
    fn neg_pi() {
        let mut a = Arena::new();
        let pi = a.pi;
        let expr = a.neg(pi);
        assert_evalf_starts_with(&a, expr, 10, "-3.14159265");
    }

    #[test]
    fn abs_neg_pi() {
        let mut a = Arena::new();
        let pi = a.pi;
        let neg_pi = a.neg(pi);
        let expr = a.abs(neg_pi);
        assert_evalf_starts_with(&a, expr, 10, "3.14159265");
    }

    #[test]
    fn abs_negative_integer() {
        let mut a = Arena::new();
        let neg_five = a.int(-5);
        let expr = a.abs(neg_five);
        assert_evalf_eq(&a, expr, 10, "5");
    }

    // ── Error cases ─────────────────────────────────────────────────

    #[test]
    fn free_symbol_errors() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = evalf(&a, x, 10);
        assert!(result.is_err(), "evalf of free symbol should error");
    }

    #[test]
    fn infinity_errors() {
        let a = Arena::new();
        let result = evalf(&a, a.infinity, 10);
        assert!(result.is_err(), "evalf of infinity should error");
    }

    #[test]
    fn nan_errors() {
        let a = Arena::new();
        let result = evalf(&a, a.nan, 10);
        assert!(result.is_err(), "evalf of NaN should error");
    }

    #[test]
    fn imaginary_unit_works() {
        let a = Arena::new();
        let result = evalf(&a, a.i_unit, 10);
        assert!(result.is_ok(), "evalf of i should work now");
        let s = result.unwrap();
        assert!(
            s.contains("i") || s.contains("I"),
            "should display as imaginary: {s}"
        );
    }

    // ── Complex number tests ────────────────────────────────────────

    #[test]
    fn evalf_one_plus_i() {
        let mut a = Arena::new();
        let one = a.one;
        let i = a.i_unit;
        let expr = a.add(&[one, i]);
        let result = evalf(&a, expr, 10);
        assert!(result.is_ok(), "evalf(1+i) should work: {:?}", result.err());
        let s = result.unwrap();
        assert!(
            s.contains("i") || s.contains("I"),
            "1+i should contain 'i': {s}"
        );
    }

    #[test]
    fn evalf_exp_i_pi() {
        // e^(i*pi) ≈ -1
        let mut a = Arena::new();
        let i = a.i_unit;
        let pi = a.pi;
        let i_times_pi = a.mul(&[i, pi]);
        let expr = a.exp(i_times_pi);
        let result = evalf(&a, expr, 15);
        assert!(
            result.is_ok(),
            "evalf(exp(i*pi)) should work: {:?}",
            result.err()
        );
        let s = result.unwrap();
        let val: f64 = s.parse().unwrap_or(999.0);
        assert!(
            (val - (-1.0)).abs() < 1e-10,
            "exp(i*pi) should be ~-1, got: {s}"
        );
    }

    #[test]
    fn evalf_i_squared() {
        // i^2 = -1 (may be canonicalized by the arena)
        let mut a = Arena::new();
        let i = a.i_unit;
        let two = a.int(2);
        let expr = a.pow(i, two);
        let result = evalf(&a, expr, 10);
        assert!(result.is_ok(), "evalf(i^2) should work: {:?}", result.err());
        let s = result.unwrap();
        let val: f64 = s.parse().unwrap_or(999.0);
        assert!((val - (-1.0)).abs() < 1e-10, "i^2 should be -1, got: {s}");
    }

    #[test]
    fn evalf_sqrt_neg_one() {
        // sqrt(-1) = (-1)^(1/2) should give i
        let mut a = Arena::new();
        let neg_one = a.neg_one;
        let expr = a.sqrt(neg_one);
        let result = evalf(&a, expr, 10);
        assert!(
            result.is_ok(),
            "evalf(sqrt(-1)) should work: {:?}",
            result.err()
        );
        let s = result.unwrap();
        assert!(
            s.contains("i") || s.contains("I"),
            "sqrt(-1) should be imaginary: {s}"
        );
    }

    #[test]
    fn evalf_abs_3_plus_4i() {
        // |3 + 4i| = 5
        let mut a = Arena::new();
        let three = a.int(3);
        let four = a.int(4);
        let i = a.i_unit;
        let four_i = a.mul(&[four, i]);
        let sum = a.add(&[three, four_i]);
        let expr = a.abs(sum);
        let result = evalf(&a, expr, 10);
        assert!(
            result.is_ok(),
            "evalf(|3+4i|) should work: {:?}",
            result.err()
        );
        let s = result.unwrap();
        let val: f64 = s.parse().unwrap_or(999.0);
        assert!((val - 5.0).abs() < 1e-10, "|3+4i| should be 5, got: {s}");
    }

    // ── Composite expressions ───────────────────────────────────────

    #[test]
    fn polynomial_at_value() {
        // Evaluate x^2 + 2*x + 1 at x=3 → 16.
        // We substitute first, then evalf.
        let mut a = Arena::new();
        let x = a.symbol("x");
        let two = a.int(2);
        let x_sq = a.pow(x, two);
        let two_x = a.mul(&[two, x]);
        let expr = a.add(&[x_sq, two_x, a.one]);

        // Substitute x → 3.
        let three = a.int(3);
        let evaluated = crate::transforms::subs::subs(&mut a, expr, x, three);
        assert_evalf_eq(&a, evaluated, 10, "16");
    }

    #[test]
    fn sin_cos_identity() {
        // sin^2(1) + cos^2(1) should be very close to 1.
        let mut a = Arena::new();
        let one = a.one;
        let two = a.int(2);
        let sin_1 = a.sin(one);
        let cos_1 = a.cos(one);
        let sin_sq = a.pow(sin_1, two);
        let cos_sq = a.pow(cos_1, two);
        let expr = a.add(&[sin_sq, cos_sq]);
        let result = evalf(&a, expr, 15).unwrap();
        let val: f64 = result.parse().unwrap_or(0.0);
        assert!(
            (val - 1.0).abs() < 1e-12,
            "sin^2(1) + cos^2(1) should be ~1, got: {result}"
        );
    }

    // ── Idempotence ─────────────────────────────────────────────────

    #[test]
    fn same_precision_same_result() {
        let a = Arena::new();
        let r1 = evalf(&a, a.pi, 20).unwrap();
        let r2 = evalf(&a, a.pi, 20).unwrap();
        assert_eq!(r1, r2, "same precision should give same result");
    }

    // ── Bessel function evaluation ─────────────────────────────────

    /// The exact-integer coefficients of `Y_n`, `K_n` and `ψ⁽ⁿ⁾` (0.34) round
    /// to the same BigFloats as the reduced `Ratio` recurrences before them:
    /// the same digits, without the gcds.
    /// The term-by-term Jacobi sum (0.34) agrees with the sum that recomputed
    /// every coefficient (before 0.34, kept here as the reference), also for
    /// a negative integer `a` (leading terms 0) and at `x = ±1`.
    #[test]
    fn jacobi_term_by_term_matches_the_explicit_sum() {
        let mut cc = Consts::new().unwrap();
        let prec = 128;
        let reference = |deg: usize, a: &BigFloat, b: &BigFloat, x: &BigFloat| -> BigFloat {
            let wp = prec + 32 + 2 * deg;
            let one = BigFloat::from_i32(1, wp);
            let two = BigFloat::from_i32(2, wp);
            let lo = x.sub(&one, wp, RM).div(&two, wp, RM);
            let hi = x.add(&one, wp, RM).div(&two, wp, RM);
            let shifted = |p: &BigFloat, lo_j: usize, hi_j: usize| -> BigFloat {
                let mut acc = one.clone();
                for j in lo_j..=hi_j {
                    acc = acc.mul(&p.add(&BigFloat::from_i128(j as i128, wp), wp, RM), wp, RM);
                }
                acc
            };
            let mut sum = BigFloat::new(wp);
            for s in 0..=deg {
                let pa = shifted(a, s + 1, deg);
                let pb = shifted(b, deg - s + 1, deg);
                let d = ratio_to_bigfloat(&(factorial_bigint(deg - s) * factorial_bigint(s)), wp);
                let term = pa
                    .mul(&pb, wp, RM)
                    .div(&d, wp, RM)
                    .mul(&lo.powi(s, wp, RM), wp, RM)
                    .mul(&hi.powi(deg - s, wp, RM), wp, RM);
                sum = sum.add(&term, wp, RM);
            }
            round_to(sum, prec)
        };
        let q =
            |n: i32, d: i32| BigFloat::from_i32(n, 256).div(&BigFloat::from_i32(d, 256), 256, RM);
        for deg in [1usize, 2, 5, 17, 60] {
            for (a, b) in [
                (q(1, 3), q(1, 5)),
                (q(-3, 1), q(2, 7)),
                (q(-1, 1), q(-2, 1)),
                (q(5, 2), q(-7, 3)),
            ] {
                for x in [q(1, 3), q(-9, 10), q(1, 1), q(-1, 1), q(3, 1), q(-5, 2)] {
                    let n = BigFloat::from_u64(deg as u64, 64);
                    let got = arb_jacobi(&n, &a, &b, &x, &mut Wp::new(prec, &mut cc)).unwrap();
                    let want = reference(deg, &a, &b, &x);
                    let diff = got.sub(&want, prec, RM).abs();
                    let ok = diff.is_zero()
                        || match (diff.exponent(), want.exponent()) {
                            (Some(d), Some(w)) => i64::from(d) < i64::from(w) - (prec as i64 - 4),
                            _ => false,
                        };
                    assert!(ok, "deg={deg} a={a} b={b} x={x}: {got} vs {want}");
                }
            }
        }
    }

    #[test]
    fn exact_coefficients_match_the_ratio_recurrences() {
        for (n, wp) in (1..60usize).flat_map(|n| [(n, 64), (n, 160)]) {
            let mut f = Ratio::<BigInt>::from_integer(BigInt::from(1));
            for i in 2..n {
                f *= Ratio::from_integer(BigInt::from(i as u64));
            }
            let mut g = NeumannCoefficients::new(n);
            for k in 0..n {
                if k > 0 {
                    f /= Ratio::from_integer(BigInt::from(((n - k) * k) as u64));
                }
                assert_eq!(g.next(wp), ratio_to_bigfloat(&f, wp), "n={n} k={k}");
            }
            let mut fact = Ratio::<BigInt>::from_integer(BigInt::from(1));
            for i in 2..=n {
                fact *= Ratio::from_integer(BigInt::from(i as u64));
            }
            assert_eq!(factorial_exact(n), fact);
        }
        for k in 1..40 {
            let b = bernoulli::even(k);
            for r in [1u64, 2, 6, 30, 42, 1155, 1_000_003, 510_510] {
                let r = BigInt::from(r) * BigInt::from(k as u64 * 7 + 1);
                assert_eq!(times_integer(&b, &r), &b * Ratio::from_integer(r.clone()));
            }
        }
    }

    #[test]
    fn bessel_j0_at_zero() {
        // J_0(0) = 1
        let mut a = Arena::new();
        let zero = a.zero;
        let j0_0 = a.besselj(zero, zero);
        let result = evalf(&a, j0_0, 15).unwrap();
        let val: f64 = result.parse().unwrap_or(f64::NAN);
        assert!(
            (val - 1.0).abs() < 1e-10,
            "J_0(0) should be 1.0, got {result}"
        );
    }

    #[test]
    fn bessel_j0_at_one() {
        // J_0(1) ≈ 0.7651976865579666
        let mut a = Arena::new();
        let zero = a.zero;
        let one = a.one;
        let j0_1 = a.besselj(zero, one);
        let result = evalf(&a, j0_1, 15).unwrap();
        let val: f64 = result.parse().unwrap_or(f64::NAN);
        assert!(
            (val - 0.7651976865579666).abs() < 1e-8,
            "J_0(1) should be ~0.7652, got {result}"
        );
    }

    #[test]
    fn bessel_j1_at_zero() {
        // J_1(0) = 0
        let mut a = Arena::new();
        let one = a.one;
        let zero = a.zero;
        let j1_0 = a.besselj(one, zero);
        let result = evalf(&a, j1_0, 15).unwrap();
        let val: f64 = result.parse().unwrap_or(f64::NAN);
        assert!(val.abs() < 1e-10, "J_1(0) should be 0, got {result}");
    }

    #[test]
    fn bessel_j1_at_one() {
        // J_1(1) ≈ 0.44005058574493355
        let mut a = Arena::new();
        let one = a.one;
        let j1_1 = a.besselj(one, one);
        let result = evalf(&a, j1_1, 15).unwrap();
        let val: f64 = result.parse().unwrap_or(f64::NAN);
        assert!(
            (val - 0.44005058574493355).abs() < 1e-8,
            "J_1(1) should be ~0.4401, got {result}"
        );
    }

    #[test]
    fn bessel_y0_at_one() {
        // Y_0(1) ≈ 0.08825696421567696
        let mut a = Arena::new();
        let zero = a.zero;
        let one = a.one;
        let y0_1 = a.bessely(zero, one);
        let result = evalf(&a, y0_1, 15).unwrap();
        let val: f64 = result.parse().unwrap_or(f64::NAN);
        assert!(
            (val - 0.08825696421567696).abs() < 1e-6,
            "Y_0(1) should be ~0.0883, got {result}"
        );
    }

    // ── 0.2: named constants ───────────────────────────────────────────────

    #[test]
    fn euler_gamma_and_catalan_high_precision() {
        let a = Arena::new();
        assert_evalf_starts_with(
            &a,
            a.euler_gamma,
            60,
            "0.57721566490153286060651209008240243104215933593992359880576",
        );
        assert_evalf_starts_with(
            &a,
            a.catalan,
            60,
            "0.91596559417721901505460351493238411077414937428167213426649",
        );
        assert_evalf_starts_with(
            &a,
            a.golden_ratio,
            40,
            "1.61803398874989484820458683436563811772",
        );
    }

    #[test]
    fn bigint_to_bigfloat_is_exact_for_huge_integers() {
        // 2^200 + 1 is far beyond i128; its BigFloat must keep all bits.
        let n = (BigInt::from(1) << 200) + BigInt::from(1);
        let bf = crate::base::numeric::bigint_to_bigfloat(&n, 256);
        let two200 = BigFloat::from_i32(2, 256).powi(200, 256, RM);
        let diff = bf.sub(&two200, 256, RM);
        assert_eq!(diff, BigFloat::from_i32(1, 256));
    }

    #[test]
    fn bf_pow_handles_exactly_representable_results() {
        // BigFloat::pow would hang on 4^(1/2) = 2; bf_pow must not.
        let mut cc = Consts::new().unwrap();
        let four = BigFloat::from_i32(4, 160);
        let half = BigFloat::from_f64(0.5, 160);
        let r = bf_pow(&four, &half, &mut Wp::new(160, &mut cc));
        let diff = r.sub(&BigFloat::from_i32(2, 160), 160, RM);
        assert!(
            diff.is_zero() || diff.exponent().is_none_or(|e| e < -140),
            "4^0.5 = {r}"
        );
        // integer exponent path
        let three = BigFloat::from_i32(3, 160);
        let r = bf_pow(
            &three,
            &BigFloat::from_i32(5, 160),
            &mut Wp::new(160, &mut cc),
        );
        assert_eq!(r, BigFloat::from_i32(243, 160));
        let r = bf_pow(
            &three,
            &BigFloat::from_i32(-1, 160),
            &mut Wp::new(160, &mut cc),
        );
        let back = r.mul(&three, 160, RM);
        let d = back.sub(&BigFloat::from_i32(1, 160), 160, RM);
        assert!(d.is_zero() || d.exponent().is_none_or(|e| e < -140));
    }

    // ── 0.2: special functions ──────────────────────────────────────────────

    #[test]
    fn si_ci_ei_li_at_one_and_two() {
        let mut a = Arena::new();
        let one = a.one;
        let two = a.int(2);
        let si = a.si(one);
        assert_evalf_starts_with(&a, si, 20, "0.94608307036718301494");
        let ci = a.ci(one);
        assert_evalf_starts_with(&a, ci, 20, "0.33740392290096813466");
        let ei = a.ei(one);
        assert_evalf_starts_with(&a, ei, 20, "1.8951178163559367555");
        let li = a.li(two);
        assert_evalf_starts_with(&a, li, 20, "1.0451637801174927848");
    }

    #[test]
    fn si_asymptotic_and_series_regimes_agree() {
        // Evaluate Si(40) at two precisions: at low precision the asymptotic
        // expansion is used (40 > 160·ln2 ≈ 111 is false …), so force the
        // regimes by comparing against the same value at higher precision
        // where the series is used.
        let mut a = Arena::new();
        let x = a.int(200);
        let si = a.si(x);
        let lo = evalf(&a, si, 12).unwrap(); // asymptotic (200 > 128·ln2)
        let hi = evalf(&a, si, 120).unwrap(); // series (200 < 472·ln2)
        assert!(hi.starts_with(&lo[..12]), "{lo} vs {hi}");
    }

    #[test]
    fn zeta_borwein_and_functional_equation() {
        let mut a = Arena::new();
        let three = a.int(3);
        let z3 = a.zeta(three);
        assert_evalf_starts_with(&a, z3, 30, "1.20205690315959428539973816151");
        let half = a.rational(1, 2);
        let zh = a.zeta(half);
        assert_evalf_starts_with(&a, zh, 20, "-1.4603545088095868129");
        // ζ(−5/2) > 0 (ζ is positive on (−4, −2); ζ(−3) = 1/120).
        let neg = a.rational(-5, 2);
        let zn = a.zeta(neg);
        assert_evalf_starts_with(&a, zn, 15, "0.00851692877785");
        // ζ(−1/2) < 0
        let neg_half = a.rational(-1, 2);
        let znh = a.zeta(neg_half);
        assert_evalf_starts_with(&a, znh, 15, "-0.20788622497735");
    }

    #[test]
    fn polygamma_matches_trigamma_closed_forms() {
        let mut a = Arena::new();
        // Build the node structurally so the exact folding does not kick in.
        let one = a.one;
        let x = a.rational(7, 3);
        let pg = a.intern(ExprNode::Polygamma(one, x));
        assert_evalf_starts_with(&a, pg, 20, "0.53309712542709408179");
        // ψ'(1/2) = π²/2 through the numerical path
        let half = a.rational(1, 2);
        let pg_half = a.intern(ExprNode::Polygamma(one, half));
        assert_evalf_starts_with(&a, pg_half, 20, "4.9348022005446793094");
    }

    #[test]
    fn bessel_i_k_reference() {
        let mut a = Arena::new();
        let zero = a.zero;
        let one = a.one;
        let i0 = a.besseli(zero, one);
        assert_evalf_starts_with(&a, i0, 20, "1.2660658777520083356");
        let k0 = a.besselk(zero, one);
        assert_evalf_starts_with(&a, k0, 20, "0.42102443824070833334");
        let k1 = a.besselk(one, one);
        assert_evalf_starts_with(&a, k1, 20, "0.60190723019723457474");
    }

    #[test]
    fn orthogonal_polynomials_large_degree() {
        let mut a = Arena::new();
        let n = a.int(100);
        let third = a.rational(1, 3);
        let t = a.chebyshev_t(n, third);
        let s = evalf(&a, t, 15).unwrap();
        let v: f64 = s.parse().unwrap();
        let expected = (100.0 * (1.0f64 / 3.0).acos()).cos();
        assert!((v - expected).abs() < 1e-12, "{v} vs {expected}");
        let p = a.legendre(n, a.one);
        assert_evalf_starts_with(&a, p, 10, "1");
    }

    #[test]
    fn complex_nodes_evaluate_numerically() {
        let mut a = Arena::new();
        let z = a.symbol("z");
        let re_z = a.intern(ExprNode::Re(z));
        let arg_z = a.intern(ExprNode::Arg(z));
        let conj_z = a.intern(ExprNode::Conjugate(z));
        let three = a.int(3);
        let four = a.int(4);
        let four_i = a.mul(&[four, a.i_unit]);
        let w = a.add(&[three, four_i]);
        // Substitute structurally *without* refolding: intern the nodes
        // directly on w.
        let _ = re_z;
        let re_w = a.intern(ExprNode::Re(w));
        let arg_w = a.intern(ExprNode::Arg(w));
        let conj_w = a.intern(ExprNode::Conjugate(w));
        let _ = (arg_z, conj_z);
        assert_eq!(evalf(&a, re_w, 10).unwrap(), "3");
        assert_evalf_starts_with(&a, arg_w, 15, "0.92729521800161");
        let s = evalf(&a, conj_w, 10).unwrap();
        assert!(s.contains('3') && s.contains('4') && s.contains('-'), "{s}");
    }

    /// A `RootSum` over a polynomial with repeated roots counts each root
    /// with its multiplicity.  Before 0.29 it was `PrecisionExhausted`: the
    /// inclusion disks of a repeated root overlap, so no root was certified.
    #[test]
    fn rootsum_over_repeated_roots_counts_multiplicity() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let two = a.int(2);
        let three = a.int(3);
        let minus_two = a.int(-2);
        let minus_one = a.int(-1);
        // (x² − 2)², body 1/(x + 3).
        let x2 = a.pow(x, two);
        let q = a.add(&[x2, minus_two]);
        let poly = a.pow(q, two);
        let xp3 = a.add(&[x, three]);
        let body = a.pow(xp3, minus_one);
        let rs = a.intern(ExprNode::RootSum(poly, body, x));
        // mpmath: mp.dps=50; 2*(1/(3+sqrt(2)) + 1/(3-sqrt(2)))
        //         = 1.71428571428571428571428571429
        assert_evalf_eq(&a, rs, 20, "1.7142857142857142857");
        // (x² + 1)²·(x − 3), body exp(x).
        let one = a.int(1);
        let x2p1 = a.add(&[x2, one]);
        let sq = a.pow(x2p1, two);
        let minus_three = a.int(-3);
        let xm3 = a.add(&[x, minus_three]);
        let poly = a.mul(&[sq, xm3]);
        let body = a.intern(ExprNode::Exp(x));
        let rs = a.intern(ExprNode::RootSum(poly, body, x));
        // mpmath: mp.dps=50; 2*(exp(1j)+exp(-1j)) + exp(3)
        //         = (22.2467461466602266105322760844 + 0.0j)
        assert_evalf_eq(&a, rs, 20, "22.246746146660226611");
    }

    // ── 0.11.1 special-function fixes (direct numeric paths that `eval`
    //    would otherwise fold away) ──────────────────────────────────────────

    fn apply(a: &mut Arena, f: LibFn, args: &[ExprId]) -> ExprId {
        a.lib_apply(f, args)
    }

    #[test]
    fn erfc_node_large_argument_is_relative_precision() {
        let mut a = Arena::new();
        // mpmath 1.3: erfc(7) = 4.18382560777941439861401e-23
        let seven = a.int(7);
        let e7 = a.erfc(seven);
        assert_evalf_starts_with(&a, e7, 16, "4.183825607779414e-23");
        // mpmath 1.3: erfc(10) = 2.088487583762544757000786294957788611560818119321163727e-45
        let ten = a.int(10);
        let e10 = a.erfc(ten);
        assert_evalf_starts_with(
            &a,
            e10,
            50,
            "2.0884875837625447570007862949577886115608181193212e-45",
        );
        // mpmath 1.3: erfc(30) = 2.564656203756111600033397e-393 (asymptotic branch)
        let thirty = a.int(30);
        let e30 = a.erfc(thirty);
        assert_evalf_starts_with(&a, e30, 16, "2.564656203756112e-393");
        // mpmath 1.3: erfc(-3) = 1.999977909503001414558627 (no cancellation)
        let m3 = a.int(-3);
        let em3 = a.erfc(m3);
        assert_evalf_starts_with(&a, em3, 16, "1.999977909503001");
    }

    #[test]
    fn erf_taylor_branch_carries_cancellation_guard_bits() {
        let mut a = Arena::new();
        // mpmath 1.3: erf(7.4) = 0.99999999999999999999999987516143536
        let x = a.rational(74, 10);
        let e = a.erf(x);
        assert_evalf_starts_with(&a, e, 30, "0.999999999999999999999999875161");
        // mpmath 1.3: erf(10) = 0.9999999999999999999999999999999999999999999979115124162
        // (50 digits: the Taylor terms peak near e^{100} ≈ 2^{144})
        let ten = a.int(10);
        let e10 = a.erf(ten);
        assert_evalf_starts_with(
            &a,
            e10,
            50,
            "0.99999999999999999999999999999999999999999999791151",
        );
        // mpmath 1.3: erf(3) = 0.999977909503001414558627223870417679620152293
        let three = a.int(3);
        let e3 = a.erf(three);
        assert_evalf_starts_with(&a, e3, 40, "0.999977909503001414558627223870417679620");
    }

    #[test]
    fn polylog_at_minus_one_is_minus_eta_for_all_s() {
        let mut a = Arena::new();
        let neg_one = a.neg_one;
        // mpmath 1.3: polylog(1/2, -1) = -0.6048986434216303702472659142359555
        let half = a.rational(1, 2);
        let li = apply(&mut a, LibFn::PolyLog, &[half, neg_one]);
        assert_evalf_starts_with(&a, li, 30, "-0.60489864342163037024726591423");
        // mpmath 1.3: polylog(1, -1) = -ln 2 = -0.69314718055994530941723212145817657
        let one = a.one;
        let li1 = apply(&mut a, LibFn::PolyLog, &[one, neg_one]);
        assert_evalf_starts_with(&a, li1, 30, "-0.69314718055994530941723212145");
        // mpmath 1.3: polylog(-1/2, -1) = -0.38010481260968401677754215655180836
        let neg_half = a.rational(-1, 2);
        let lim = apply(&mut a, LibFn::PolyLog, &[neg_half, neg_one]);
        assert_evalf_starts_with(&a, lim, 30, "-0.38010481260968401677754215655");
        // mpmath 1.3: polylog(3/2, -1) = -0.76514702462540794536726875860347818
        let three_halves = a.rational(3, 2);
        let li32 = apply(&mut a, LibFn::PolyLog, &[three_halves, neg_one]);
        assert_evalf_starts_with(&a, li32, 30, "-0.76514702462540794536726875860");
        // z = +1 still diverges for s ≤ 1
        let one_id = a.one;
        let div = apply(&mut a, LibFn::PolyLog, &[half, one_id]);
        assert!(evalf(&a, div, 16).is_err());
    }

    #[test]
    fn dirichlet_eta_near_one_uses_alternating_sum() {
        let mut a = Arena::new();
        // mpmath 1.3: altzeta(1 + 1e-20) = 0.69314718055994530941883081049560088
        let ten = a.int(10);
        let e20 = a.int(-20);
        let tiny = a.pow(ten, e20);
        let s = a.add(&[a.one, tiny]);
        let eta = apply(&mut a, LibFn::DirichletEta, &[s]);
        assert_evalf_starts_with(&a, eta, 30, "0.69314718055994530941883081049");
        // mpmath 1.3: altzeta(1 - 1e-20) = 0.69314718055994530941563343242075226
        let s2 = a.sub(a.one, tiny);
        let eta2 = apply(&mut a, LibFn::DirichletEta, &[s2]);
        assert_evalf_starts_with(&a, eta2, 30, "0.69314718055994530941563343242");
        // Exactly 1 through the numeric path: mpmath altzeta(1) = ln 2
        let one = a.one;
        let eta1 = apply(&mut a, LibFn::DirichletEta, &[one]);
        assert_evalf_starts_with(&a, eta1, 30, "0.69314718055994530941723212145");
        // s ≤ 0 still goes through ζ: mpmath altzeta(-1/2) = 0.38010481260968401677754215655180836
        let neg_half = a.rational(-1, 2);
        let etam = apply(&mut a, LibFn::DirichletEta, &[neg_half]);
        assert_evalf_starts_with(&a, etam, 30, "0.38010481260968401677754215655");
    }

    #[test]
    fn airy_derivatives_at_exact_zero() {
        let mut a = Arena::new();
        let zero = a.zero;
        // mpmath 1.3: airyai(0, derivative=1) = -0.25881940379280679840518356018920396
        let aip = apply(&mut a, LibFn::AiryAiPrime, &[zero]);
        assert_evalf_starts_with(&a, aip, 30, "-0.25881940379280679840518356018");
        // mpmath 1.3: airybi(0, derivative=1) = 0.44828835735382635791482371039882839
        let bip = apply(&mut a, LibFn::AiryBiPrime, &[zero]);
        assert_evalf_starts_with(&a, bip, 30, "0.44828835735382635791482371039");
    }

    #[test]
    fn polylog_series_bound_accounts_for_negative_order() {
        // s ≥ 0: the classic wp·ln 2 / (−ln|z|) bound (+20).
        let n = polylog_series_terms(2.0, 0.5, 100);
        assert!((120..=122).contains(&n), "{n}");
        // s < 0: |z|^k k^{|s|} peaks near k = |s|/(−ln|z|) before decaying.
        let n = polylog_series_terms(-40.5, 0.5, 132);
        let a = -(0.5f64).ln();
        let k = (n - 20) as f64;
        assert!(
            k * a - 40.5 * k.ln() >= 132.0 * std::f64::consts::LN_2,
            "{n}"
        );
        assert!(n > 200, "{n}");
    }

    /// The `f64` of a constant is rounded once, from its value.  Up to 0.57
    /// `eval_const_f64(√2)` parsed the 16-digit decimal `1.414213562373095`
    /// and was `1.4142135623730949`, an ulp below `SQRT_2` (the correctly
    /// rounded value; mpmath `sqrt(2)` = 1.41421356237309504880).
    #[test]
    fn const_values_round_once() {
        let mut a = Arena::new();
        let two = a.int(2);
        let r2 = a.sqrt(two);
        assert_eq!(eval_const_f64(&mut a, r2), Some(std::f64::consts::SQRT_2));
        let v = const_value(&a, r2, 16).unwrap();
        assert_eq!(v.to_f64(), Some(std::f64::consts::SQRT_2));
        assert_eq!(v.re_sign(), crate::base::truth::SignSet::POSITIVE);
        assert_eq!(v.re_decimal(), Decimal::parse("1.414213562373095", 10));
    }

    /// The parts of a value are read with their certified signs: an
    /// imaginary part far below the real one is kept, and the value has no
    /// real `f64` (up to 0.52 the printout dropped it; up to 0.57 callers
    /// searched the printout for `i`).  A value zero only to the precision
    /// reached has parts of undecided sign.
    #[test]
    fn const_value_parts_carry_certified_signs() {
        use crate::base::truth::SignSet;
        let mut a = Arena::new();
        let m60 = a.int(-60);
        let tiny = a.exp(m60);
        let im = a.mul(&[tiny, a.i_unit]);
        let z = a.add(&[a.one, im]);
        let v = const_value(&a, z, 30).unwrap();
        assert_eq!(v.re_sign(), SignSet::POSITIVE);
        assert_eq!(v.im_sign(), SignSet::POSITIVE);
        assert_eq!(v.to_f64(), None);
        assert!(Decimal::of(&v.z.1, 30).is_some_and(|d| d.magnitude() == -27));
        let zero = const_value(&a, a.zero, 30).unwrap();
        assert_eq!(zero.re_sign(), SignSet::ZERO);
        assert_eq!(zero.to_f64(), Some(0.0));
        // √2·√3 − √6 is not simplified here: zero to the precision reached.
        let (two, three, six) = (a.int(2), a.int(3), a.int(6));
        let (r2, r3, r6) = (a.sqrt(two), a.sqrt(three), a.sqrt(six));
        let p = a.mul(&[r2, r3]);
        let neg = a.neg(r6);
        let d = a.add(&[p, neg]);
        if let Ok(v) = const_value(&a, d, 16)
            && v.settled == Settled::ZeroToPrecision
        {
            assert_eq!(v.re_sign(), SignSet::REAL);
        }
    }
}

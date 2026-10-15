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
use wp::{Guard, Prec, RM, Wp};

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

pub(crate) use quad::nonzero_on;

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
    evalf_with(arena, expr, digits, ZeroSearch::Number, Underflow::Refuse)
}

/// What a decimal result does with a nonzero number below the exponent
/// range ([`Adaptive::Underflow`]).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Underflow {
    /// Refuse it ([`underflow_error`]): 0 is not one of its digits.
    Refuse,
    /// Print `0`, for a caller that reads the result as an `f64`, whose
    /// correctly rounded value it is.
    Zero,
}

/// [`evalf`] with the [`ZeroSearch`] `search` for a value that is zero to
/// the working precision, and the [`Underflow`] policy `underflow`.
fn evalf_with(
    arena: &Arena,
    expr: ExprId,
    digits: u32,
    search: ZeroSearch,
    underflow: Underflow,
) -> Result<String, SymplexError> {
    // Convert decimal digits to binary precision with guard bits.
    // log2(10) ≈ 3.3219, so we use digits * 3.4 + 64 extra bits.
    let binary_prec = (digits as usize) * 34 / 10 + 64;
    // Ensure a reasonable minimum.
    let prec = binary_prec.max(128);

    debug!(
        digits = digits,
        binary_precision = prec,
        "evalf: starting numerical evaluation"
    );

    // Enforce the configured maximum precision.
    let max_prec = arena.config.max_evalf_precision as usize;
    if prec > max_prec {
        return Err(SymplexError::PrecisionExhausted {
            requested: digits,
            achieved: (max_prec * 10 / 34).saturating_sub(6) as u32,
        });
    }

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

    let mut cc = Consts::new().map_err(|e| {
        SymplexError::NotImplemented(format!("astro-float constants init failed: {e:?}"))
    })?;

    match evaluate_adaptive(arena, expr, digits, search, &mut cc)? {
        Adaptive::Settled(result, _) => format_complex(&result, digits),
        Adaptive::Underflow(_)
        | Adaptive::Beyond(_)
        | Adaptive::Undecided
        | Adaptive::Hidden { .. }
            if underflow == Underflow::Zero =>
        {
            Ok("0".to_owned())
        }
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
    with_f64_consts(|cc| {
        Ok((high != f64::INFINITY
            || improper::f64_tail_trusted(arena, body, var, true, compiled, allowed, cc))
            && (low != f64::NEG_INFINITY
                || improper::f64_tail_trusted(arena, body, var, false, compiled, allowed, cc)))
    })
    .unwrap_or(false)
}

/// The working precision behind `digits` decimal digits: `3.4` bits per
/// digit plus 64 guard bits, at least 128.
fn working_precision(digits: u32) -> usize {
    ((digits as usize) * 34 / 10 + 64).max(128)
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
            Err(SymplexError::Unevaluable { .. }) if tainted_child() => {
                Ok(CBall::unknown(c_zero(prec)))
            }
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
                balls.insert(id, CBall::new(value, e));
            }
            Err(e) => {
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
            for &c in children.iter() {
                let v = get_cached(cache, c)?;
                parts.push((v, cache.err(&c)));
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
                Err(SymplexError::Unevaluable { .. }) if inexact_zero_child(node, cache) => {
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
                Err(SymplexError::Unevaluable { .. }) if inexact_zero_child(node, cache) => {
                    return Ok(CBall::new(c_zero(prec), accuracy::Bound::UNKNOWN));
                }
                Err(e) => return Err(e),
            };
            let err = cspecial::bound(arena, id, f, &args, &routed, cache, w);
            CBall::new(routed.value, err)
        }
        // Elliptic integrals of a non-real parameter report their own bound
        // (`elliptic.rs`; the sensitivity rules are for real arguments).
        ExprNode::Lib(f, args) if elliptic::nonreal_args(*f, args, cache) => {
            let value = eval_node(arena, id, cache, w)?;
            let err = elliptic::bound(arena, id, args, &value, cache, w);
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
                Err(SymplexError::Unevaluable { .. })
                    if inexact_zero_child(node, cache) || rounded_onto_pole(node, cache) =>
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
                Err(SymplexError::Unevaluable { .. }) if rounded_onto_pole(node, cache) => {
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
                Err(SymplexError::Unevaluable { .. })
                    if inexact_zero_child(arena.node(id), cache)
                        || rounded_onto_pole(arena.node(id), cache) =>
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
        if finite && let Some((v, zero_part)) = settle(&value, bound, digits, needed, prec, search)
        {
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
        let lost_below_range = absorbed.is_some_and(|a| a.below_range);
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
/// The flag says that a part was set to 0 although it is not exactly 0
/// ([`evaluate_adaptive`] pursues such a part of a number to the cap).
fn settle(
    value: &Complex,
    bound: accuracy::Bound,
    digits: u32,
    needed: i64,
    prec: usize,
    search: ZeroSearch,
) -> Option<(Complex, bool)> {
    if accuracy::mag(value).is_none() {
        return bound.is_exact().then(|| (value.clone(), false));
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
    Some((out, zero_re || zero_im))
}

// ═══════════════════════════════════════════════════════════════════════════
// Direct `f64` / `Complex64` results (no decimal string in between)
// ═══════════════════════════════════════════════════════════════════════════

/// Decimal digits requested by `Ex::eval_f64` and `Ex::eval_complex64`:
/// 16 digits map to a 128-bit working precision (see [`evalf`]), the same
/// precision [`evalf_f64`] and [`evalf_complex64`] evaluate at.
pub(crate) const F64_DIGITS: u32 = 16;

thread_local! {
    /// The constants cache of the `f64` route, kept per thread.
    ///
    /// `Consts::new()` alone costs ~170 µs and a cold cache recomputes π
    /// and `ln 2` on every `sin`/`exp` (`exp(2)` at 128 bits: 475 µs cold,
    /// 2.5 µs warm), which dominated every `eval_f64`.  The route always
    /// evaluates at the same precision, so the cache stays small, and
    /// astro-float rounds each constant correctly to the precision asked
    /// for, so a warm cache returns the same bits a cold one would.
    static F64_CONSTS: std::cell::RefCell<Option<Consts>> = const { std::cell::RefCell::new(None) };
}

/// Run `f` with this thread's constants cache, creating it on first use.
///
/// The cache is taken out of its slot for the duration of `f` and put back
/// afterwards, so a nested evaluation on the same thread (a definite
/// integral's integrand, say) finds the slot empty and simply uses a fresh
/// cache instead of panicking on a double borrow.
fn with_f64_consts<T>(
    f: impl FnOnce(&mut Consts) -> Result<T, SymplexError>,
) -> Result<T, SymplexError> {
    let mut cc = match F64_CONSTS.with(|slot| slot.borrow_mut().take()) {
        Some(cc) => cc,
        None => Consts::new().map_err(|e| {
            SymplexError::NotImplemented(format!("astro-float constants init failed: {e:?}"))
        })?,
    };
    let out = f(&mut cc);
    F64_CONSTS.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            *slot = Some(cc);
        }
    });
    out
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
    let prec = ((digits as usize) * 34 / 10 + 64).max(128);
    // A number below the exponent range is 0 within its bound, and not
    // known to be 0 by that bound: zero to the precision, for the tolerance
    // checks of the crate and for the `f64` routes, whose correctly rounded
    // value it is.
    evalf_adaptive_value(arena, expr, digits, search).map(|a| match a {
        Adaptive::Settled(z, settled) => (z, settled),
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
    let prec = ((digits as usize) * 34 / 10 + 64).max(128);
    let max_prec = arena.config.max_evalf_precision as usize;
    if prec > max_prec {
        return Err(SymplexError::PrecisionExhausted {
            requested: digits,
            achieved: (max_prec * 10 / 34).saturating_sub(6) as u32,
        });
    }

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

    with_f64_consts(|cc| evaluate_adaptive(arena, expr, digits, search, cc))
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
    let prec = ((digits as usize) * 34 / 10 + 64).max(128);
    match evalf_adaptive_value(arena, expr, digits, search)? {
        Adaptive::Settled(z, settled) => Ok((z, settled)),
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
        Adaptive::Underflow(_) | Adaptive::Undecided | Adaptive::Hidden { .. } => return None,
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
    let (z, _) = evalf_value(arena, expr, digits, ZeroSearch::Number)?;
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

    // Fall back to arbitrary-precision evaluation to 16 decimal digits,
    // then parse the resulting string.  The `evalf` call takes `&Arena`
    // (immutable), which Rust allows via automatic reborrowing of our
    // `&mut Arena`.
    match evalf_with(arena, evaled, 16, ZeroSearch::Cap, Underflow::Zero) {
        Ok(s) => {
            let result = s.parse::<f64>().ok();
            tracing::trace!(?result, decimal_str = %s, "eval_const_f64: evalf path");
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
                return Err(SymplexError::Unevaluable {
                    reason: "Ci(0) is zoo".into(),
                });
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
                return Err(SymplexError::Unevaluable {
                    reason: "Ei(0) is -∞".into(),
                });
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
                return Err(SymplexError::Unevaluable {
                    reason: "li(1) is -∞".into(),
                });
            }
            debug!(prec, "evalf: li via Ei(ln x)");
            let r = arb_ei(&ln_x, w)?;
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
            Err(SymplexError::Unevaluable {
                reason: "cannot evaluate infinity to finite precision".into(),
            })
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
                        let pow_pos = b.0.powi((-n) as usize, prec, RM);
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
                return Err(SymplexError::Unevaluable {
                    reason: if e.1.is_zero() {
                        "cannot evaluate infinity to finite precision".into()
                    } else {
                        "0 to a non-real power has no value".into()
                    },
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

// ═══════════════════════════════════════════════════════════════════════════
// Definite-integral helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Does the tree contain a `DefiniteIntegral` node?
fn contains_definite_integral(arena: &Arena, root: ExprId) -> bool {
    let mut stack = vec![root];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        let node = arena.node(id);
        if matches!(node, ExprNode::DefiniteIntegral(..)) {
            return true;
        }
        node.for_each_child(|c| stack.push(c));
    }
    false
}

/// An integration bound as `f64`, with a bound on its distance to the true
/// bound (the rounding to `f64`, and the evaluated value's own error):
/// `±∞` nodes directly (exact), anything else from the evaluated cache
/// (must be real).
fn definite_bound_f64(
    arena: &Arena,
    cache: Cache<'_>,
    id: ExprId,
) -> Result<(f64, f64), SymplexError> {
    match arena.node(id) {
        ExprNode::Infinity => Ok((f64::INFINITY, 0.0)),
        ExprNode::NegInfinity => Ok((f64::NEG_INFINITY, 0.0)),
        _ => {
            let v = get_cached(cache, id)?;
            if !v.1.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "integration bounds of a definite integral must be real".into(),
                });
            }
            let x = f64_of(&v.0)?;
            let bound = cache.err(&id);
            if bound.is_unknown() || !x.is_finite() {
                return Err(SymplexError::PrecisionExhausted {
                    requested: F64_DIGITS,
                    achieved: 0,
                });
            }
            let exp2 = |e: accuracy::ErrExp| {
                if accuracy::is_exact(e) {
                    0.0
                } else {
                    e.clamp(-1100.0, 1100.0).exp2()
                }
            };
            // |x − v| ≤ ½ ulp(x) ≤ |x|·2⁻⁵³; exact when `v` is an `f64`.
            let exact_in_f64 = bigfloat::from_f64(x, 64) == v.0;
            let conversion = if exact_in_f64 {
                0.0
            } else {
                x.abs() * f64::EPSILON / 2.0
            };
            Ok((x, conversion + exp2(bound.re) + exp2(bound.im)))
        }
    }
}

/// Multiple of `ε·∫|f|` (`ε = 2⁻⁵²`) that bounds the rounding of the `f64`
/// integrand values in a quadrature sum: each value carries a few roundings
/// of its own and the condition number of its evaluation (`sin(1/x)` near
/// `x = 10⁻³` loses three digits to the rounding of `1/x`).
const QUADRATURE_ROUNDING_FACTOR: f64 = 16.0;

/// Decimal digits served by [`evalf_complex64`] (`eval_f64`,
/// `eval_complex64`) for an expression containing a definite integral: the
/// cheap `f64` Gauss–Kronrod estimate is returned when its error bound
/// certifies this many digits (it seldom certifies the 16 of the rest of the
/// `f64` route: its own rounding floor is `50ε` per segment); otherwise the
/// arbitrary-precision quadrature serves it, to every digit of the `f64`.
/// Before 0.29 it was served at a tolerance of `10⁻¹⁰`, accepted up to
/// `10⁻⁷`, with no check at all.
pub(crate) const QUADRATURE_F64_DIGITS: u32 = 8;

/// The highest working precision at which a definite integral is tried by
/// `f64` quadrature first: `evalf`'s minimum, the precision of every
/// request up to 18 digits.
const QUADRATURE_F64_MAX_PREC: usize = 128;

/// A definite integral whose bounds exceed its width by more than `2^this`
/// (or round to one `f64`) is evaluated by its closed form first, when it
/// has one: an `f64` node has fewer than 5 bits inside such a range, and
/// the arbitrary-precision ones need this many bits more than the working
/// precision.  Up to 0.51 the quadrature of `∫ |cos(7x + 1/2)| dx` over
/// `[10¹⁰⁰ + 1/3, 10¹⁰⁰ + 28/3]` spent 17 s before refusing.
const FAR_RANGE_BITS: f64 = 48.0;

/// `∫_lo^hi body d(var)` with an error bound: the quadratures of
/// [`eval_definite_integral_numeric`], and when they refuse (the integrand
/// has no bound at the nodes, the quadrature does not converge) the closed
/// form of the definite integrator evaluated at the working precision, or
/// its proof of divergence (`calculus::definite::closed_form`, sought once
/// per integral; once found it serves the higher precisions directly).
/// Up to 0.51 `eval_f64` of `∫ |sin(x/3 + 1/2)| dx` over
/// `[10¹⁰⁰ + 7/10, 10¹⁰⁰ + 100]` was refused (in 16 s at 0.50): the
/// nodes need 330 bits before the first fractional one of `x/3`.
#[allow(clippy::too_many_arguments)]
fn eval_definite_integral(
    arena: &Arena,
    body: ExprId,
    var: ExprId,
    lo: ExprId,
    hi: ExprId,
    cache: Cache<'_>,
    w: &mut Wp<'_>,
) -> Result<CBall, SymplexError> {
    // The value of a closed form, its proof of divergence, or `None`.
    let serve = |closed: &quad::ClosedForm, w: &mut Wp<'_>| match &**closed {
        Ok(Some((pa, value))) => Some(closed_form_ball(pa, *value, w)),
        Err(divergent) => Some(Err(divergent.clone())),
        Ok(None) => None,
    };
    let seek = || crate::calculus::definite::closed_form(arena, body, var, lo, hi);
    let key = quad::memo_key(arena, body, var, lo, hi);
    // Bounds far out next to the width: every node needs that many bits
    // more than the working precision, and the closed form comes first.
    let far = match (
        definite_bound_f64(arena, cache, lo),
        definite_bound_f64(arena, cache, hi),
    ) {
        (Ok((a, _)), Ok((b, _))) if a.is_finite() && b.is_finite() => {
            a == b || (a.abs().max(b.abs()) / (b - a).abs()).log2() > FAR_RANGE_BITS
        }
        _ => false,
    };
    let known = if far {
        quad::closed_form(key, seek)
    } else {
        quad::recall_closed_form(&key)
    };
    if let Some(closed) = known
        && let Some(served) = serve(&closed, w)
        && (served.is_ok() || matches!(served, Err(SymplexError::Divergent { .. })))
    {
        return served;
    }
    let refused = match eval_definite_integral_numeric(arena, body, var, lo, hi, cache, w) {
        Err(e) if !matches!(e, SymplexError::Divergent { .. }) => e,
        done => return done,
    };
    let Some(closed) = quad::closed_form(key, seek) else {
        return Err(refused);
    };
    match serve(&closed, w) {
        Some(Ok(ball)) => Ok(ball),
        Some(Err(e @ SymplexError::Divergent { .. })) => Err(e),
        _ => Err(refused),
    }
}

/// The closed form `value` (in the private arena `pa`) of a definite
/// integral as a ball for `w`'s precision: evaluated at that precision, and
/// at twice as many bits, and so on to the configured maximum, while it
/// certifies fewer bits than that (or holds 0).  A closed form may cancel
/// where the quadrature it replaces did not — `∫ sin(10²⁰x) dx` over
/// `[10²⁰, 10²⁰ + 2·10⁻¹⁹]` is `(cos(10⁴⁰) − cos(10⁴⁰ + 20))/10²⁰` — and
/// a ball around 0 within the digits asked is read as an integral equal to
/// 0 ([`evaluate_adaptive`]): in a draft, `eval_f64` gave `0` for
/// `−1.0065408055519122·10⁻²⁰`.
fn closed_form_ball(pa: &Arena, value: ExprId, w: &mut Wp<'_>) -> Result<CBall, SymplexError> {
    let order = walk::post_order_ids(pa, value);
    let target = w.bits();
    let cap = (pa.config.max_evalf_precision as usize).max(target);
    let mut bits = target;
    loop {
        let ball = evaluate_tree(pa, value, &order, None, false, &mut w.at(bits))?;
        let enough = ball.err.is_exact()
            || accuracy::accurate_bits(&ball.mid, ball.err.joint())
                .is_some_and(|a| a >= target as f64);
        if enough || bits >= cap {
            return Ok(ball);
        }
        bits = (2 * bits).min(cap);
    }
}

/// `∫_lo^hi body d(var)` with an error bound by quadrature.
///
/// At the lowest working precision ([`QUADRATURE_F64_MAX_PREC`]) the `f64`
/// adaptive Gauss–Kronrod quadrature is tried first
/// ([`eval_definite_integral_f64`]) and its value served when it resolves
/// the integral to [`QUADRATURE_F64_DIGITS`] digits: `Ex::eval_f64` of a
/// smooth integrand stays a matter of microseconds.  Anything else — more
/// digits, an endpoint singularity the `f64` bisection cannot resolve, an
/// integrand the `f64` compiler does not support — goes to the
/// arbitrary-precision tanh-sinh quadrature (`quad`): the body and the
/// bounds are copied into a private arena (the bounds re-evaluated at every
/// precision a node needs; one that depends on an enclosing bound variable
/// keeps its cached value), interior breakpoints are located there by the
/// definite integrator's scanner, and the pieces are integrated to
/// `2^−(prec − 48)·∫|f|`.  At the lowest precision, on `[0, ∞)`, `(−∞, 0]`
/// and the whole line (where its nodes keep their distance to the finite
/// end), the `f64` quadrature under the endpoint-smoothing substitution
/// (its raw error estimates resolved) is the fallback when the
/// arbitrary-precision one fails or certifies less — an oscillating tail
/// tanh-sinh does not resolve, `∫₀^∞ sin x·e^{−√x}/(2√x) dx` — and the
/// latter then gets a quarter of its work.  The DiracDelta of the variable is refused: 0 at every node,
/// its integral is no sum of values (before 0.43 `∫₋₁¹ δ(x) dx` was `0`).
///
/// An `f64` value over an infinite range is served only when the integrand's
/// mass beyond `|x| = 2¹⁶`, measured by certified evaluations out to
/// `2²⁰⁴⁸`, is within its bound or was seen by the compiled body
/// (`improper::f64_tail_trusted`), and the smoothed one never when the
/// arbitrary-precision quadrature found the terms not decaying: an `f64`
/// body whose intermediate values overflow is 0 beyond the overflow, and
/// before 0.47 `∫_{−∞}^{∞} |x|/((x² + 1)π) dx`, which diverges, was
/// `225.93`.  An integral refused for terms that do not decay is
/// [`SymplexError::Divergent`] when the comparison tests prove it
/// (`improper::divergence`).
///
/// Before 0.43 an integral was refused beyond 16
/// digits, an endpoint singularity was certified to 8 or 9 digits with a
/// bound its error exceeded (`∫₀¹ x⁴e^{x/10}/(√(1 − x)·B(5, ½)) dx`,
/// mpmath `1.0952385618059876`, was `1.0952385480452327` from `eval_f64`
/// with a bound below `10⁻⁹`), `∫₀¹ x^(−9/10) dx` did not converge, and
/// an integral equal to 0 was refused.
#[allow(clippy::too_many_arguments)]
fn eval_definite_integral_numeric(
    arena: &Arena,
    body: ExprId,
    var: ExprId,
    lo: ExprId,
    hi: ExprId,
    cache: Cache<'_>,
    w: &mut Wp<'_>,
) -> Result<CBall, SymplexError> {
    let prec = w.bits();
    if !matches!(arena.node(var), ExprNode::Symbol(_)) {
        return Err(SymplexError::Unevaluable {
            reason: "integration variable of a definite integral must be a symbol".into(),
        });
    }
    // `DiracDelta` is 0 at every node a quadrature can sample: its integral
    // is not a sum of values.
    if walk::post_order_ids(arena, body).iter().any(|&id| {
        matches!(arena.node(id), ExprNode::DiracDelta(inner) if walk::contains(arena, *inner, var))
    }) {
        return Err(SymplexError::Unevaluable {
            reason: "a definite integral of DiracDelta of the integration variable has no \
                     numerical value (integrate it symbolically)"
                .into(),
        });
    }
    if prec > QUADRATURE_F64_MAX_PREC {
        return eval_definite_integral_bigfloat(arena, body, var, lo, hi, cache, w, false)
            .map_err(SymplexError::from);
    }
    if let Some(served) = eval_definite_integral_f64(arena, body, var, lo, hi, cache, w, false)? {
        return Ok(served);
    }
    // An endpoint singularity the plain bisection cannot resolve: the
    // arbitrary-precision quadrature, unless it fails or certifies less than
    // the smoothed `f64` one (an oscillating tail it does not resolve,
    // `∫₀^∞ sin x·e^{−√x}/√x dx`), with less work when the latter exists.
    let fallback = eval_definite_integral_f64(arena, body, var, lo, hi, cache, w, true)?;
    let big =
        eval_definite_integral_bigfloat(arena, body, var, lo, hi, cache, w, fallback.is_some());
    match (big, fallback) {
        (Ok(b), Some(f)) if b.err.joint() > f.err.joint() => Ok(f),
        (Ok(b), _) => Ok(b),
        // Terms that do not decay far beyond the reach of `f64` (out to
        // `x ≈ 10¹¹⁰⁰⁰⁰`, or next to a breakpoint): the `f64` value did not
        // see them.
        (Err(e), Some(_))
            if matches!(
                e,
                quad::QuadError::NotDecaying
                    | quad::QuadError::Failed(SymplexError::Divergent { .. })
            ) =>
        {
            Err(e.into())
        }
        (Err(_), Some(f)) => Ok(f),
        (Err(e), None) => Err(e.into()),
    }
}

/// A bound of a definite integral for `quad`: `None` for `±∞` (with its
/// sign), otherwise a finite end in the private arena `pa`.
fn quadrature_end(
    arena: &Arena,
    id: ExprId,
    cache: Cache<'_>,
    pa: &mut Arena,
    map: &mut FxHashMap<ExprId, ExprId>,
) -> Result<Result<quad::End, bool>, SymplexError> {
    match arena.node(id) {
        ExprNode::Infinity => return Ok(Err(true)),
        ExprNode::NegInfinity => return Ok(Err(false)),
        ExprNode::Num(n) => return Ok(Ok(quad::End::Rational(arena.num(*n).clone()))),
        _ => {}
    }
    if walk::free_symbols(arena, id).is_empty() {
        let p = crate::base::compact::transfer_subtree(arena, pa, id, map);
        return Ok(Ok(quad::End::Expr(p)));
    }
    // A bound that depends on an enclosing bound variable: its cached value.
    let v = get_cached(cache, id)?;
    let b = cache.err(&id);
    if b.is_unknown() || !accuracy::exact_zero(&v.1, b.im) || v.0.is_inf() || v.0.is_nan() {
        return Err(SymplexError::Unevaluable {
            reason: "integration bounds of a definite integral must be real".into(),
        });
    }
    Ok(Ok(quad::End::Value(v.0.clone(), b.re)))
}

/// The arbitrary-precision route of [`eval_definite_integral`].
#[allow(clippy::too_many_arguments)]
fn eval_definite_integral_bigfloat(
    arena: &Arena,
    body: ExprId,
    var: ExprId,
    lo: ExprId,
    hi: ExprId,
    cache: Cache<'_>,
    w: &mut Wp<'_>,
    light: bool,
) -> Result<CBall, quad::QuadError> {
    let prec = w.bits();
    debug!(
        prec,
        light, "evalf: DefiniteIntegral — tanh-sinh quadrature"
    );
    let key = quad::memo_key(arena, body, var, lo, hi);
    if let Some(recalled) = quad::recall(&key, prec) {
        debug!(
            prec,
            "evalf: a quadrature that did not converge at a lower precision"
        );
        let (value, err, real) = recalled?;
        return Ok(quadrature_value(value, err, real, prec));
    }
    let mut pa = Arena::with_config(arena.config.clone());
    let mut map: FxHashMap<ExprId, ExprId> = FxHashMap::default();
    let body_p = crate::base::compact::transfer_subtree(arena, &mut pa, body, &mut map);
    let var_p = crate::base::compact::transfer_subtree(arena, &mut pa, var, &mut map);
    let lo_e = quadrature_end(arena, lo, cache, &mut pa, &mut map)?;
    let hi_e = quadrature_end(arena, hi, cache, &mut pa, &mut map)?;
    let exact_zero = || Ok(CBall::exact(c_zero(prec)));
    let probe = prec + 32;
    let mut q = quad::Quad::new(&pa, body_p, var_p, Vec::new(), prec);
    // The range from below to above (`None`: −∞ below, +∞ above), and
    // whether it was reversed.
    let (lo_e, hi_e, negate) = match (lo_e, hi_e) {
        (Err(a), Err(b)) if a == b => return exact_zero(),
        (Err(false), Err(_)) => (None, None, false),
        (Err(_), Err(_)) => (None, None, true),
        (Ok(a), Err(true)) => (Some(a), None, false),
        (Err(false), Ok(b)) => (None, Some(b), false),
        (Err(true), Ok(b)) => (Some(b), None, true),
        (Ok(a), Err(false)) => (None, Some(a), true),
        (Ok(a), Ok(b)) => {
            if quad::same_end(&a, &b) {
                return exact_zero();
            }
            let (av, _) = q.end_value(&a, &mut w.at(probe))?;
            let (bv, _) = q.end_value(&b, &mut w.at(probe))?;
            match av.cmp(&bv) {
                Some(c) if c < 0 => (Some(a), Some(b), false),
                Some(c) if c > 0 => (Some(b), Some(a), true),
                // Equal to the precision: no value yet.
                _ => return Ok(CBall::new(c_zero(prec), accuracy::Bound::UNKNOWN)),
            }
        }
    };
    let mut f64_of = |e: &Option<quad::End>, inf: f64| -> Result<f64, SymplexError> {
        match e {
            Some(e) => f64_of(&q.end_value(e, &mut w.at(probe))?.0),
            None => Ok(inf),
        }
    };
    let lo_f = f64_of(&lo_e, f64::NEG_INFINITY)?;
    let hi_f = f64_of(&hi_e, f64::INFINITY)?;
    drop(q);
    // Interior breakpoints, located in the private arena.
    let range = crate::base::interval::Interval::closed(lo_f, hi_f);
    let points = quadrature_breakpoints(
        &mut pa,
        body_p,
        var_p,
        lo_e.as_ref(),
        hi_e.as_ref(),
        range,
        prec,
    )?;
    debug!(
        lo_f,
        hi_f,
        breakpoints = points.len(),
        "evalf: quadrature pieces"
    );
    // The body shifted to each bound and breakpoint `e` other than 0,
    // `body(e ± x)`: the nodes next to `e` are placed at their distance from
    // it, which an algebraic or logarithmic singularity at `e` then sees
    // exactly (`(x − 1/3)^(−9/10)` becomes `x^(−9/10)`) instead of through a
    // cancellation that needs ten times the precision.
    let mut variants: Vec<(quad::End, bool, ExprId)> = Vec::new();
    let mut shift = |pa: &mut Arena, e: &quad::End, plus: bool| {
        let anchor = match e {
            quad::End::Rational(q) if !q.is_zero() => pa.num_ratio(q.clone()),
            quad::End::Expr(id) => *id,
            _ => return,
        };
        let moved = if plus {
            pa.add(&[anchor, var_p])
        } else {
            pa.sub(anchor, var_p)
        };
        let shifted = crate::transforms::subs::subs(pa, body_p, var_p, moved);
        variants.push((e.clone(), plus, shifted));
    };
    if let Some(a) = &lo_e {
        shift(&mut pa, a, true);
    }
    if let Some(b) = &hi_e {
        shift(&mut pa, b, false);
    }
    for p in &points {
        shift(&mut pa, p, false);
        shift(&mut pa, p, true);
    }
    // Where a refusal is examined for a proof of divergence: the ends and
    // both sides of every breakpoint.
    let mut approaches: Vec<improper::Approach> = Vec::new();
    let anchor = |pa: &mut Arena, e: &quad::End| match e {
        quad::End::Rational(q) => Some(pa.num_ratio(q.clone())),
        quad::End::Expr(id) => Some(*id),
        _ => None,
    };
    match &lo_e {
        None => approaches.push(improper::Approach::NegInf),
        Some(a) => approaches.extend(anchor(&mut pa, a).map(improper::Approach::Above)),
    }
    match &hi_e {
        None => approaches.push(improper::Approach::PosInf),
        Some(b) => approaches.extend(anchor(&mut pa, b).map(improper::Approach::Below)),
    }
    for p in &points {
        if let Some(c) = anchor(&mut pa, p) {
            approaches.push(improper::Approach::Below(c));
            approaches.push(improper::Approach::Above(c));
        }
    }
    let pieces = quad::pieces(lo_e, hi_e, points);
    let mut q = quad::Quad::new(&pa, body_p, var_p, variants, prec);
    if light {
        q.lighten();
    }
    let run = q.run(pieces, w.cc);
    drop(q);
    let out = match run {
        Ok(out) => out,
        Err(e) => {
            // Terms that do not decay, or an integrand without a value at a
            // far node (`eˣ` overflows): a proof of divergence is sought.
            let e = if e.may_diverge() {
                match improper::divergence(&mut pa, body_p, var_p, &approaches) {
                    Some(reason) => quad::QuadError::Failed(SymplexError::Divergent {
                        operation: "evalf",
                        reason: format!("the definite integral diverges: {reason}"),
                    }),
                    None => e,
                }
            } else {
                e
            };
            let divergent = matches!(e, quad::QuadError::Failed(SymplexError::Divergent { .. }));
            if (!light || divergent) && e.is_node_failure() {
                quad::remember(key, prec, Err(e.clone()));
            }
            return Err(e);
        }
    };
    let value = if negate { c_neg(&out.value) } else { out.value };
    if !light && !out.converged && out.discretisation {
        quad::remember(key, prec, Ok((value.clone(), out.err, out.real)));
    }
    debug!(
        err = out.err,
        out.real, out.converged, "evalf: DefiniteIntegral evaluated"
    );
    Ok(quadrature_value(value, out.err, out.real, prec))
}

/// A quadrature's sum `value` with the error bound `2^err` as a node value
/// at `prec` bits: real when every integrand value was.  A sum of exact
/// zeros is no proof of an exact zero: its bound is `2^(−4·prec)`.
fn quadrature_value(mut value: Complex, err: accuracy::ErrExp, real: bool, prec: usize) -> CBall {
    let err = if accuracy::is_exact(err) {
        -(4.0 * prec as f64)
    } else {
        err
    };
    if real {
        value.1 = BigFloat::new(prec);
    }
    let bound = accuracy::Bound {
        re: err,
        im: if real { accuracy::EXACT } else { err },
    };
    CBall::new(value, bound)
}

/// The interior breakpoints of `body` strictly between the ends `lo` and
/// `hi` (`None`: infinite), whose values are `range`, for `quad`: the
/// definite integrator's breakpoint scan between the exact ends (the `f64`
/// ones when an end is not exact), the kinks it cannot place located to
/// `prec + 16` bits.  An error for a family of more than 10,000 poles,
/// kinks or steps, or kinks that cannot be located.
fn quadrature_breakpoints(
    pa: &mut Arena,
    body: ExprId,
    var: ExprId,
    lo: Option<&quad::End>,
    hi: Option<&quad::End>,
    range: crate::base::interval::Interval<f64>,
    prec: usize,
) -> Result<Vec<quad::End>, SymplexError> {
    use crate::base::interval::Interval;
    let lo_x = lo.and_then(|e| e.exact(pa));
    let hi_x = hi.and_then(|e| e.exact(pa));
    let exact = lo_x.zip(hi_x).map(|(l, h)| Interval::closed(l, h));
    let bits = u32::try_from(prec.saturating_add(16)).unwrap_or(u32::MAX);
    let points =
        crate::calculus::definite::quadrature_breakpoints(pa, body, var, range, exact, Some(bits))?;
    Ok(points
        .into_iter()
        .map(|(p, _)| match pa.as_num(p) {
            Some(r) => quad::End::Rational(r.clone()),
            None => quad::End::Expr(p),
        })
        .collect())
}

/// `∫_lo^hi body d(var)` by adaptive `f64` Gauss–Kronrod quadrature
/// (`quadrature_resolved` of the definite integrator), with an honest error
/// bound, when it resolves the integral to [`QUADRATURE_F64_DIGITS`] digits;
/// `None` otherwise (the arbitrary-precision quadrature takes over).
///
/// The body is compiled to a stack-VM function of the integration variable
/// (no arena mutation) and the bounds come from the cache (`±∞` allowed).
/// The quadrature is asked for the accuracy the working precision wants,
/// down to `10⁻¹³` relative (below that it only runs into its own
/// rounding floor).  Only plain bisection is used, and only a run whose
/// every segment was resolved is served: the endpoint-smoothing retry of
/// `quadrature` gives no bound at an algebraic endpoint singularity.  The
/// reported bound is the sum of
///
/// * the Gauss–Kronrod error estimate,
/// * the rounding of the `f64` integrand values,
///   [`QUADRATURE_ROUNDING_FACTOR`]`·ε·∫|f|` — `∫|f|`, not `|∫f|`, so an
///   oscillating integrand whose integral cancels is not credited with
///   digits its values do not have,
/// * the rounding of finite bounds to `f64` (and their own error) times the
///   largest integrand value seen,
///
/// so [`evaluate_adaptive`] refuses the digits the quadrature does not
/// have.  Before 0.29 the result was trusted to the working precision less
/// 16 bits: `eval_decimal(16)` of `∫₀¹ xˣ dx` printed
/// `0.7834305107121128` (mpmath: `0.78343051071213441…`), three wrong
/// digits.
#[allow(clippy::too_many_arguments)]
fn eval_definite_integral_f64(
    arena: &Arena,
    body: ExprId,
    var: ExprId,
    lo: ExprId,
    hi: ExprId,
    cache: Cache<'_>,
    w: &mut Wp<'_>,
    smoothing: bool,
) -> Result<Option<CBall>, SymplexError> {
    let prec = w.bits();
    use crate::calculus::definite::{QuadOpts, QuadResult, quadrature, quadrature_resolved};
    debug!("evalf: DefiniteIntegral — f64 Gauss–Kronrod quadrature");
    let var_name = match arena.node(var) {
        ExprNode::Symbol(sid) => arena.symbol_name(*sid).to_string(),
        _ => return Ok(None),
    };
    let (Ok((a, a_err)), Ok((b, b_err))) = (
        definite_bound_f64(arena, cache, lo),
        definite_bound_f64(arena, cache, hi),
    ) else {
        return Ok(None);
    };
    // Across a kink or a step the estimate is no bound: split at them
    // (all located), or leave the integral to the arbitrary-precision
    // quadrature.
    let kinks = if crate::calculus::definite::has_kink(arena, body, var) {
        match crate::calculus::definite::f64_pieces(arena, body, var, a.min(b), a.max(b)) {
            Some(p) if !smoothing => Some(p),
            _ => return Ok(None),
        }
    } else {
        None
    };
    // The smoothed run is trusted only where its nodes keep their distance
    // to a finite end, which in `f64` they do next to 0 alone (see
    // `quadrature_resolved`): on `[0, ∞)`, `(−∞, 0]` and the whole line.
    // (Float evidence: an end is 0 only with an exact error, `a_err == 0`;
    // every other case refuses.)
    let half_lines = (a == 0.0 && b == f64::INFINITY)
        || (a == f64::NEG_INFINITY && b == 0.0)
        || (a == 0.0 && b == f64::NEG_INFINITY)
        || (a == f64::INFINITY && b == 0.0)
        || (a.is_infinite() && b.is_infinite());
    if smoothing && (!half_lines || a_err != 0.0 || b_err != 0.0) {
        return Ok(None);
    }
    let Ok(func) = crate::output::lambdify::compile_raw(arena, body, &[&var_name]) else {
        return Ok(None);
    };
    let peak = std::cell::Cell::new(0.0f64);
    let f = |t: f64| {
        let v = func(&[t]);
        if v.is_finite() {
            peak.set(peak.get().max(v.abs()));
        }
        v
    };
    // ∫|f|, to the accuracy a rounding estimate needs: the rounding floor
    // of the quadrature sum, below which its tolerance is pointless.
    let abs_f = |t: f64| func(&[t]).abs();
    let loose = QuadOpts {
        rel_tol: 1e-3,
        abs_tol: 0.0,
        max_subdivisions: 400,
    };
    let l1 = quadrature(&abs_f, a, b, &loose).map_or(f64::INFINITY, |r| r.value + r.error);
    let floor = QUADRATURE_ROUNDING_FACTOR * f64::EPSILON * l1;
    let default = QuadOpts::default();
    let wanted = 2f64.powi(
        -i32::try_from(prec.saturating_sub(64))
            .unwrap_or(i32::MAX)
            .min(1000),
    );
    let opts = QuadOpts {
        rel_tol: wanted.max(1e-13).min(default.rel_tol),
        abs_tol: if floor.is_finite() {
            floor
        } else {
            default.abs_tol
        },
        ..default
    };
    let run = match &kinks {
        Some(k) => crate::calculus::definite::quadrature_resolved_split(
            &f,
            a.min(b),
            a.max(b),
            &k.cuts,
            &opts,
        )
        .map(|r| {
            r.map(|r| QuadResult {
                value: if a > b { -r.value } else { r.value },
                ..r
            })
        }),
        None => quadrature_resolved(&f, a, b, &opts, smoothing),
    };
    let Ok(Some(QuadResult { value, error })) = run else {
        return Ok(None);
    };
    let rounding = QUADRATURE_ROUNDING_FACTOR * f64::EPSILON * l1.max(value.abs());
    let endpoints = (a_err + b_err) * peak.get();
    let misplaced = kinks.as_ref().map_or(0.0, |k| k.misplaced) * peak.get();
    let total = error + rounding + endpoints + misplaced;
    debug!(
        value,
        error, rounding, endpoints, "evalf: DefiniteIntegral evaluated in f64"
    );
    // Served only when it certifies the digits of `eval_f64`; a zero bound
    // (a zero integrand at every node) is no proof of an exact zero.  (Both
    // tests only refuse.)
    let needed = f64::from(QUADRATURE_F64_DIGITS) * 3.322 + 4.0;
    if !value.is_finite()
        || !total.is_finite()
        || total <= 0.0
        || total > value.abs() * (-needed).exp2()
    {
        return Ok(None);
    }
    // Beyond the reach of the compiled body (an overflow cuts it off, and
    // no node lies past `f64::MAX`): the integrand's own far tail.
    let compiled = |t: f64| func(&[t]);
    let (low, high) = (a.min(b), a.max(b));
    let tail_seen = |upper: bool, cc: &mut Consts| {
        improper::f64_tail_trusted(arena, body, var, upper, &compiled, total, cc)
    };
    if (high == f64::INFINITY && !tail_seen(true, w.cc))
        || (low == f64::NEG_INFINITY && !tail_seen(false, w.cc))
    {
        debug!("evalf: the f64 quadrature cannot have seen the far tail");
        return Ok(None);
    }
    Ok(Some(CBall::new(
        (bigfloat::from_f64(value, prec), BigFloat::new(prec)),
        accuracy::Bound::real(total.log2() + 1e-12),
    )))
}

// ═══════════════════════════════════════════════════════════════════════════
// Sub-tree evaluation helpers (Sum, Product, Piecewise, RootOf, RootSum)
// ═══════════════════════════════════════════════════════════════════════════

/// Most terms of a finite `Sum` or `Product` evaluated numerically.
const MAX_RANGE_TERMS: u32 = 10_000;

/// The value and error bound of `expr` (with post-order `post_order`) at
/// the integer `k` for the bound variable `var`: the index is exact when
/// it fits the precision.
#[allow(clippy::too_many_arguments)]
fn evaluate_at_integer(
    arena: &Arena,
    expr: ExprId,
    post_order: &[ExprId],
    var: ExprId,
    k: &BigInt,
    w: &mut Wp<'_>,
) -> Result<CBall, SymplexError> {
    let prec = w.bits();
    let value = (
        crate::base::numeric::bigint_to_bigfloat(k, prec),
        BigFloat::new(prec),
    );
    let err = if k.bits() <= prec as u64 {
        accuracy::EXACT
    } else {
        accuracy::rounding(&value, prec)
    };
    evaluate_tree(
        arena,
        expr,
        post_order,
        Some((var, CBall::new(value, accuracy::Bound::real(err)))),
        true,
        w,
    )
}

/// A bound of a `Sum` or `Product` range as an integer: an integer literal
/// exactly, another value when it is real and within `10⁻⁹` of an integer.
fn range_bound(
    arena: &Arena,
    cache: Cache<'_>,
    id: ExprId,
    what: &str,
) -> Result<BigInt, SymplexError> {
    if let Some(q) = arena.as_num(id) {
        if q.is_integer() {
            return Ok(q.to_integer());
        }
        return Err(unevaluable(format!("{what} bounds are not integers")));
    }
    let v = get_cached(cache, id)?;
    if !v.1.is_zero() {
        return Err(unevaluable(format!("{what} bounds must be real")));
    }
    // An integer only when its value is one exactly, with an exact bound
    // (`floor(…)`): up to 0.53 a value within `10⁻⁹` of an integer was
    // taken for it, and `Σ_{k=1}^{3−e⁻¹⁰⁰} k^k` was 32.
    let exact = cache.bound(&id).is_some_and(|b| b.is_exact());
    if !exact || !(v.0.is_zero() || v.0.is_int()) {
        return Err(unevaluable(format!(
            "{what} bounds are not known to be integers"
        )));
    }
    let r = f64_of(&v.0)?;
    if r.abs() > 9.0e15 {
        return Err(unevaluable(format!("{what} bounds are too large")));
    }
    Ok(BigInt::from(r as i64))
}

/// The number of terms of `lo..=hi` (0 when empty), refused beyond
/// [`MAX_RANGE_TERMS`].
fn range_terms(lo: &BigInt, hi: &BigInt, what: &str) -> Result<u32, SymplexError> {
    if hi < lo {
        return Ok(0);
    }
    let n = hi - lo + 1;
    match u32::try_from(n) {
        Ok(n) if n <= MAX_RANGE_TERMS + 1 => Ok(n),
        _ => Err(unevaluable(format!(
            "{what} range too large for numerical evaluation (> {MAX_RANGE_TERMS} terms)"
        ))),
    }
}

/// A running sum of values with error bounds: per part, the terms' bounds
/// add, and every partial sum is rounded to the working precision
/// ([`accuracy::PartSum`]); a part that is an exact 0 in every term stays
/// exact (a sum of real terms is real).
struct SumAcc {
    value: Complex,
    re: accuracy::PartSum,
    im: accuracy::PartSum,
    unknown: bool,
    prec: usize,
}

impl SumAcc {
    fn new(prec: usize) -> SumAcc {
        SumAcc {
            value: c_zero(prec),
            re: accuracy::PartSum::new(),
            im: accuracy::PartSum::new(),
            unknown: false,
            prec,
        }
    }

    fn add(&mut self, term: &Complex, err: accuracy::Bound) {
        self.value = c_add(&self.value, term, self.prec, RM);
        self.unknown |= err.is_unknown();
        self.re.add(&term.0, err.re, &self.value.0, self.prec);
        self.im.add(&term.1, err.im, &self.value.1, self.prec);
    }

    /// The sum and its error bound.
    fn finish(self) -> CBall {
        if self.unknown {
            return CBall::new(self.value, accuracy::Bound::UNKNOWN);
        }
        let bound = accuracy::Bound {
            re: self.re.bound(),
            im: self.im.bound(),
        };
        CBall::new(self.value, bound)
    }
}

/// `Σ_{var = lo}^{hi} body`: a finite range of at most [`MAX_RANGE_TERMS`]
/// terms term by term, `hi = ∞` as a hypergeometric series ([`hypsum`]).
/// The bound is the terms' bounds plus the roundings (before 0.29 it was
/// the working precision whatever the terms lost, and an infinite sum was
/// `Unevaluable`).
#[allow(clippy::too_many_arguments)]
fn eval_sum(
    arena: &Arena,
    body: ExprId,
    var: ExprId,
    lo: ExprId,
    hi: ExprId,
    cache: Cache<'_>,
    w: &mut Wp<'_>,
) -> Result<CBall, SymplexError> {
    let prec = w.bits();
    if matches!(arena.node(lo), ExprNode::NegInfinity) {
        return Err(unevaluable("a Sum from −∞ is not evaluated numerically"));
    }
    let lo_i = range_bound(arena, cache, lo, "Sum")?;
    let post_order = walk::post_order_ids(arena, body);
    if matches!(arena.node(hi), ExprNode::Infinity) {
        if let Some(result) = emsum::rational_sum(arena, body, var, &lo_i, prec) {
            debug!("evalf: Sum — infinite sum of a rational term");
            return result;
        }
        if let Some(p) = emsum::power_summand(arena, body, var) {
            debug!("evalf: Sum — a power of a linear term: Hurwitz zeta");
            let mut consts = Vec::with_capacity(p.consts.len());
            for &c in &p.consts {
                let order = walk::post_order_ids(arena, c);
                consts.push(evaluate_at_integer(arena, c, &order, var, &lo_i, w)?);
            }
            return emsum::power_sum(&p, &consts, &lo_i, w);
        }
        debug!("evalf: Sum — infinite hypergeometric series");
        let mut orders: FxHashMap<ExprId, Vec<ExprId>> = FxHashMap::default();
        orders.insert(body, post_order);
        let mut part = |id: ExprId, k: &BigInt, p: usize| {
            let order = orders
                .entry(id)
                .or_insert_with(|| walk::post_order_ids(arena, id));
            evaluate_at_integer(arena, id, order, var, k, &mut w.at(p))
        };
        return hypsum::infinite_sum(arena, body, var, &lo_i, prec, &mut part);
    }
    let hi_i = range_bound(arena, cache, hi, "Sum")?;
    // Karr's convention: `Σ_{lo}^{hi} = −Σ_{hi+1}^{lo−1}` for `hi < lo − 1`
    // (it was the empty sum, as `eval` had it up to 0.54).
    let (lo_i, hi_i, negate) = if hi_i < &lo_i - 1 {
        (&hi_i + 1, &lo_i - 1, true)
    } else {
        (lo_i, hi_i, false)
    };
    let terms = range_terms(&lo_i, &hi_i, "Sum")?;
    let mut sum = SumAcc::new(prec);
    let mut k = lo_i.clone();
    for _ in 0..terms {
        let CBall { mid: term, err } = evaluate_at_integer(arena, body, &post_order, var, &k, w)?;
        let term = if negate {
            (term.0.neg(), term.1.neg())
        } else {
            term
        };
        sum.add(&term, err);
        k += 1;
    }
    debug!(lo = %lo_i, hi = %hi_i, "evalf: Sum evaluated");
    Ok(sum.finish())
}

/// `∏_{var = lo}^{hi} body` over a finite range of at most
/// [`MAX_RANGE_TERMS`] terms, bounded as a product of its terms; `hi = ∞`
/// for a rational factor tending to 1 fast enough (`emsum`), or `1 + a(k)`
/// with `a` a hypergeometric term of geometric decay times factors of
/// polynomial growth (`hypsum`).  Up to 0.43 every infinite product was
/// refused ("cannot evaluate infinity": `∏ (1 + 1/k²)`, `∏ (1 − 2^(−k))`).
#[allow(clippy::too_many_arguments)]
fn eval_product(
    arena: &Arena,
    body: ExprId,
    var: ExprId,
    lo: ExprId,
    hi: ExprId,
    cache: Cache<'_>,
    w: &mut Wp<'_>,
) -> Result<CBall, SymplexError> {
    let prec = w.bits();
    if matches!(arena.node(lo), ExprNode::NegInfinity) {
        return Err(unevaluable(
            "a Product from −∞ is not evaluated numerically",
        ));
    }
    let lo_i = range_bound(arena, cache, lo, "Product")?;
    if matches!(arena.node(hi), ExprNode::Infinity) {
        if let Some(result) = emsum::rational_product(arena, body, var, &lo_i, prec) {
            debug!("evalf: Product — infinite product of a rational factor");
            return result;
        }
        let mut orders: FxHashMap<ExprId, Vec<ExprId>> = FxHashMap::default();
        let mut part = |id: ExprId, k: &BigInt, p: usize| {
            let order = orders
                .entry(id)
                .or_insert_with(|| walk::post_order_ids(arena, id));
            evaluate_at_integer(arena, id, order, var, k, &mut w.at(p))
        };
        if let Some(result) = hypsum::dominated_product(arena, body, var, &lo_i, prec, &mut part) {
            return result;
        }
        return Err(unevaluable(format!(
            "infinite product of `{}`: the factor is neither rational nor 1 plus a \
             term of geometric decay, so no error bound is available",
            arena.display(body)
        )));
    }
    let hi_i = range_bound(arena, cache, hi, "Product")?;
    let terms = range_terms(&lo_i, &hi_i, "Product")?;
    let post_order = walk::post_order_ids(arena, body);
    let mut factors: Vec<CBall> = Vec::with_capacity(terms as usize);
    let mut k = lo_i.clone();
    for _ in 0..terms {
        factors.push(evaluate_at_integer(arena, body, &post_order, var, &k, w)?);
        k += 1;
    }
    let factors: Vec<&CBall> = factors.iter().collect();
    debug!(lo = %lo_i, hi = %hi_i, "evalf: Product evaluated");
    Ok(CBall::product(&factors, prec))
}

/// An operand of a relation without a value in the cache of `evalf`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Special {
    /// `+∞` (`1`) or `−∞` (`−1`).
    Infinity(i8),
    /// `zoo`.
    ComplexInfinity,
    /// `nan`.
    NaN,
}

/// A condition decided from the values of its operands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Decision {
    /// The truth value read off the computed values.
    value: bool,
    /// Does it hold for the true values too?  Every comparison involved
    /// was of exact values, or of a difference outside its error ball.
    certain: bool,
}

/// Decide a boolean condition numerically from the evaluated operands and
/// their error bounds.
///
/// A relation compares the difference of its operands (`Gt`, `Ge` need
/// both real); the decision is certain when both operands are exact (the
/// sign of a rounded difference of exact values is exact) or the same
/// expression, or when the difference is outside its error ball — as
/// `sign` is decided (see [`accuracy`]).  So an equality of distinct
/// inexact expressions is never certain, nor is an order between operands
/// whose imaginary parts are only within their error of 0 (they may not be
/// real).  `And`/`Or`/`Not` combine decisions; a certain `false`
/// operand decides an `And` (a certain `true` an `Or`) whatever the others.
/// An operand `±∞`, `zoo` or `nan` is compared by kind (`−∞ < x < +∞`
/// for a finite real `x`; `nan` equals nothing; orders with `zoo` or `nan`
/// are undecided, as SymPy raises).  `None` when an operand is missing from
/// the cache (free symbol, complex value in an order, unsupported node).
fn decide_condition(
    arena: &Arena,
    cond: ExprId,
    cache: Cache<'_>,
    prec: usize,
) -> Option<Decision> {
    let bound_of = |a: &ExprId| cache.err(a);
    // `a − b`, the exponent of its (joint) error bound, and whether it is
    // exact (both operands exact, or the same expression).
    let difference = |a: &ExprId, b: &ExprId| -> Option<(Complex, accuracy::ErrExp, bool)> {
        let (av, bv) = (cache.get(a)?, cache.get(b)?);
        if a == b {
            return Some((c_zero(prec), accuracy::EXACT, true));
        }
        let (ea, eb) = (bound_of(a), bound_of(b));
        let d = c_sub(av, bv, prec + 64, RM);
        let exact = ea.is_exact() && eb.is_exact();
        let e = if ea.is_unknown() || eb.is_unknown() {
            accuracy::UNKNOWN
        } else {
            // The operands' errors add; the difference rounds at `prec + 64`.
            accuracy::lsum(
                accuracy::lsum(ea.joint(), eb.joint()),
                accuracy::rounding(&d, prec + 64),
            )
        };
        Some((d, e, exact))
    };
    let away_from_zero = |d: &Complex, e: accuracy::ErrExp| {
        !accuracy::is_unknown(e) && !accuracy::contains_zero(d, e)
    };
    // `±∞`, `zoo` and `nan` have no value in the cache: their relations
    // with each other and with a finite (real) value are decided by kind.
    // Up to 0.41 every such relation was undecided: `Piecewise((1, exp(1000)
    // < ∞), (0, True))` was refused (SymPy 1.14: `1`).
    let special = |a: &ExprId| -> Option<Special> {
        match arena.node(*a) {
            ExprNode::Infinity => Some(Special::Infinity(1)),
            ExprNode::NegInfinity => Some(Special::Infinity(-1)),
            ExprNode::ComplexInfinity => Some(Special::ComplexInfinity),
            ExprNode::NaN => Some(Special::NaN),
            _ => None,
        }
    };
    // A finite value of the cache: `Some(true)` when it is real (exactly),
    // `Some(false)` when it may not be.
    let finite_real = |a: &ExprId| -> Option<bool> {
        let v = cache.get(a)?;
        let finite = !(v.0.is_inf() || v.0.is_nan() || v.1.is_inf() || v.1.is_nan());
        finite.then(|| v.1.is_zero() && accuracy::exactly_real(v, bound_of(a)))
    };
    let certainly = |value: bool| Decision {
        value,
        certain: true,
    };
    let order_special = |a: &ExprId, b: &ExprId, strict: bool| -> Option<Decision> {
        // `−∞ < finite real < +∞`; `zoo`, `nan` and complex values have no
        // order (SymPy raises).
        let tier = |x: &ExprId| -> Option<i8> {
            match special(x) {
                Some(Special::Infinity(s)) => Some(s),
                Some(_) => None,
                None => finite_real(x)?.then_some(0),
            }
        };
        let (ta, tb) = (tier(a)?, tier(b)?);
        Some(certainly(match ta.cmp(&tb) {
            std::cmp::Ordering::Greater => true,
            std::cmp::Ordering::Equal => !strict,
            std::cmp::Ordering::Less => false,
        }))
    };
    let equal_special = |a: &ExprId, b: &ExprId| -> Option<Decision> {
        // `nan` equals nothing (SymPy: `Eq(nan, nan)` is False); an
        // infinity only itself.
        let (sa, sb) = (special(a), special(b));
        if sa == Some(Special::NaN) || sb == Some(Special::NaN) {
            return Some(certainly(false));
        }
        match (sa, sb) {
            (Some(x), Some(y)) => Some(certainly(x == y)),
            (Some(_), None) => finite_real(b).map(|_| certainly(false)),
            (None, Some(_)) => finite_real(a).map(|_| certainly(false)),
            (None, None) => None,
        }
    };
    let order = |a: &ExprId, b: &ExprId, strict: bool| -> Option<Decision> {
        if special(a).is_some() || special(b).is_some() {
            return order_special(a, b, strict);
        }
        let (av, bv) = (cache.get(a)?, cache.get(b)?);
        if !(av.1.is_zero() && bv.1.is_zero()) {
            return None;
        }
        let real =
            accuracy::exactly_real(av, bound_of(a)) && accuracy::exactly_real(bv, bound_of(b));
        let (d, e, exact) = difference(a, b)?;
        // Strictly: astro-float's `is_positive` holds for `+0`, the
        // difference of an operand with itself (up to 0.41 `besselj(1, 2) >
        // besselj(1, 2)` was decided true, and certain).
        let value = bf_strictly_positive(&d.0) || (!strict && d.0.is_zero());
        Some(Decision {
            value,
            certain: exact || (real && away_from_zero(&d, e)),
        })
    };
    let equal = |a: &ExprId, b: &ExprId| -> Option<Decision> {
        if special(a).is_some() || special(b).is_some() {
            return equal_special(a, b);
        }
        let (d, e, exact) = difference(a, b)?;
        let value = d.0.is_zero() && d.1.is_zero();
        Some(Decision {
            value,
            certain: exact || (!value && away_from_zero(&d, e)),
        })
    };

    let post_order = walk::post_order_ids(arena, cond);
    let mut truth: FxHashMap<ExprId, Option<Decision>> = FxHashMap::default();
    for &id in &post_order {
        let kids = |ids: &[ExprId]| -> Vec<Option<Decision>> {
            ids.iter()
                .map(|k| truth.get(k).copied().flatten())
                .collect()
        };
        let v: Option<Decision> = match arena.node(id) {
            ExprNode::BoolTrue => Some(Decision {
                value: true,
                certain: true,
            }),
            ExprNode::BoolFalse => Some(Decision {
                value: false,
                certain: true,
            }),
            ExprNode::Gt(a, b) => order(a, b, true),
            ExprNode::Ge(a, b) => order(a, b, false),
            ExprNode::Eq_(a, b) => equal(a, b),
            ExprNode::Ne(a, b) => equal(a, b).map(|d| Decision {
                value: !d.value,
                ..d
            }),
            ExprNode::Not(inner) => truth.get(inner).copied().flatten().map(|d| Decision {
                value: !d.value,
                ..d
            }),
            ExprNode::And(ks) => combine(&kids(ks), false),
            ExprNode::Or(ks) => combine(&kids(ks), true),
            // Numeric operands and anything else carry no truth value.
            _ => None,
        };
        truth.insert(id, v);
    }
    truth.get(&cond).copied().flatten()
}

/// `Or` (`absorbing = true`) or `And` (`false`) of decisions.  The certain
/// operands combine by Kleene's rules ([`Truth::any`], [`Truth::all`]):
/// a certain absorbing operand decides, and so do certain operands only.
/// Otherwise every operand must be decided, and the result, read off the
/// values, is not certain.
///
/// [`Truth::any`]: crate::base::truth::Truth::any
/// [`Truth::all`]: crate::base::truth::Truth::all
fn combine(operands: &[Option<Decision>], absorbing: bool) -> Option<Decision> {
    use crate::base::truth::Truth;
    let certain = operands.iter().map(|d| match d {
        Some(d) if d.certain => Truth::from_bool(d.value),
        _ => Truth::Unknown,
    });
    let proved = if absorbing {
        Truth::any(certain)
    } else {
        Truth::all(certain)
    };
    if proved.is_known() {
        return Some(Decision {
            value: proved.is_true(),
            certain: true,
        });
    }
    let decided: Option<Vec<Decision>> = operands.iter().copied().collect();
    let decided = decided?;
    Some(Decision {
        value: if absorbing {
            decided.iter().any(|d| d.value)
        } else {
            decided.iter().all(|d| d.value)
        },
        certain: false,
    })
}

/// `Piecewise`: the value of the first branch whose condition is true.
///
/// Conditions are decided in order ([`decide_condition`]).  An undecidable
/// condition is an error — falling through to a later `True` branch would
/// be silently wrong.  A decision that is not certain (a difference within
/// its error of 0) still picks a branch, but the result then has no bound,
/// so the evaluation is repeated at a higher precision and refused when
/// the budget is spent, as `sign` of such a difference is.  Before 0.29 the
/// decision was read off the rounded values: `Piecewise((0, x < √2), (1,
/// True))` at `x = √2 + 10⁻¹⁰⁰` came out `0`.
fn eval_piecewise(
    arena: &Arena,
    pairs: &[(ExprId, ExprId)],
    cache: Cache<'_>,
    w: &mut Wp<'_>,
) -> Result<CBall, SymplexError> {
    let prec = w.bits();
    debug!(
        branches = pairs.len(),
        "evalf: Piecewise — evaluating conditions"
    );
    let mut certain = true;
    for &(value_id, cond_id) in pairs {
        let Some(decision) = decide_condition(arena, cond_id, cache, prec) else {
            return Err(unevaluable(format!(
                "cannot evaluate piecewise: condition `{}` is undecided",
                arena.display(cond_id)
            )));
        };
        certain &= decision.certain;
        if !decision.value {
            continue;
        }
        let CBall { mid: value, err } = match cache.get(&value_id) {
            Some(v) => CBall::new(v.clone(), cache.err(&value_id)),
            None => eval_ball(arena, value_id, cache, w)?,
        };
        return Ok(CBall {
            mid: value,
            err: if certain {
                err
            } else {
                accuracy::Bound::UNKNOWN
            },
        });
    }
    if !certain {
        // A condition found false only to this precision may be true: no
        // value is known yet.
        return Ok(CBall::new(c_zero(prec), accuracy::Bound::UNKNOWN));
    }
    Err(unevaluable(
        "cannot evaluate piecewise: every condition is false",
    ))
}

/// The error bound of a root ball: its certified radius on both parts of a
/// non-real root, on the real part of a root certified real (whose
/// imaginary part is exactly 0); no bound without a certified radius.
fn radius_error(ball: &crate::poly::roots::RootBall) -> accuracy::Bound {
    let e = match ball.radius.as_ref() {
        None => return accuracy::Bound::UNKNOWN,
        Some(r) if r.is_zero() => accuracy::EXACT,
        Some(r) => accuracy::part_lg(r),
    };
    if ball.real {
        accuracy::Bound::real(e)
    } else {
        accuracy::Bound::both(e)
    }
}

/// `RootOf(poly, var, index)`: the `index`-th entry of
/// [`rootof_roots`](crate::poly::roots::rootof_roots) of `poly` as a
/// polynomial in `var`, bounded by the radius of its certified inclusion
/// disk ([`root_balls`](crate::poly::roots::root_balls)).  A root certified
/// real has an exactly zero imaginary part; without an isolating disk there
/// is no bound.
fn eval_rootof(
    arena: &Arena,
    poly_id: ExprId,
    var_id: ExprId,
    idx_id: ExprId,
    prec: usize,
) -> Result<CBall, SymplexError> {
    let idx: usize = match arena.as_num(idx_id) {
        Some(r) if r.is_integer() => {
            let n: i64 = r
                .to_integer()
                .try_into()
                .map_err(|_| unevaluable("RootOf index out of range"))?;
            usize::try_from(n).map_err(|_| unevaluable("RootOf index must be non-negative"))?
        }
        Some(_) => return Err(unevaluable("RootOf index must be an integer")),
        None => return Err(unevaluable("RootOf index must be numeric")),
    };
    let poly = crate::poly::polybridge::expr_to_poly(arena, poly_id, var_id)
        .ok_or_else(|| unevaluable("could not convert RootOf expression to polynomial"))?;
    // The roots with multiplicity in the (re, im) order of `rootof_roots`,
    // the one definition of the order a `RootOf` index refers to
    // (`real_roots` derives its indices from the same call).
    let balls = crate::poly::roots::certified_roots(&poly, prec);
    let found = balls.len();
    let ball = balls.into_iter().nth(idx).ok_or_else(|| {
        unevaluable(format!(
            "RootOf index {idx} exceeds the {found} root(s) found"
        ))
    })?;
    let err = radius_error(&ball);
    Ok(CBall::new(ball.value, err))
}

/// `RootSum(poly, body, sumvar) = Σ_{α: poly(α) = 0} body(α)`: the body at
/// every root found by Aberth's method, each root bounded by its certified
/// inclusion disk.  A body with other free variables fails.
fn eval_rootsum(
    arena: &Arena,
    poly_id: ExprId,
    body_id: ExprId,
    sumvar_id: ExprId,
    w: &mut Wp<'_>,
) -> Result<CBall, SymplexError> {
    let prec = w.bits();
    debug!("evalf: RootSum — attempting numerical evaluation via Aberth roots");
    let poly =
        crate::poly::polybridge::expr_to_poly(arena, poly_id, sumvar_id).ok_or_else(|| {
            unevaluable(
                "RootSum: cannot convert polynomial expression to Poly (may contain free symbols)",
            )
        })?;
    let degree = poly.degree().unwrap_or(0);
    let balls = crate::poly::roots::certified_roots(&poly, prec);
    if balls.len() != degree {
        debug!(
            expected = degree,
            found = balls.len(),
            "evalf: RootSum — fewer roots than expected"
        );
        return Err(unevaluable(
            "RootSum: not every root of the polynomial was found",
        ));
    }
    let n_roots = balls.len();
    let post_order = walk::post_order_ids(arena, body_id);
    let mut sum = SumAcc::new(prec);
    for ball in balls {
        let err = radius_error(&ball);
        let seed = Some((sumvar_id, CBall::new(ball.value, err)));
        let CBall { mid: term, err: e } =
            evaluate_tree(arena, body_id, &post_order, seed, true, w)?;
        sum.add(&term, e);
    }
    debug!(n_roots, "evalf: RootSum — evaluated");
    Ok(sum.finish())
}

// ═══════════════════════════════════════════════════════════════════════════
// Complex arithmetic helpers
// ═══════════════════════════════════════════════════════════════════════════

// The field operations live in `base::bigcomplex` (shared with the root
// finders and algebraic-number verification); the transcendental ones
// below need this module's constant cache.
pub(crate) use crate::base::bigcomplex::{
    c_abs, c_add, c_div, c_from_real, c_i, c_mul, c_neg, c_one, c_powi, c_sub, c_zero,
};

fn c_exp(z: &Complex, w: &mut Wp<'_>) -> Complex {
    let prec = w.bits();
    // exp(a+bi) = exp(a)(cos(b) + i·sin(b))
    let exp_a = z.0.exp(prec, RM, w.cc);
    let (sin_b, cos_b) = sin_cos(&z.1, w);
    (exp_a.mul(&cos_b, prec, RM), exp_a.mul(&sin_b, prec, RM))
}

/// `(sin x, cos x)`, exact at `x ∈ {0, ±π/2, ±π}` when `x` is bit for bit
/// the working-precision constant.
///
/// The evaluator produces exactly those constants itself — `arg` of a
/// negative real is π, `asin` of a real beyond 1 has real part ±π/2 — and
/// `sin` of the *rounded* π is a residue of about `2^(−prec)`, not 0.
/// Carried on, that residue picks the side of the next branch cut by its
/// sign: `exp(ln(−5/7))` came out `−5/7 + 1e-50·i`, and
/// `acosh(1/exp(ln x))` at `x = −5/7` the conjugate of `acosh(−7/5)`.  A
/// value equal to the rounded π is indistinguishable from π at this
/// precision, so its exact sine is the right answer.
fn sin_cos(x: &BigFloat, w: &mut Wp<'_>) -> (BigFloat, BigFloat) {
    let prec = w.bits();
    let zero = BigFloat::new(prec);
    let one = BigFloat::from_i32(1, prec);
    if x.is_zero() {
        return (zero, one);
    }
    let pi = w.cc.pi(prec, RM).clone();
    let half_pi = pi.div(&BigFloat::from_i32(2, prec), prec, RM);
    if *x == half_pi {
        return (one, zero);
    }
    if *x == half_pi.neg() {
        return (one.neg(), zero);
    }
    if *x == pi || *x == pi.neg() {
        return (zero, one.neg());
    }
    (x.sin(prec, RM, w.cc), x.cos(prec, RM, w.cc))
}

fn c_ln(z: &Complex, w: &mut Wp<'_>) -> Complex {
    let prec = w.bits();
    // ln(z) = ln|z| + i·arg(z)
    let r = c_abs(z, prec, RM);
    let theta = atan2_bf(&z.1, &z.0, w);
    (r.ln(prec, RM, w.cc), theta)
}

fn c_sin(z: &Complex, w: &mut Wp<'_>) -> Complex {
    let prec = w.bits();
    // sin(a+bi) = sin(a)cosh(b) + i·cos(a)sinh(b)
    let (sin_a, cos_a) = sin_cos(&z.0, w);
    let cosh_b = z.1.cosh(prec, RM, w.cc);
    let sinh_b = z.1.sinh(prec, RM, w.cc);
    (sin_a.mul(&cosh_b, prec, RM), cos_a.mul(&sinh_b, prec, RM))
}

fn c_cos(z: &Complex, w: &mut Wp<'_>) -> Complex {
    let prec = w.bits();
    // cos(a+bi) = cos(a)cosh(b) - i·sin(a)sinh(b)
    let (sin_a, cos_a) = sin_cos(&z.0, w);
    let cosh_b = z.1.cosh(prec, RM, w.cc);
    let sinh_b = z.1.sinh(prec, RM, w.cc);
    (
        cos_a.mul(&cosh_b, prec, RM),
        sin_a.mul(&sinh_b, prec, RM).neg(),
    )
}

fn c_sinh(z: &Complex, w: &mut Wp<'_>) -> Complex {
    let prec = w.bits();
    // sinh(a+bi) = sinh(a)cos(b) + i·cosh(a)sin(b)
    let sinh_a = z.0.sinh(prec, RM, w.cc);
    let cosh_a = z.0.cosh(prec, RM, w.cc);
    let (sin_b, cos_b) = sin_cos(&z.1, w);
    (sinh_a.mul(&cos_b, prec, RM), cosh_a.mul(&sin_b, prec, RM))
}

fn c_cosh(z: &Complex, w: &mut Wp<'_>) -> Complex {
    let prec = w.bits();
    // cosh(a+bi) = cosh(a)cos(b) + i·sinh(a)sin(b)
    let cosh_a = z.0.cosh(prec, RM, w.cc);
    let sinh_a = z.0.sinh(prec, RM, w.cc);
    let (sin_b, cos_b) = sin_cos(&z.1, w);
    (cosh_a.mul(&cos_b, prec, RM), sinh_a.mul(&sin_b, prec, RM))
}

fn c_pow(base: &Complex, exp: &Complex, w: &mut Wp<'_>) -> Complex {
    let prec = w.bits();
    // base^exp = exp(exp * ln(base))
    // Special case: base is zero.
    if base.0.is_zero() && base.1.is_zero() {
        // 0^0 = 1 by convention; 0^(positive) = 0.
        if exp.0.is_zero() && exp.1.is_zero() {
            return c_one(prec);
        }
        return c_zero(prec);
    }
    let ln_base = c_ln(base, w);
    let product = c_mul(exp, &ln_base, prec, RM);
    c_exp(&product, w)
}

fn c_sqrt(z: &Complex, w: &mut Wp<'_>) -> Complex {
    let prec = w.bits();
    if z.0.is_zero() && z.1.is_zero() {
        return c_zero(prec);
    }
    let half = c_from_real(
        BigFloat::from_i32(1, prec).div(&BigFloat::from_i32(2, prec), prec, RM),
        prec,
    );
    c_pow(z, &half, w)
}

// ── Inverse trig (complex) ─────────────────────────────────────────

/// `asin`, `atan`, `asinh` and `atanh` of a complex `z` are `z·(1 + O(z²))`,
/// but their logarithmic formulas take `ln(1 + O(z))`: rounding `1 ± iz`
/// to `prec` bits is an absolute error of `2^−prec`, `log₂(1/|z|)` bits of
/// the value that the node's bound (its rounding at the value's magnitude)
/// does not know of.  `f(z, prec)` is evaluated with that many more bits and
/// rounded; below `2^−prec` the value is `z` itself.  Before 0.30 neither:
/// `atan(i·10⁻³⁰)` was `9.99999998762973·10⁻³¹·i` at 16 digits (truly
/// `1·10⁻³⁰·i`).
fn small_argument(z: &Complex, w: &mut Wp<'_>, f: fn(&Complex, &mut Wp<'_>) -> Complex) -> Complex {
    let prec = w.bits();
    let m = accuracy::mag(z).unwrap_or(0);
    if m >= 0 {
        return f(z, w);
    }
    let lost = usize::try_from(-m).unwrap_or(usize::MAX);
    if lost > prec + 8 {
        return (round_to(z.0.clone(), prec), round_to(z.1.clone(), prec));
    }
    let v = f(z, &mut w.at(prec + lost + 16));
    (round_to(v.0, prec), round_to(v.1, prec))
}

fn c_asin(z: &Complex, w: &mut Wp<'_>) -> Complex {
    small_argument(z, w, c_asin_at)
}

fn c_asin_at(z: &Complex, w: &mut Wp<'_>) -> Complex {
    let prec = w.bits();
    // asin(z) = −i·ln(iz + √(1 − z²)).  In half the plane the two terms
    // nearly cancel for large |z|; there the sum is taken as 1/(√(1 − z²) −
    // iz), the same number ((iz + s)(s − iz) = 1), without the cancellation.
    let i_unit = c_i(prec);
    let one = c_one(prec);
    let z_sq = c_mul(z, z, prec, RM);
    let one_minus_z_sq = c_sub(&one, &z_sq, prec, RM);
    let sqrt_term = c_sqrt(&one_minus_z_sq, w);
    let iz = c_mul(&i_unit, z, prec, RM);
    let sum = sum_without_cancellation(&iz, &sqrt_term, prec);
    let ln_sum = c_ln(&sum, w);
    let neg_i = c_neg(&i_unit);
    c_mul(&neg_i, &ln_sum, prec, RM)
}

/// `a + s` for `a`, `s` with `s² − a² = 1` (then `(a + s)(s − a) = 1`):
/// the direct sum when it does not cancel, `1/(s − a)` when it does.
fn sum_without_cancellation(a: &Complex, s: &Complex, prec: usize) -> Complex {
    let plus = c_add(a, s, prec, RM);
    let minus = c_sub(s, a, prec, RM);
    if c_abs(&plus, prec, RM) >= c_abs(&minus, prec, RM) {
        plus
    } else {
        c_div(&c_one(prec), &minus, prec, RM)
    }
}

fn c_acos(z: &Complex, w: &mut Wp<'_>) -> Complex {
    let prec = w.bits();
    // acos(z) = pi/2 - asin(z)
    let half_pi_val =
        w.cc.pi(prec, RM)
            .div(&BigFloat::from_i32(2, prec), prec, RM);
    let half_pi = c_from_real(half_pi_val, prec);
    let asin_z = c_asin(z, w);
    c_sub(&half_pi, &asin_z, prec, RM)
}

fn c_atan(z: &Complex, w: &mut Wp<'_>) -> Complex {
    small_argument(z, w, c_atan_at)
}

fn c_atan_at(z: &Complex, w: &mut Wp<'_>) -> Complex {
    let prec = w.bits();
    // atan(z) = (i/2)·(ln(1 − iz) − ln(1 + iz)).  The quotient form
    // (i/2)·ln((i + z)/(i − z)) differs from it by π on the cut
    // z ∈ i·(1, ∞) (atan(2i) would be −π/2 + …, mpmath says +π/2 + …).
    let one = c_one(prec);
    let iz = c_mul(&c_i(prec), z, prec, RM);
    let ln_minus = c_ln(&c_sub(&one, &iz, prec, RM), w);
    let ln_plus = c_ln(&c_add(&one, &iz, prec, RM), w);
    let diff = c_sub(&ln_minus, &ln_plus, prec, RM);
    let half = BigFloat::from_i32(1, prec).div(&BigFloat::from_i32(2, prec), prec, RM);
    // (i/2)·(a + bi) = −b/2 + (a/2)·i
    (
        diff.1.mul(&half, prec, RM).neg(),
        diff.0.mul(&half, prec, RM),
    )
}

// ── Inverse hyperbolic (complex) ──────────────────────────────────

fn c_asinh(z: &Complex, w: &mut Wp<'_>) -> Complex {
    small_argument(z, w, c_asinh_at)
}

fn c_asinh_at(z: &Complex, w: &mut Wp<'_>) -> Complex {
    let prec = w.bits();
    // asinh(z) = ln(z + √(z² + 1)), the sum taken without cancellation
    // (it cancels for Re z < 0 and large |z|).
    let one = c_one(prec);
    let z_sq = c_mul(z, z, prec, RM);
    let z_sq_plus_one = c_add(&z_sq, &one, prec, RM);
    let sqrt_term = c_sqrt(&z_sq_plus_one, w);
    let sum = sum_without_cancellation(z, &sqrt_term, prec);
    c_ln(&sum, w)
}

fn c_acosh(z: &Complex, w: &mut Wp<'_>) -> Complex {
    let prec = w.bits();
    // acosh(z) = ln(z + √(z + 1)·√(z − 1)).  The single root √(z² − 1)
    // is the other branch whenever Re z < 0 (acosh(−2 + i/2) came out
    // negated); the product of two roots is the principal value everywhere.
    let one = c_one(prec);
    let root_plus = c_sqrt(&c_add(z, &one, prec, RM), w);
    let root_minus = c_sqrt(&c_sub(z, &one, prec, RM), w);
    let sum = c_add(z, &c_mul(&root_plus, &root_minus, prec, RM), prec, RM);
    c_ln(&sum, w)
}

fn c_atanh(z: &Complex, w: &mut Wp<'_>) -> Complex {
    small_argument(z, w, c_atanh_at)
}

fn c_atanh_at(z: &Complex, w: &mut Wp<'_>) -> Complex {
    let prec = w.bits();
    // atanh(z) = (ln(1 + z) − ln(1 − z))/2.  The quotient form
    // ln((1 + z)/(1 − z))/2 differs from it by πi on the cut z ∈ (1, ∞).
    let one = c_one(prec);
    let ln_plus = c_ln(&c_add(&one, z, prec, RM), w);
    let ln_minus = c_ln(&c_sub(&one, z, prec, RM), w);
    let diff = c_sub(&ln_plus, &ln_minus, prec, RM);
    let half = BigFloat::from_i32(1, prec).div(&BigFloat::from_i32(2, prec), prec, RM);
    (diff.0.mul(&half, prec, RM), diff.1.mul(&half, prec, RM))
}

/// The real part of a trigonometric node's argument, and the bits it
/// carries beyond `prec`.  A rational child (`sin(10^100)`) is converted
/// again with `log₂|r|` extra bits, so that reducing it modulo 2π starts
/// from a value correct to `prec` bits after the point, not one rounded to
/// `prec` significant bits.
fn trig_arg(arena: &Arena, inner: ExprId, val: &Complex, prec: usize) -> (BigFloat, usize) {
    if let ExprNode::Num(nid) = arena.node(inner)
        && let Some(e) = val.0.exponent()
        && e > 0
        && val.0.inexact()
    {
        let extra = usize::try_from(e).unwrap_or(0);
        if extra <= arena.config.max_evalf_precision as usize {
            return (ratio_to_bigfloat(arena.num(*nid), prec + extra), extra);
        }
    }
    (val.0.clone(), 0)
}

/// π/2 at `prec` bits.
fn half_pi_bf(w: &mut Wp<'_>) -> BigFloat {
    let prec = w.bits();
    w.cc.pi(prec, RM)
        .div(&BigFloat::from_i32(2, prec), prec, RM)
}

/// `|x| ≤ 1`: the real domain of `asin`, `acos` and (open) `atanh`.
fn abs_le_one(x: &BigFloat, prec: usize) -> bool {
    x.abs() <= BigFloat::from_i32(1, prec)
}

/// Binary exponent above which an *inexact* trigonometric argument is
/// refused.  Reducing `x` modulo 2π turns its relative rounding error into
/// an absolute error of about `2^(e − prec)`, and [`evalf`] carries at least
/// 64 guard bits beyond the digits requested, so past `2^64` the reduction
/// eats into the digits that are printed.
const TRIG_ARG_MAX_INEXACT_EXPONENT: i64 = 64;

/// Is the argument `c` of a trigonometric node one that [`check_trig_arg`]
/// refuses at every precision although its error bound is tracked: real,
/// inexact, beyond `2⁶⁴` (and within the configured maximum precision),
/// and not a rational literal (which [`trig_arg`] converts again with more
/// bits)?  `eval_ball` evaluates it from its value and bound.
fn large_inexact_trig_argument(arena: &Arena, c: ExprId, cache: Cache<'_>) -> bool {
    if matches!(arena.node(c), ExprNode::Num(_)) {
        return false;
    }
    let Some(v) = cache.get(&c) else {
        return false;
    };
    if !v.1.is_zero() || !v.0.inexact() || v.0.is_nan() || v.0.is_inf() {
        return false;
    }
    v.0.exponent().is_some_and(|e| {
        let e = i64::from(e);
        e > TRIG_ARG_MAX_INEXACT_EXPONENT && e <= i64::from(arena.config.max_evalf_precision)
    })
}

/// Refuse a trigonometric argument whose value is not known well enough to
/// reduce it modulo 2π at this precision: an inexact one of magnitude
/// ≥ `2^64` (see [`TRIG_ARG_MAX_INEXACT_EXPONENT`]; `exp(100) ≈ 2^144` at
/// 30 digits is known only to within about `2^(144 − 166)`), or any beyond
/// `EvalConfig::max_evalf_precision` bits — astro-float would compute π to
/// that many bits, and `cos(cosh(cosh(4i)))` ran for minutes.
///
/// `extra` is the number of bits `x` carries beyond `prec` (see
/// [`trig_arg`]); they widen the inexact bound by as much.
fn check_trig_arg(
    x: &BigFloat,
    extra: usize,
    prec: usize,
    arena: &Arena,
) -> Result<(), SymplexError> {
    let Some(e) = x.exponent() else {
        return Ok(());
    };
    let e = i64::from(e);
    let cap = i64::from(arena.config.max_evalf_precision);
    let bound =
        TRIG_ARG_MAX_INEXACT_EXPONENT.saturating_add(i64::try_from(extra).unwrap_or(i64::MAX));
    if e > cap || (x.inexact() && e > bound) {
        let digits = u32::try_from(prec.saturating_sub(64) * 10 / 34).unwrap_or(u32::MAX);
        return Err(SymplexError::PrecisionExhausted {
            requested: digits,
            achieved: 0,
        });
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Real-valued helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Compute atan2(y, x) using BigFloat arithmetic.
fn atan2_bf(y: &BigFloat, x: &BigFloat, w: &mut Wp<'_>) -> BigFloat {
    let prec = w.bits();
    if x.is_zero() && y.is_zero() {
        return BigFloat::new(prec);
    }
    // On the real axis the sign of a zero `y` is an accident of how the
    // value was computed (`c_neg` turns `+0` into `−0`), not a side of the
    // cut: a negative real has argument +π on the principal branch.
    if y.is_zero() {
        return if x.is_negative() {
            w.cc.pi(prec, RM).clone()
        } else {
            BigFloat::new(prec)
        };
    }

    let pi_val = w.cc.pi(prec, RM);
    let two = BigFloat::from_i32(2, prec);
    let half_pi = pi_val.div(&two, prec, RM);

    if x.is_zero() {
        return if is_sign_positive(y) {
            half_pi
        } else {
            half_pi.neg()
        };
    }

    let ratio = y.div(x, prec, RM);
    let atan_val = ratio.atan(prec, RM, w.cc);

    if is_strictly_positive(x) {
        atan_val
    } else {
        let pi_val2 = w.cc.pi(prec, RM);
        if y.is_negative() {
            atan_val.sub(&pi_val2, prec, RM)
        } else {
            atan_val.add(&pi_val2, prec, RM)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Special function helpers (f64-based)
// ═══════════════════════════════════════════════════════════════════════════

/// `bf` as an `f64` ([`crate::base::bigfloat::to_f64_or_inf`]: the nearest
/// `f64`, ties to even, `±∞` beyond the range; zero is `+0.0`).
///
/// # Errors
///
/// [`SymplexError::Unevaluable`] for NaN.
fn f64_of(bf: &BigFloat) -> Result<f64, SymplexError> {
    crate::base::bigfloat::to_f64_or_inf(bf).ok_or_else(|| SymplexError::Unevaluable {
        reason: "NaN in BigFloat to f64 conversion".into(),
    })
}

/// `erf x` by its Taylor series in `f64`, for the Newton start of
/// [`arb_erfinv`] (`abs(x) < 1/2`; the start only guides the certified
/// Halley iteration).  (An unused Lanczos `Γ` in `f64` and the asymptotic
/// branch of this function, which no caller reached, were removed at 0.54.)
fn erf_f64(x: f64) -> f64 {
    // erf(x) = 2/sqrt(pi) * sum_{n=0}^{inf} (-1)^n * x^(2n+1) / (n! * (2n+1))
    let mut sum = 0.0;
    let mut term = x; // first term: x
    sum += term;
    for n in 1..50 {
        term *= -x * x / n as f64;
        sum += term / (2 * n + 1) as f64;
    }
    sum * 2.0 / std::f64::consts::PI.sqrt()
}

/// `loggamma(x)` for real `x` (not a pole): the analytic continuation of
/// `ln Γ` from `x > 0` (SymPy's and mpmath's `loggamma`), real for `x > 0`
/// and on the cut `x < 0` its limit from above, `ln|Γ(x)| − iπ⌈−x⌉`
/// (mpmath: `loggamma(-2.5) = −0.0562 − 3πi`).  Before 0.30 the value left
/// of 0 was the real `ln|Γ(x)|` (the compiled back-ends' `lgamma`, which is
/// now `ln(abs(gamma(x)))`).
///
/// * `x > 0`: Stirling's series ([`stirling_log_gamma`]).
/// * `x < 0`: real part `ln π − ln|sin πd| − ln Γ(1 − x)`, `d = x − round(x)`
///   exact; imaginary part `−π·⌈−x⌉ = π·⌊x⌋`.
///
/// Near a zero of `ln|Γ|` (`x = 1`, `2`, and two between consecutive poles
/// left of 0) the terms summed are much larger than the value: the loss is
/// measured against a bound on them and the evaluation repeated with that
/// many more bits, up to [`cancellation_cap`].  Before 0.29 neither:
/// `loggamma(1 + 2⁻¹⁰⁰)` was wrong from its 13th digit, `πx` was rounded
/// before `sin`, and arguments within `10⁻¹²` of a pole (in `f64`) were
/// refused as poles.
fn arb_log_gamma(x: &BigFloat, w: &mut Wp<'_>) -> Result<Complex, SymplexError> {
    let prec = w.bits();
    if x.is_nan() || x.is_inf() {
        return Err(unevaluable("LogGamma of special float value"));
    }
    if (x.is_int() || x.is_zero()) && !bf_strictly_positive(x) {
        return Err(unevaluable("LogGamma at non-positive integer pole"));
    }
    let exact = x.mantissa_max_bit_len().unwrap_or(prec).max(prec) + 64;
    let cap = cancellation_cap(prec);
    // On the cut: −π·⌈−x⌉ = π·⌊x⌋, the limit from the upper half-plane.
    let imaginary = if bf_strictly_positive(x) {
        BigFloat::new(prec)
    } else {
        let pi = w.cc.pi(prec + 16, RM).clone();
        round_to(pi.mul(&x.floor(), prec + 16, RM), prec)
    };
    // The first attempt covers the terms' bits (a value of magnitude 2⁻⁸
    // and up needs no second one).
    let zs0 = f64_of(&x.abs())?.min(1e300) + 0.5 * (prec + 48) as f64 + 3.0;
    let mut extra = (24 + terms_bits(2.0 * zs0 * (zs0.ln() + 1.0)).max(0) as usize).min(cap);
    loop {
        let wp = prec + extra;
        // The value and a bound on the terms summed.
        let (value, largest) = if bf_strictly_positive(x) {
            // Stirling at z + s, s ‹ 0.5·wp: terms up to (z+s)·(|ln(z+s)| + 1),
            // and the shift Σ ln(z + i) below that.
            let v = stirling_log_gamma(x, &mut w.at(wp))?;
            let zs = f64_of(x)?.max(0.0) + 0.5 * wp as f64 + 2.0;
            (v, 2.0 * zs * (zs.ln() + 1.0))
        } else {
            let n = x.add(&BigFloat::from_f64(0.5, 64), exact, RM).floor();
            let d = x.sub(&n, exact, RM);
            let pi = w.cc.pi(wp, RM).clone();
            let ln_sin = pi.mul(&d, wp, RM).sin(wp, RM, w.cc).abs().ln(wp, RM, w.cc);
            let one_minus_x = BigFloat::from_i32(1, 64).sub(x, exact, RM);
            let lg1 = stirling_log_gamma(&one_minus_x, &mut w.at(wp))?;
            let ln_pi = pi.ln(wp, RM, w.cc);
            let v = ln_pi.sub(&ln_sin, wp, RM).sub(&lg1, wp, RM);
            let zs = f64_of(&one_minus_x)? + 0.5 * wp as f64 + 2.0;
            let terms = 2.0 * zs * (zs.ln() + 1.0) + f64_of(&ln_sin)?.abs() + 2.0;
            (v, terms)
        };
        // The loss relative to the whole value: on the cut its imaginary
        // part, at least π, bounds it below (a zero of ln|Γ| there costs no
        // extra bits).
        let magnitude = match (value.exponent(), imaginary.exponent()) {
            _ if value.is_zero() && imaginary.is_zero() => None,
            (Some(e), Some(i)) if !value.is_zero() && !imaginary.is_zero() => Some(e.max(i)),
            (Some(e), _) if !value.is_zero() => Some(e),
            (_, i) => i,
        };
        let lost = match magnitude {
            Some(e) => (terms_bits(largest) - i64::from(e)).max(0) as usize,
            None => wp,
        };
        if lost + 16 <= extra {
            return Ok((round_to(value, prec), imaginary));
        }
        if extra >= cap {
            return Err(special_exhausted(prec));
        }
        extra = (lost + 32).max(2 * extra).min(cap);
    }
}

/// A precision at which a sum or difference of `x` and a small integer is
/// exact (or within `2^−64` of it relative to its larger operand's bits).
fn exact_bits(x: &BigFloat, prec: usize) -> usize {
    // The lowest bit of `x` is at `e − m`; `x + c` for `|c| < 2^62` reaches
    // up to `max(e, 62)`.  Before 0.30 only `m + 64` bits: a tiny `x` lost
    // its bits in `x − k + 1`, and `binomial(10⁻⁷⁰, 6)` — `Γ` next to its
    // pole at `−5` — was wrong from its 8th digit.
    // (At most 2¹⁶ bits for the offset: beyond, `x + c` rounds to `c`, and
    // a pole of `Γ` there is refused rather than met with a million-bit
    // number.)
    let m = x.mantissa_max_bit_len().unwrap_or(prec).max(prec);
    let below = x.exponent().map_or(0, |e| {
        usize::try_from(64 - i64::from(e)).unwrap_or(0).min(1 << 16)
    });
    m + below + 64
}

/// `⌈log₂ v⌉ + 1` for a positive `f64` magnitude bound (`i64::MAX/4` beyond
/// range).
fn terms_bits(v: f64) -> i64 {
    if v.is_finite() && v > 0.0 {
        v.log2().ceil() as i64 + 1
    } else {
        i64::MAX / 4
    }
}

/// `ψ(x)` for real `x` (not a pole) to a relative error of about `2^−prec`.
///
/// 1. `x < 0`: the reflection `ψ(x) = ψ(1 − x) − π cot(πd)`, `d = x − round(x)`
///    exact (`cot` has period 1), so a pole is only an integer `x` and `πd`
///    is rounded relative to `d`, not to `x`.
/// 2. `x > 0`: `ψ(x) = ψ(x + N) − Σ_{k<N} 1/(x + k)` with `x + N ≥ wp/3`.
/// 3. `ψ(y) ~ ln y − 1/(2y) − Σ_{k≥1} B_{2k}/(2k·y^{2k})` (DLMF 5.11.2), with
///    exact Bernoulli numbers until a term is below the working precision
///    (the smallest term, near `k ≈ πy`, is `~e^{−2πy}`).
///
/// Near a zero of `ψ` (`x₀ = 1.4616…`, and one between consecutive poles
/// left of 0) the terms are much larger than the value: the loss, `log₂` of
/// the largest term over the result, is measured and the evaluation
/// repeated with that many more bits, up to [`cancellation_cap`].  Before
/// 0.29 the series stopped after 12 terms and the loss went unmeasured:
/// `digamma(1/3)` was wrong from its 46th digit, and `digamma` at a
/// 33-digit rational next to `x₀` from its 5th; `πx` was rounded before
/// `cot`, and poles were taken within `2^{−wp/2}`.
fn arb_digamma(x: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    if x.is_nan() || x.is_inf_pos() || x.is_inf_neg() {
        return Err(SymplexError::Unevaluable {
            reason: "Digamma of special float value".into(),
        });
    }
    let cap = cancellation_cap(prec);
    let mut extra = 20;
    loop {
        let wp = prec + extra;
        let (value, largest) = digamma_wp(x, &mut w.at(wp))?;
        let lost = match (largest.exponent(), value.exponent()) {
            _ if value.is_zero() => wp,
            (Some(l), Some(v)) => usize::try_from(i64::from(l) - i64::from(v)).unwrap_or(0),
            _ => 0,
        };
        if lost + 16 <= extra {
            return Ok(round_to(value, prec));
        }
        if extra >= cap {
            return Err(special_exhausted(prec));
        }
        extra = (lost + 32).max(2 * extra).min(cap);
    }
}

/// `ψ(x)` at working precision `wp` (see [`arb_digamma`]), with the
/// magnitude of the largest term summed: the absolute error is about
/// `2^−wp` of it.
fn digamma_wp(x: &BigFloat, w: &mut Wp<'_>) -> Result<(BigFloat, BigFloat), SymplexError> {
    let wp = w.bits();
    let pole = || SymplexError::Unevaluable {
        reason: "Digamma at non-positive integer pole".into(),
    };
    let exact = x.mantissa_max_bit_len().unwrap_or(wp).max(wp) + 64;
    let one = BigFloat::from_i32(1, wp);
    if x.is_negative() {
        // x = n + d, n = round(x), |d| ≤ ½, both exact.
        let n = x.add(&BigFloat::from_f64(0.5, 64), exact, RM).floor();
        let d = x.sub(&n, exact, RM);
        if d.is_zero() {
            return Err(pole());
        }
        let pi = w.cc.pi(wp, RM).clone();
        let pi_d = pi.mul(&d, wp, RM);
        let pi_cot = pi.mul(
            &pi_d.cos(wp, RM, w.cc).div(&pi_d.sin(wp, RM, w.cc), wp, RM),
            wp,
            RM,
        );
        let one_minus_x = BigFloat::from_i32(1, 64).sub(x, exact, RM);
        let (psi, largest) = digamma_wp(&one_minus_x, w)?;
        let value = psi.sub(&pi_cot, wp, RM);
        let largest = if pi_cot.abs() > largest {
            pi_cot.abs()
        } else {
            largest
        };
        return Ok((value, largest));
    }
    if x.is_zero() {
        return Err(pole());
    }
    // ψ(x) = ψ(y) − Σ_{k<N} 1/(x + k), y = x + N ≥ wp/3.
    let threshold = BigFloat::from_i32(i32::try_from(wp / 3).unwrap_or(i32::MAX).max(10), wp);
    let mut shift = BigFloat::new(wp);
    let mut y = x.clone();
    while bf_lt(&y, &threshold) {
        shift = shift.add(&one.div(&y, wp, RM), wp, RM);
        y = y.add(&one, exact, RM);
    }
    let ln_y = y.ln(wp, RM, w.cc);
    let largest = if shift > ln_y.abs() {
        shift.clone()
    } else {
        ln_y.abs()
    };
    let two = BigFloat::from_i32(2, wp);
    let mut result = ln_y
        .sub(&one.div(&y.mul(&two, wp, RM), wp, RM), wp, RM)
        .sub(&shift, wp, RM);
    let y2 = y.mul(&y, wp, RM);
    let mut y_pow = y2.clone();
    let floor = largest.exponent().map_or(i64::MIN / 2, i64::from) - wp as i64 - 2;
    let mut prev: Option<i32> = None;
    for k in 1..=wp {
        let b2k = bernoulli::even(k);
        let denom = y_pow.mul(&BigFloat::from_i128(2 * k as i128, wp), wp, RM);
        let term = ratio_to_bigfloat(&b2k, wp).div(&denom, wp, RM);
        match term.exponent() {
            Some(te) if i64::from(te) < floor => break,
            Some(te) if prev.is_some_and(|p| te > p + 2) => break, // diverging
            Some(te) => prev = Some(te),
            None => {}
        }
        result = result.sub(&term, wp, RM);
        y_pow = y_pow.mul(&y2, wp, RM);
    }
    Ok((result, largest))
}

// ═══════════════════════════════════════════════════════════════════════════
// Arbitrary-precision Gamma via Stirling series
// ═══════════════════════════════════════════════════════════════════════════

/// Compute log Γ(z) for real z > 0 using the Stirling asymptotic series.
///
/// Algorithm:
/// 1. Argument reduction — shift z by integer r until z + r ≥ 0.5 · p.
/// 2. Stirling series:
///    log Γ(z) = (z − ½) ln(z) − z + ½ ln(2π) + Σ B₂ₖ / [2k(2k−1) z^{2k−1}].
/// 3. Undo shift:
///    log Γ(z) = log Γ(z + r) − ln Π_{i=0}^{r−1} (z + i).
///
/// The series' smallest term is about `e^{−2π(z+r)}`; at `z + r ≥ 0.5·p`
/// it is far below `2^{−p}` after about `0.2·p` terms (the Bernoulli
/// numbers `B₂ₖ`, exact rationals, dominate the cost: at `0.2·p` they took
/// `0.35·p` terms, and `Γ(1/3)` at 1,200 digits 20 s before 0.30).
fn stirling_log_gamma(z: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    tracing::debug!(prec, "stirling_log_gamma: entry");

    let guard = 20; // extra guard bits
    let wp = prec + guard;

    // Determine how far to shift z for convergence.
    let z_approx = f64_of(z)?;
    let threshold = ((wp as f64) * 0.5).ceil() as i64;
    let shift = if z_approx < threshold as f64 {
        (threshold - z_approx.floor() as i64).max(0) as usize
    } else {
        0
    };

    tracing::trace!(
        z_approx,
        threshold,
        shift,
        "stirling_log_gamma: argument reduction"
    );

    // z_shifted = z + shift
    let shift_bf = BigFloat::from_i128(shift as i128, wp);
    let z_shifted = z.add(&shift_bf, wp, RM);

    // Main Stirling formula:
    //   log Γ(z) = (z − 1/2)·ln(z) − z + (1/2)·ln(2π) + Σ ...
    let half = BigFloat::from_f64(0.5, wp);
    let z_minus_half = z_shifted.sub(&half, wp, RM);
    let log_z = z_shifted.ln(wp, RM, w.cc);

    let mut result = z_minus_half.mul(&log_z, wp, RM);
    result = result.sub(&z_shifted, wp, RM);

    // (1/2) · ln(2π) at working precision.
    let two = BigFloat::from_i32(2, wp);
    let pi_val = w.cc.pi(wp, RM).clone();
    let two_pi = two.mul(&pi_val, wp, RM);
    let half_log_2pi = two_pi.ln(wp, RM, w.cc).mul(&half, wp, RM);
    result = result.add(&half_log_2pi, wp, RM);

    // Series: Σ_{k=1}^{N} B_{2k} / [2k(2k−1) · z^{2k−1}]
    let z_sq = z_shifted.mul(&z_shifted, wp, RM);
    let mut z_pow = z_shifted.clone(); // z^1 → z^3 → z^5 → …

    // Upper bound on useful terms: safely below the divergence point πz.
    let max_terms = ((wp as f64) * 0.35) as usize + 10;
    let mut prev_term_exp: Option<i32> = None;

    for k in 1..=max_terms {
        let b2k = bernoulli::even(k);
        if b2k.is_zero() {
            continue;
        }

        // Advance z_pow: z^1, z^3, z^5, …
        if k > 1 {
            z_pow = z_pow.mul(&z_sq, wp, RM);
        }

        // term = B_{2k} / (2k · (2k−1) · z^{2k−1})
        let b2k_bf = ratio_to_bigfloat(&b2k, wp);
        let denom_int = (2 * k * (2 * k - 1)) as i128;
        let denom = BigFloat::from_i128(denom_int, wp);
        let denom_full = denom.mul(&z_pow, wp, RM);
        let term = b2k_bf.div(&denom_full, wp, RM);

        // Divergence / convergence detection via binary exponents.
        if let Some(t_exp) = term.exponent() {
            if let Some(prev) = prev_term_exp
                && t_exp > prev + 10
            {
                tracing::trace!(k, "stirling series: diverging, stopping");
                break;
            }
            prev_term_exp = Some(t_exp);

            // Term is negligible relative to accumulated result.
            if let Some(r_exp) = result.exponent()
                && (r_exp as i64 - t_exp as i64) > wp as i64
            {
                tracing::trace!(k, "stirling series: converged");
                break;
            }
        }

        result = result.add(&term, wp, RM);
    }

    // Undo argument reduction:
    //   log Γ(z) = log Γ(z+shift) − ln Π_{i=0}^{shift−1} (z + i),
    // one logarithm of the product instead of one per factor (850
    // logarithms at 4,000 bits took a second).  The product carries the
    // roundings of its factors, `shift·2^−wp` relative, as the sum of
    // logarithms did.
    if shift > 0 {
        let mut product = BigFloat::from_i32(1, wp);
        for i in 0..shift {
            let z_plus_i = z.add(&BigFloat::from_i128(i as i128, wp), wp, RM);
            product = product.mul(&z_plus_i, wp, RM);
        }
        result = result.sub(&product.ln(wp, RM, w.cc), wp, RM);
    }

    Ok(result)
}

/// Compute Γ(x) at arbitrary precision for real x.
///
/// For positive x, evaluates exp(stirling_log_gamma(x)).
/// For negative non-integer x, applies the reflection formula
/// Γ(z) = π / (sin(πd) · (−1)ⁿ · Γ(1 − z)) with `z = n + d`, `n = round(z)`
/// and `d` exact, so that `sin` is accurate next to a pole (before 0.29 `πz`
/// was rounded first, losing `log₂(1/d)` bits, and arguments within
/// `10⁻¹²` of a pole — in `f64` — or `2^{−prec/2}` were refused as poles).
fn arb_gamma_real(x: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    // The poles: the non-positive integers, exactly.
    if x.is_zero() || (x.is_negative() && x.is_int()) {
        return Err(SymplexError::Unevaluable {
            reason: "Gamma at non-positive integer pole".into(),
        });
    }

    if !x.is_negative() {
        // Direct Stirling path.
        tracing::debug!(prec, "arb_gamma_real: positive argument");
        let log_gamma = stirling_log_gamma(x, w)?;
        Ok(log_gamma.exp(prec, RM, w.cc))
    } else {
        tracing::debug!(prec, "arb_gamma_real: reflection formula");
        let exact = x.mantissa_max_bit_len().unwrap_or(prec).max(prec) + 64;
        let one_minus_x = BigFloat::from_i32(1, 64).sub(x, exact, RM);
        let log_gamma_1mx = stirling_log_gamma(&one_minus_x, w)?;
        let gamma_1mx = log_gamma_1mx.exp(prec, RM, w.cc);

        // sin(πz) = (−1)ⁿ sin(πd).
        let n = x.add(&BigFloat::from_f64(0.5, 64), exact, RM).floor();
        let d = x.sub(&n, exact, RM);
        let pi_val = w.cc.pi(prec, RM).clone();
        let mut sin_pi_x = pi_val.mul(&d, prec, RM).sin(prec, RM, w.cc);
        let odd = !n
            .div(&BigFloat::from_i32(2, 64), exact, RM)
            .fract()
            .is_zero();
        if odd {
            sin_pi_x = sin_pi_x.neg();
        }
        if sin_pi_x.is_zero() {
            return Err(SymplexError::Unevaluable {
                reason: "Gamma at non-positive integer pole".into(),
            });
        }

        let denom = sin_pi_x.mul(&gamma_1mx, prec, RM);
        Ok(pi_val.div(&denom, prec, RM))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Arbitrary-precision erf via Taylor series / asymptotic expansion
// ═══════════════════════════════════════════════════════════════════════════

/// `erfc(x)` for `x > 0` from the asymptotic expansion
///
/// ```text
/// erfc(x) = exp(−x²)/(x√π) · Σ_{n≥0} (−1)^n (2n−1)!! / (2x²)^n
/// ```
///
/// truncated at its smallest term (near `n ≈ x²`), so the *relative* error
/// is `≈ e^{−x²}`: callers must ensure `x² ≳ wp·ln 2` when they need
/// `erfc` itself to `wp` bits, or `2x² ≳ wp·ln 2` when they only need
/// `erf = 1 − erfc` to `wp` bits (the absolute error is then `≈ e^{−2x²}`).
fn erfc_asymptotic(ax: &BigFloat, w: &mut Wp<'_>) -> BigFloat {
    let wp = w.bits();
    let x_sq = ax.mul(ax, wp, RM);
    let (sum, x_sqrt_pi) = erfc_asymptotic_series(ax, w);
    let exp_neg_x_sq = x_sq.neg().exp(wp, RM, w.cc);
    exp_neg_x_sq.mul(&sum, wp, RM).div(&x_sqrt_pi, wp, RM)
}

/// The sum `Σ_{n≥0} (−1)^n (2n−1)!!/(2x²)^n` of [`erfc_asymptotic`] at `wp`
/// bits, truncated at its smallest term, and `x·√π`: `erfc(x)` is
/// `e^(−x²)·sum/(x√π)` (`extended` takes `e^(−x²)` at its own scale, below
/// the exponent range).
fn erfc_asymptotic_series(ax: &BigFloat, w: &mut Wp<'_>) -> (BigFloat, BigFloat) {
    let wp = w.bits();
    let x_sq = ax.mul(ax, wp, RM);
    let two_x_sq = x_sq.mul(&BigFloat::from_i32(2, wp), wp, RM);

    let mut sum = BigFloat::from_i32(1, wp);
    let mut term = BigFloat::from_i32(1, wp);
    let mut prev_abs: Option<BigFloat> = None;
    // The terms shrink until `n ≈ x²`; past that they grow and the loop
    // stops, so `wp + 40` is only reached when the terms are already tiny.
    let max_terms = wp + 40;
    for n in 1..=max_terms {
        // term *= −(2n−1) / (2x²)
        let factor = BigFloat::from_i128(2 * n as i128 - 1, wp);
        term = term.mul(&factor, wp, RM).div(&two_x_sq, wp, RM).neg();
        let t_abs = term.abs();
        if let Some(ref p) = prev_abs
            && bf_gt(&t_abs, p)
        {
            break; // past the smallest term: optimal truncation
        }
        sum = sum.add(&term, wp, RM);
        if negligible(&term, &sum, wp) {
            break;
        }
        prev_abs = Some(t_abs);
    }

    let sqrt_pi = w.cc.pi(wp, RM).clone().sqrt(wp, RM);
    let x_sqrt_pi = ax.mul(&sqrt_pi, wp, RM);
    (sum, x_sqrt_pi)
}

/// Compute erf(x) at arbitrary precision for real x.  The result carries
/// `prec + 32` bits.
///
/// For small-to-moderate |x|, uses the Taylor series:
///   erf(x) = (2/√π) Σ_{n=0}^{N} (-1)^n x^{2n+1} / (n! (2n+1))
/// whose terms peak near `n ≈ x²` at `≈ e^{x²}` before the alternating sum
/// cancels down to `|erf| < 1`, so the working precision carries
/// `x² log₂ e` extra bits.
///
/// For large |x| (`2x² ≳ wp·ln 2`), uses [`erfc_asymptotic`]:
///   erf(x) = sign(x) · (1 − erfc(|x|))
fn arb_erf(x: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    if x.is_zero() {
        return Ok(BigFloat::new(prec));
    }

    let guard = 32;
    let wp = prec + guard;

    let x_approx = f64_of(x)?;
    // Threshold: for |x| beyond this, the asymptotic expansion (absolute
    // error ≈ e^{−2x²} for erf) is accurate to wp bits. Roughly √(wp·ln 2/2).
    let threshold = ((wp as f64) * 0.35).sqrt() + 2.0;

    if x_approx.abs() < threshold {
        // ── Taylor series ──────────────────────────────────────────────────────────
        // erf(x) = (2/√π) · Σ_{n=0}^{N} prod_n / (2n+1)
        // where prod_0 = x, prod_{n+1} = prod_n · (-x²) / (n+1)
        let x_sq_f = x_approx * x_approx;
        let wpt = Prec::new(wp)
            .guarded(
                Guard::for_log2(x_sq_f * std::f64::consts::LOG2_E),
                Prec::MAX,
            )
            .ok_or_else(|| special_exhausted(prec))?
            .plus(8)
            .bits();
        let mut xw = x.clone();
        let _ = xw.set_precision(wpt, RM);
        let neg_x_sq = xw.mul(&xw, wpt, RM).neg();
        let mut prod = xw.clone(); // (-x²)^n · x / n!
        let mut sum = xw.clone(); // accumulator (first term = x)

        // |term_n| < 2^{−wpt} needs n ≈ 5x² at the threshold, fewer below it.
        let max_terms = (6.0 * x_sq_f) as usize + wpt + 60;
        let mut converged = false;
        for n in 1..=max_terms {
            // prod *= -x² / n
            prod = prod.mul(&neg_x_sq, wpt, RM);
            prod = prod.div(&BigFloat::from_i128(n as i128, wpt), wpt, RM);

            // term = prod / (2n+1)
            let divisor = BigFloat::from_i128(2 * n as i128 + 1, wpt);
            let term = prod.div(&divisor, wpt, RM);

            if negligible(&term, &sum, wpt) {
                converged = true;
                break;
            }
            sum = sum.add(&term, wpt, RM);
        }
        if !converged {
            return Err(series_did_not_converge("erf", max_terms));
        }

        // Multiply by 2/√π
        let two = BigFloat::from_i32(2, wpt);
        let pi_val = w.cc.pi(wpt, RM).clone();
        let sqrt_pi = pi_val.sqrt(wpt, RM);
        let two_over_sqrt_pi = two.div(&sqrt_pi, wpt, RM);

        Ok(round_to(sum.mul(&two_over_sqrt_pi, wpt, RM), wp))
    } else {
        // ── Asymptotic expansion for large |x| ────────────────────────────────────
        let mut ax = x.abs();
        let _ = ax.set_precision(wp, RM);
        let erfc_val = erfc_asymptotic(&ax, &mut w.at(wp));
        let one = BigFloat::from_i32(1, wp);
        let erf_val = one.sub(&erfc_val, wp, RM);

        if x.is_negative() {
            Ok(erf_val.neg())
        } else {
            Ok(erf_val)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════
// Euler-Mascheroni constant at arbitrary precision (Brent-McMillan B1)
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the Euler-Mascheroni constant γ at `prec` bits of precision
/// using the Brent-McMillan B1 algorithm.
///
/// This is the fastest known algorithm for computing γ.  It uses the identity
/// involving modified Bessel functions, simplified to a fixed-point summation
/// with convergence rate O(e^{-4n}) where n = 2^p.
///
/// Reference: Brent & McMillan (1980), mpmath `euler_fixed`.
fn arb_euler_gamma(w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    // We need ln(2) at working precision.
    let extra = 30;
    let wp = prec + extra;

    // Choose p such that e^{-4·2^p} < 2^{-wp}, i.e. 4·2^p > wp·ln(2),
    // i.e. p > log2(wp·ln(2)/4).
    let p = ((wp as f64 / 4.0) * std::f64::consts::LN_2).log2().ceil() as u64 + 1;
    let n: i128 = 1i128 << p;
    let n_sq = n * n;

    // ln(2) at working precision.
    let two_bf = BigFloat::from_i32(2, wp);
    let ln2 = two_bf.ln(wp, RM, w.cc);

    // A = U = -p · ln(2)   (= -ln(n) since n = 2^p)
    let p_bf = BigFloat::from_i128(p as i128, wp);
    let neg_p_ln2 = p_bf.mul(&ln2, wp, RM).neg();

    let mut a = neg_p_ln2.clone();
    let mut u = neg_p_ln2;
    // B = V = 1
    let one = BigFloat::from_i32(1, wp);
    let mut b = one.clone();
    let mut v = one.clone();

    let n_sq_bf = BigFloat::from_i128(n_sq, wp);

    let mut k: i128 = 1;
    loop {
        let k_bf = BigFloat::from_i128(k, wp);
        let k_sq_bf = k_bf.mul(&k_bf, wp, RM);

        // B = B · n² / k²
        b = b.mul(&n_sq_bf, wp, RM).div(&k_sq_bf, wp, RM);

        // A = (A · n² / k + B) / k
        a = a.mul(&n_sq_bf, wp, RM).div(&k_bf, wp, RM);
        a = a.add(&b, wp, RM).div(&k_bf, wp, RM);

        u = u.add(&a, wp, RM);
        v = v.add(&b, wp, RM);

        // Convergence: both A and B are negligibly small.
        let a_small = a.exponent().is_none_or(|e| e < -(wp as i32) + 10);
        let b_small = b.exponent().is_none_or(|e| e < -(wp as i32) + 10);
        if a_small && b_small {
            tracing::trace!(k, "arb_euler_gamma: converged");
            break;
        }

        k += 1;
        if k > 10 * (wp as i128) {
            return Err(SymplexError::ComputationFailed {
                operation: "euler_gamma",
                reason: "Brent-McMillan did not converge".into(),
            });
        }
    }

    // γ = U / V
    Ok(u.div(&v, prec, RM))
}

// ═══════════════════════════════════════════════════════════════════════════
// Hankel P/Q series for Bessel asymptotic expansion
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the Hankel auxiliary functions P_ν(z) and Q_ν(z) for the
/// Bessel asymptotic expansion (DLMF §10.17).
///
/// P_ν(z) = Σ_{k=0}^{N} (-1)^k · a_{2k}(ν) / z^{2k}
/// Q_ν(z) = Σ_{k=0}^{N} (-1)^k · a_{2k+1}(ν) / z^{2k+1}
///
/// where a_k(ν) = (4ν²−1²)(4ν²−3²)⋯(4ν²−(2k−1)²) / (k! · 8^k)
/// (DLMF 10.17.1), so a_k = a_{k−1} · (4ν² − (2k−1)²) / (8k).
///
/// Uses optimal truncation: stops when terms start increasing (divergent
/// series).  The error is bounded by the first omitted term for real
/// positive z (DLMF §10.17(iii)).  The returned flag is `true` when the
/// truncation happened because the terms fell below `2^{-prec}` relative
/// to the sums — i.e. the expansion actually delivered `prec` bits.  When
/// it is `false` the series started diverging before reaching the target
/// and the caller must fall back to the power series.
fn hankel_pq(
    order: &BigFloat,
    z: &BigFloat,
    prec: usize,
    _cc: &mut Consts,
) -> (BigFloat, BigFloat, bool) {
    let one = BigFloat::from_i32(1, prec);
    let four = BigFloat::from_i32(4, prec);
    let eight = BigFloat::from_i32(8, prec);

    let four_nu_sq = four.mul(&order.mul(order, prec, RM), prec, RM);

    let z_inv = one.div(z, prec, RM);

    // P and Q accumulators; the k = 0 term (a_0 = 1) is added by the loop.
    let mut p_sum = BigFloat::new(prec);
    let mut q_sum = BigFloat::new(prec);

    let mut a_k = one.clone(); // a_0 = 1
    let mut z_power = one.clone(); // z^0 = 1

    // The terms reach their minimum near k ≈ 2|z|; allow a little slack.
    let max_terms = (3 * prec + 100).min(100_000);

    let mut prev_a_abs: Option<BigFloat> = None;
    let mut converged = false;

    for k in 0..max_terms {
        if k > 0 {
            // a_k = a_{k-1} · (4ν² - (2k-1)²) / (8k)
            let two_km1 = BigFloat::from_i32((2 * k as i32) - 1, prec);
            let two_km1_sq = two_km1.mul(&two_km1, prec, RM);
            let numer = four_nu_sq.sub(&two_km1_sq, prec, RM);
            let k_bf = BigFloat::from_i32(k as i32, prec);
            let denom = eight.mul(&k_bf, prec, RM);
            a_k = a_k.mul(&numer, prec, RM).div(&denom, prec, RM);
            // A half-integer order ends the expansion: `4ν² = (2k−1)²`
            // exactly, and every later coefficient has the factor 0 — the
            // sums are complete (spherical Bessel functions).  Before 0.30
            // the zero terms never met the convergence test, the expansion
            // was reported as failing, and the power series took over
            // with `1.44·x` guard bits: `besselj(5/2, 10²⁵)` printed
            // `3.8·10¹⁰⁵²⁷`, `besselj(−3/2, 12!)` hung.
            if a_k.is_zero() {
                converged = true;
                break;
            }

            // Update z_power: multiply by 1/z each step.
            z_power = z_power.mul(&z_inv, prec, RM);
        }

        // term = a_k / z^k
        let term = a_k.mul(&z_power, prec, RM);

        // Check for divergence: if |term| > |prev_term|, stop.
        let term_abs = term.abs();
        if let Some(ref prev) = prev_a_abs
            && term_abs.partial_cmp(prev) == Some(std::cmp::Ordering::Greater)
            && k > 2
        {
            tracing::trace!(k, "hankel_pq: optimal truncation (terms diverging)");
            break;
        }

        // Convergence: term negligible relative to accumulated sums.
        let p_exp = p_sum.exponent().unwrap_or(0);
        let q_exp = q_sum.exponent().unwrap_or(0);
        let ref_exp = p_exp.max(q_exp);
        let t_exp = term.exponent().unwrap_or(i32::MIN / 2);
        if k > 0 && (ref_exp as i64 - t_exp as i64) > prec as i64 {
            tracing::trace!(k, "hankel_pq: converged (term negligible)");
            converged = true;
            break;
        }

        prev_a_abs = Some(term_abs);

        // Even k → contributes to P with sign (-1)^(k/2)
        // Odd k → contributes to Q with sign (-1)^((k-1)/2)
        if k % 2 == 0 {
            // P term: (-1)^(k/2) · a_k / z^k
            if (k / 2) % 2 == 0 {
                p_sum = p_sum.add(&term, prec, RM);
            } else {
                p_sum = p_sum.sub(&term, prec, RM);
            }
        } else {
            // Q term: (-1)^((k-1)/2) · a_k / z^k
            if ((k - 1) / 2) % 2 == 0 {
                q_sum = q_sum.add(&term, prec, RM);
            } else {
                q_sum = q_sum.sub(&term, prec, RM);
            }
        }
    }

    (p_sum, q_sum, converged)
}

/// Smallest `|x|` at which the Hankel expansion is expected to reach `wp`
/// bits for order `ν`.
///
/// The smallest term of the (divergent) expansion is `~e^{−2|x|}` times a
/// factor that grows like `e^{2|ν|}` (the first `|ν|` coefficients are
/// inflated by `4ν²/(2j−1)²`), so `2|x| ≳ (wp + 3|ν|)·ln 2`.  This is
/// only a heuristic for choosing the algorithm; [`hankel_pq`] reports
/// whether it actually converged.
fn hankel_threshold(wp: usize, order_abs: f64) -> f64 {
    (wp as f64 + 3.0 * order_abs) * std::f64::consts::LN_2 / 2.0 + 4.0
}

/// Evaluate `J_ν(x)` (`want_j == true`) or `Y_ν(x)` via the Hankel
/// asymptotic expansion at `x > 0`,
///
/// ```text
/// J_ν(x) = √(2/(πx)) · [cos(ω)·P_ν(x) − sin(ω)·Q_ν(x)]
/// Y_ν(x) = √(2/(πx)) · [sin(ω)·P_ν(x) + cos(ω)·Q_ν(x)],   ω = x − νπ/2 − π/4
/// ```
///
/// Returns `None` when the expansion does not reach `wp` bits at this
/// `x` (the caller then uses the power series).
fn bessel_jy_hankel(
    order: &BigFloat,
    x: &BigFloat,
    want_j: bool,
    w: &mut Wp<'_>,
) -> Option<BigFloat> {
    let wp = w.bits();
    // The phase ω = x − νπ/2 − π/4 loses the leading `log2 x` bits of x to
    // cancellation, so the trigonometric part needs that many extra bits
    // (the caller refuses beyond `cancellation_cap`).
    let wp = wp + x.exponent().unwrap_or(0).max(0) as usize;
    let mut xw = x.clone();
    let _ = xw.set_precision(wp, RM);
    let mut nu = order.clone();
    let _ = nu.set_precision(wp, RM);

    let (p_val, q_val, converged) = hankel_pq(&nu, &xw, wp, w.cc);
    if !converged {
        tracing::trace!("bessel_jy_hankel: expansion did not reach target precision");
        return None;
    }
    let pi = w.cc.pi(wp, RM).clone();
    let two = BigFloat::from_i32(2, wp);
    let four = BigFloat::from_i32(4, wp);

    let two_over_pi_x = two.div(&pi.mul(&xw, wp, RM), wp, RM);
    let amplitude = two_over_pi_x.sqrt(wp, RM);

    let nu_pi_half = nu.mul(&pi, wp, RM).div(&two, wp, RM);
    let pi_quarter = pi.div(&four, wp, RM);
    let phase = xw.sub(&nu_pi_half, wp, RM).sub(&pi_quarter, wp, RM);

    let cos_phase = phase.cos(wp, RM, w.cc);
    let sin_phase = phase.sin(wp, RM, w.cc);

    let combo = if want_j {
        cos_phase
            .mul(&p_val, wp, RM)
            .sub(&sin_phase.mul(&q_val, wp, RM), wp, RM)
    } else {
        sin_phase
            .mul(&p_val, wp, RM)
            .add(&cos_phase.mul(&q_val, wp, RM), wp, RM)
    };
    Some(amplitude.mul(&combo, wp, RM))
}

// ═══════════════════════════════════════════════════════════════════════════
// Bessel function evaluation (arbitrary precision)
// ═══════════════════════════════════════════════════════════════════════════

/// Arbitrary-precision Bessel function of the first kind J_ν(x).
///
/// * `|x|` below [`hankel_threshold`] (or whenever the asymptotic expansion
///   cannot reach the target precision): ascending power series
///   `J_ν(x) = Σ_k (-1)^k (x/2)^(ν+2k) / (k! Γ(ν+k+1))`.  Its terms grow
///   to `~e^{|x|}` before the alternating sum cancels down to `O(1)`, so
///   the working precision is raised by [`cancellation_guard_bits`] — this
///   makes the series correct at *any* `x`, merely slower for huge `x`.
/// * Otherwise the Hankel expansion with the full `P/Q` series
///   ([`bessel_jy_hankel`]), whose optimal-truncation error is `~e^{−2|x|}`.
fn arb_bessel_j(order: &BigFloat, x: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let classic = arb_bessel_j_classic(order, x, w);
    debye_fallback(bessel_debye::Kind::J, order, x, w, classic)
}

/// Where the series and the Hankel expansion of the Bessel functions give
/// up (`classic` is an error), the uniform expansions of large order
/// (`bessel_debye`), at `abs(ν)` for `K_ν` and for the integer orders of
/// `I_ν` (`I_{−n} = I_n`) and at `abs(x)` for an integer order of `J_ν` and
/// `I_ν` (`f_n(−x) = (−1)ⁿ f_n(x)`); the error stands where they do not
/// apply.  Before 0.35 `bessely(20000, 300)`, `besselk(20000, 300)`,
/// `bessely(20000, 30000)`, `besseli(1000, 120000)` and `besselj(20000,
/// 19000)` were refused.
fn debye_fallback(
    kind: bessel_debye::Kind,
    order: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
    classic: Result<BigFloat, SymplexError>,
) -> Result<BigFloat, SymplexError> {
    use bessel_debye::Kind;
    let Err(err) = classic else {
        return classic;
    };
    let integer = order.is_int();
    let negative_x = x.is_negative();
    // `J_{−ν}`, `Y_{−ν}` of a non-integer order from `J_ν` and `Y_ν`.
    if order.is_negative()
        && !integer
        && matches!(kind, Kind::J | Kind::Y)
        && !negative_x
        && let Some(v) = bessel_reflected(kind, &order.abs(), x, w)
    {
        return v;
    }
    let nu = match kind {
        Kind::K => order.abs(),
        Kind::I if integer => order.abs(),
        _ if order.is_negative() => return Err(err),
        _ => order.clone(),
    };
    if negative_x && !(integer && matches!(kind, Kind::J | Kind::I)) {
        return Err(err);
    }
    match bessel_jy_large(kind, &nu, &x.abs(), w) {
        Some(Ok(v)) if negative_x && bessel_order::is_odd(order) => Ok(v.neg()),
        Some(r) => r,
        None => Err(err),
    }
}

/// A Bessel function of large positive order `nu` at `x > 0` by the uniform
/// expansions (`bessel_debye`), and for `J`, `Y` where they stop short — the
/// turning point `x ≈ ν`, `Y_ν(x)` for `x < ν` — by the recurrence in the
/// order from where they do not (`bessel_recur`).  Before 0.37
/// `besselj(20000, 20000)`, `bessely(20000, 20100)` and `bessely(20000,
/// 14800)` were refused.
fn bessel_jy_large(
    kind: bessel_debye::Kind,
    nu: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Option<Result<BigFloat, SymplexError>> {
    use bessel_debye::Kind;
    match bessel_debye::debye(kind, nu, x, w) {
        Some(r) => Some(r),
        None if matches!(kind, Kind::J | Kind::Y) => bessel_recur::recur(kind, nu, x, w),
        None => None,
    }
}

/// `J_{−ν}(x) = cos(νπ)·J_ν(x) − sin(νπ)·Y_ν(x)` and
/// `Y_{−ν}(x) = sin(νπ)·J_ν(x) + cos(νπ)·Y_ν(x)` (DLMF 10.4.7, 10.4.8)
/// for a non-integer `ν > 0` and `x > 0`, `J_ν` and `Y_ν` by
/// [`bessel_jy_large`] (`None` where they are not available).  `νπ` is reduced exactly: `ν = n + f` with an integer `n`, the
/// signs `(−1)ⁿ`.  The cancellation of the two terms is measured and made up
/// with more bits.  Before 0.37 `besselj(−(1000 + 1/3), 2000)` was refused.
fn bessel_reflected(
    kind: bessel_debye::Kind,
    nu: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Option<Result<BigFloat, SymplexError>> {
    let prec = w.bits();
    use bessel_debye::Kind;
    let mut wp = prec + 32;
    let cap = 4 * prec + 1024;
    loop {
        let j = match bessel_jy_large(Kind::J, nu, x, &mut w.at(wp))? {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
        };
        let y = match bessel_jy_large(Kind::Y, nu, x, &mut w.at(wp))? {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
        };
        // ν = n + f, 0 < f < 1: cos(νπ) = (−1)ⁿ cos(fπ), sin(νπ) = (−1)ⁿ sin(fπ).
        let n = nu.floor();
        let f = nu.sub(&n, exact_bits(nu, wp), RM);
        let pi = w.cc.pi(wp + 16, RM).clone();
        let angle = f.mul(&pi, wp + 16, RM);
        let odd = bessel_order::is_odd(&n);
        let sign = |v: BigFloat| if odd { v.neg() } else { v };
        let (c, s) = (sign(angle.cos(wp, RM, w.cc)), sign(angle.sin(wp, RM, w.cc)));
        let (a, b) = match kind {
            Kind::J => (c.mul(&j, wp, RM), s.mul(&y, wp, RM).neg()),
            _ => (s.mul(&j, wp, RM), c.mul(&y, wp, RM)),
        };
        let v = a.add(&b, wp, RM);
        // The error is at most `2^(2 − wp)·(abs(J) + abs(Y))` (`J`, `Y` to
        // `2^(−wp)` relative, the sine and cosine to `2^(1 − wp)` absolute, the
        // products) and `2^(−wp)·abs(v)`: the bits lost against the larger
        // of `J`, `Y`.
        let big = j
            .exponent()
            .unwrap_or(i32::MIN / 2)
            .max(y.exponent().unwrap_or(i32::MIN / 2));
        let lost = if v.is_zero() {
            wp as i64
        } else {
            i64::from(big) - i64::from(v.exponent().unwrap_or(i32::MIN / 2)) + 1
        };
        if lost + 6 + (prec as i64) <= wp as i64 {
            return Some(Ok(round_to(v, prec)));
        }
        if wp >= cap {
            return None;
        }
        wp = (wp + lost.max(32) as usize).min(cap);
    }
}

/// [`arb_bessel_j`] by the ascending series and the Hankel expansion.
fn arb_bessel_j_classic(
    order: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    let guard = 32;
    let wp = prec + guard;

    // Special case: x = 0 (`J_0(0) = 1` for the order exactly 0: up to 0.53
    // an order below `10⁻¹⁵`, `besselj(e⁻¹⁰⁰, 0)`, gave 1, truly 0).
    if x.is_zero() {
        let order_f64 = f64_of(order)?;
        return if order.is_zero() {
            // J_0(0) = 1
            Ok(BigFloat::from_i32(1, prec))
        } else if order_f64 > 0.0 {
            // J_n(0) = 0 for n > 0
            Ok(BigFloat::new(prec))
        } else {
            Err(SymplexError::Unevaluable {
                reason: "BesselJ at x=0 with negative order".into(),
            })
        };
    }

    refuse_bessel_argument(x, prec)?;
    // `(x/2)^ν` of a negative `x` is complex for a non-integer order.
    // Before 0.30 the power series took `|x/2|^ν` and returned a real
    // number: `besselj(−1/2, −1)` was `0.431…`, truly `−0.431…·i`.
    if x.is_negative() && !order.is_int() {
        return Err(SymplexError::Unevaluable {
            reason: "BesselJ of negative argument with non-integer order is complex".into(),
        });
    }
    let x_f64 = f64_of(x)?;
    let order_f64 = f64_of(order)?;

    // Check for integer order (most common case) — exactly: before 0.29 an
    // order within 10⁻¹² of an integer was taken for it, and
    // `besselj(2 + 10⁻¹⁴, 1)` was wrong from its 15th digit.
    let order_int = order_f64.round() as i64;
    let is_int_order = order.is_int();
    let large = order_f64.abs() >= bessel_order::LARGE_ORDER;

    // Negative integer order: J_{-n}(x) = (-1)^n J_n(x) — the order negated
    // and its parity taken exactly (before 0.34 through an `i64`, which
    // saturates beyond 9.2·10¹⁸: `besselj(−10²⁰, x)` recursed without end).
    if is_int_order && order.is_negative() {
        let j = arb_bessel_j(&order.neg(), x, w)?;
        return Ok(if bessel_order::is_odd(order) {
            j.neg()
        } else {
            j
        });
    }

    // Large |x|: Hankel asymptotic expansion (J is even/odd in x for
    // integer order; non-integer order with x < 0 is complex and rejected).
    if x_f64.abs() >= hankel_threshold(wp, order_f64.abs()) {
        if x.is_negative() {
            if !is_int_order {
                return Err(SymplexError::Unevaluable {
                    reason: "BesselJ of negative argument with non-integer order is complex".into(),
                });
            }
            let j = arb_bessel_j(order, &x.abs(), w)?;
            return Ok(if bessel_order::is_odd(order) {
                j.neg()
            } else {
                j
            });
        }
        if let Some(j) = bessel_jy_hankel(order, x, true, &mut w.at(wp)) {
            return Ok(round_to(j, prec));
        }
    }

    // ── Ascending series with cancellation guard bits ──────────────────
    // J_ν(x) = (x/2)^ν · Σ_{k=0}^N (-1)^k · (x/2)^{2k} / (k! · Γ(ν+k+1))
    {
        let wp = wp + series_guard_bits(x_f64.abs(), prec)?;
        let mut xw = x.clone();
        let _ = xw.set_precision(wp, RM);
        let mut order_w = order.clone();
        let _ = order_w.set_precision(wp, RM);
        let order = &order_w;
        let two = BigFloat::from_i32(2, wp);
        let x_half = xw.div(&two, wp, RM);
        let x_half_sq = x_half.mul(&x_half, wp, RM);
        let neg_x_half_sq = x_half_sq.neg();

        // Series: sum = Σ (-1)^k · (x/2)^{2k} / (k! · Γ(ν+k+1))
        // We track term = (-x²/4)^k / (k! · Γ(ν+k+1)) incrementally.
        // term_{k+1} = term_k · (-x²/4) / ((k+1) · (ν+k+1))
        let mut sum = BigFloat::new(wp); // will add 1/Γ(ν+1) as first term

        // A large order: the factor (x/2)^ν/Γ(ν+1) at once, its exponent
        // to an absolute 2^(−wp) (`bessel_order`), and the series from 1.
        // Before 0.34 an integer order took a multiplication per unit:
        // `besselj(8345185991999992, 61/10²⁷)` never finished.
        let (prefix, mut term) = if large {
            let bounds = bessel_order::series_bounds(order_f64, x_f64.abs(), true);
            // Below the range (only for `0 < abs(x) < ν`): 0, which the
            // accuracy layer holds as the underflow of a nonzero `J_ν`
            // (`J_ν` has no zero there, `accuracy::underflowed`) and
            // `extended` scales.  Before 0.34 `besselj(10⁸ + 1/2, 1)`
            // printed `0`; in 0.34 it was refused where it arose, and so was
            // `1 + besselj(10⁹, 1)`.
            let Some(p) = bessel_order::power_over_gamma(
                order,
                &x_half.abs(),
                bounds,
                "besselj",
                &mut w.at(wp),
            )?
            else {
                return Ok(BigFloat::new(prec));
            };
            // J_n(−x) = (−1)ⁿ J_n(x) (a negative x has an integer order here).
            let p = if x.is_negative() && bessel_order::is_odd(order) {
                p.neg()
            } else {
                p
            };
            (p, BigFloat::from_i32(1, wp))
        } else {
            // Compute (x/2)^ν.  For integer order, use repeated multiplication.
            let prefix = if is_int_order && order_int >= 0 {
                let mut p = BigFloat::from_i32(1, wp);
                for _ in 0..order_int {
                    p = p.mul(&x_half, wp, RM);
                }
                p
            } else {
                // General: (x/2)^ν = exp(ν · ln(x/2))
                let ln_xh = x_half.abs().ln(wp, RM, w.cc);
                let nu_ln = order.mul(&ln_xh, wp, RM);
                nu_ln.exp(wp, RM, w.cc)
            };
            // First term (k=0): 1 / Γ(ν+1)
            let gamma_nu1 = arb_gamma_real(
                &order.add(&BigFloat::from_i32(1, wp), wp, RM),
                &mut w.at(wp),
            )?;
            (prefix, BigFloat::from_i32(1, wp).div(&gamma_nu1, wp, RM))
        };
        sum = sum.add(&term, wp, RM);

        // The terms peak near k ≈ |x|/2 and then decay super-exponentially;
        // `|x| + wp` comfortably covers the tail down to 2^{-wp}.
        let max_terms = (x_f64.abs() * 1.5) as usize + wp + 40;
        for k in 1..=max_terms {
            // term *= (-x²/4) / (k · (ν + k))
            let k_bf = BigFloat::from_i32(k as i32, wp);
            let nu_plus_k = order.add(&k_bf, wp, RM);
            let denom = k_bf.mul(&nu_plus_k, wp, RM);
            term = term.mul(&neg_x_half_sq, wp, RM);
            term = term.div(&denom, wp, RM);

            // Convergence check (only once past the peak of the terms).
            if (k as f64) > x_f64.abs() / 2.0
                && let (Some(t_exp), Some(s_exp)) = (term.exponent(), sum.exponent())
                && (s_exp as i64 - t_exp as i64) > wp as i64
            {
                tracing::trace!(k, "arb_bessel_j: series converged");
                break;
            }

            sum = sum.add(&term, wp, RM);
        }

        let r = prefix.mul(&sum, wp, RM);
        if large && (r.is_zero() || r.is_inf()) {
            return Err(SymplexError::Unevaluable {
                reason: "besselj of large order: the value leaves the arbitrary-precision \
                         exponent range"
                    .into(),
            });
        }
        Ok(round_to(r, prec))
    }
}

/// Arbitrary-precision Bessel function of the second kind Y_ν(x), `x > 0`.
///
/// * Large `x` (see [`hankel_threshold`]): Hankel expansion
///   ([`bessel_jy_hankel`]), falling back to the series if it cannot reach
///   the target precision.
/// * Otherwise, integer order `n` only: the Neumann series (A&S 9.1.11)
///   with guard bits for the `e^{x}` cancellation.
fn arb_bessel_y(order: &BigFloat, x: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let classic = arb_bessel_y_classic(order, x, w);
    debye_fallback(bessel_debye::Kind::Y, order, x, w, classic)
}

/// `Y_ν(x)`, `x > 0`, for a non-integer order below the Hankel range, from
/// the Bessel functions of the first kind (DLMF 10.2.3):
/// `Y_ν = (J_ν cos νπ − J_{−ν})/sin νπ`, and for a half-integer order
/// `ν = n + ½` exactly `Y_ν = (−1)^{n+1} J_{−ν}` (`cos νπ = 0`; a finite
/// trigonometric form, DLMF 10.16.1: `Y_{1/2}(x) = −√(2/(πx)) cos x`).
/// The difference cancels where `sin νπ` is small or `J_ν cos νπ ≈
/// J_{−ν}`: it is evaluated again with the bits lost.  Before 0.54 refused
/// ("not yet implemented in small-|x| regime": `bessely(1/2, 2)`).
fn bessel_y_from_j(
    order: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    let exact = order.mantissa_max_bit_len().unwrap_or(64) + 8;
    let twice = order.mul(&BigFloat::from_i32(2, 8), exact, RM);
    if twice.is_int() {
        // ν = n + ½ with 2ν = 2n + 1: (−1)^{n+1} is −1 for 2ν ≡ 1 (mod 4).
        let Some(t) = bf_as_int(&twice)? else {
            return Err(unevaluable("bessely: half-integer order out of range"));
        };
        let j = arb_bessel_j(&order.neg(), x, w)?;
        return Ok(if t.rem_euclid(4) == 1 { j.neg() } else { j });
    }
    let mut extra = 32usize;
    for _ in 0..4 {
        let wp = prec + extra;
        let mut ww = w.at(wp);
        let jp = arb_bessel_j(order, x, &mut ww)?;
        let jm = arb_bessel_j(&order.neg(), x, &mut ww)?;
        let pi = ww.cc.pi(wp, RM).clone();
        let t = order.mul(&pi, wp, RM);
        let (s, c) = (t.sin(wp, RM, ww.cc), t.cos(wp, RM, ww.cc));
        let jc = jp.mul(&c, wp, RM);
        let num = jc.sub(&jm, wp, RM);
        let e = |v: &BigFloat| v.exponent().map_or(i64::MIN / 4, i64::from);
        // Bits lost in the difference, and in `sin νπ` (its argument's
        // rounding is relative to `νπ`, not to the sine).
        let lost = (e(&jc).max(e(&jm)) - e(&num)).max(0) + (e(&t) - e(&s)).max(0) + 4;
        let lost = usize::try_from(lost).unwrap_or(usize::MAX / 8);
        if lost + 16 <= extra {
            return Ok(round_to(num.div(&s, wp, RM), prec));
        }
        if lost > 4 * prec + 512 {
            break;
        }
        extra = lost + 32;
    }
    Err(unevaluable(
        "bessely: J_ν cos νπ − J_{−ν} cancels beyond the working precision",
    ))
}

/// [`arb_bessel_y`] by the Hankel expansion and the Neumann series.
fn arb_bessel_y_classic(
    order: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    let guard = 32;
    let wp = prec + guard;

    if x.is_zero() || x.is_negative() {
        return Err(SymplexError::Unevaluable {
            reason: "BesselY undefined at x ≤ 0".into(),
        });
    }

    refuse_bessel_argument(x, prec)?;
    let x_f64 = f64_of(x)?;
    let order_f64 = f64_of(order)?;
    let order_int = order_f64.round() as i64;
    // Exactly (see `arb_bessel_j`): a non-integer order near an integer is
    // refused below the Hankel range rather than taken for the integer.
    let is_int_order = order.is_int();

    // Negative integer order: Y_{-n}(x) = (-1)^n Y_n(x) (exactly, see
    // `arb_bessel_j`).
    if is_int_order && order.is_negative() {
        let y = arb_bessel_y(&order.neg(), x, w)?;
        return Ok(if bessel_order::is_odd(order) {
            y.neg()
        } else {
            y
        });
    }

    if x_f64 >= hankel_threshold(wp, order_f64.abs())
        && let Some(y) = bessel_jy_hankel(order, x, false, &mut w.at(wp))
    {
        tracing::trace!("arb_bessel_y: used Hankel P/Q expansion");
        return Ok(round_to(y, prec));
    }

    // ── Small |x|, integer order n ≥ 0: Neumann series (A&S 9.1.11) ─────
    //
    //   Y_n(x) = −(1/π)(x/2)^{−n} Σ_{k=0}^{n−1} (n−k−1)!/k! (x²/4)^k
    //            + (2/π) ln(x/2) J_n(x)
    //            − (1/π)(x/2)^n Σ_{k≥0} [ψ(k+1) + ψ(n+k+1)] (−x²/4)^k / (k!(n+k)!)
    //
    // with ψ(m+1) = −γ + H_m.  Computed directly for every n (no forward
    // recurrence or Wronskian division, which is unstable near zeros of J).
    // The log term and the series cancel to O(e^{x}) relative size, so a
    // few extra guard bits are added.

    if !is_int_order {
        return bessel_y_from_j(order, x, w);
    }
    // A large order: the finite part of the series alone where the rest is
    // negligible (`bessel_order::neumann_leading`).  Before 0.34 the series
    // took `n` steps in exact rationals: `bessely(5000, 1)` 17 s,
    // `bessely(10⁵, 1)` never finished.
    if order_f64 >= bessel_order::LARGE_ORDER {
        return bessel_order::neumann_leading(bessel_order::Second::Y, order, x, w)
            .unwrap_or_else(|| Err(large_order_refused("bessely", order_f64, x_f64)));
    }
    let n = order_int as usize;
    // Beyond the cap of the cancellation, the finite part of the series alone
    // where the rest is negligible (`bessel_order::neumann_leading`): before
    // 0.35 `bessely(3909, 3909/2)` was refused.
    let guard = match series_guard_bits(x_f64, prec) {
        Ok(g) => g,
        Err(e) => {
            return bessel_order::neumann_leading(bessel_order::Second::Y, order, x, w)
                .unwrap_or(Err(e));
        }
    };
    let wp = wp + guard;
    let mut xw = x.clone();
    let _ = xw.set_precision(wp, RM);

    let pi = w.cc.pi(wp, RM).clone();
    let one = BigFloat::from_i32(1, wp);
    let two = BigFloat::from_i32(2, wp);
    let inv_pi = one.div(&pi, wp, RM);
    let x_half = xw.div(&two, wp, RM);
    let x_half_sq = x_half.mul(&x_half, wp, RM);
    let ln_x_half = x_half.ln(wp, RM, w.cc);
    let euler_gamma = arb_euler_gamma(&mut w.at(wp))?;
    let n_bf = BigFloat::from_i128(n as i128, wp);
    let j_n = arb_bessel_j(&n_bf, &xw, &mut w.at(wp))?;

    // Finite sum: −(1/π)(x/2)^{−n} Σ_{k<n} (n−k−1)!/k! (x²/4)^k
    let mut finite = BigFloat::new(wp);
    if n > 0 {
        let mut f = NeumannCoefficients::new(n);
        let mut pow = one.clone();
        for k in 0..n {
            if k > 0 {
                pow = pow.mul(&x_half_sq, wp, RM);
            }
            let coeff = f.next(wp);
            finite = finite.add(&coeff.mul(&pow, wp, RM), wp, RM);
        }
        let x_half_pow_neg_n = one.div(&x_half.powi(n, wp, RM), wp, RM);
        finite = inv_pi
            .mul(&x_half_pow_neg_n, wp, RM)
            .mul(&finite, wp, RM)
            .neg();
    }

    // Log term: (2/π) ln(x/2) J_n(x)
    let log_term = two
        .mul(&inv_pi, wp, RM)
        .mul(&ln_x_half, wp, RM)
        .mul(&j_n, wp, RM);

    // Series: −(1/π)(x/2)^n Σ_k [ψ(k+1) + ψ(n+k+1)] (−x²/4)^k / (k!(n+k)!)
    let mut h_k = BigFloat::new(wp);
    let mut h_nk = BigFloat::new(wp);
    for m in 1..=n {
        h_nk = h_nk.add(
            &one.div(&BigFloat::from_i128(m as i128, wp), wp, RM),
            wp,
            RM,
        );
    }
    let n_fact = factorial_exact(n);
    let mut inv_fact = one.div(&ratio_to_bigfloat(&n_fact, wp), wp, RM);
    let neg_x_half_sq = x_half_sq.neg();
    let mut pow = one.clone();
    let mut series = BigFloat::new(wp);
    let two_gamma = euler_gamma.mul(&two, wp, RM);
    let max_terms = (x_f64.abs() * 2.0) as usize + wp + 40;
    for k in 0..=max_terms {
        if k > 0 {
            let k_bf = BigFloat::from_i128(k as i128, wp);
            let nk_bf = BigFloat::from_i128((n + k) as i128, wp);
            h_k = h_k.add(&one.div(&k_bf, wp, RM), wp, RM);
            h_nk = h_nk.add(&one.div(&nk_bf, wp, RM), wp, RM);
            inv_fact = inv_fact.div(&k_bf.mul(&nk_bf, wp, RM), wp, RM);
            pow = pow.mul(&neg_x_half_sq, wp, RM);
        }
        let psi_sum = h_k.add(&h_nk, wp, RM).sub(&two_gamma, wp, RM);
        let term = psi_sum.mul(&pow, wp, RM).mul(&inv_fact, wp, RM);
        series = series.add(&term, wp, RM);
        if k > 2
            && let (Some(t_exp), Some(s_exp)) = (term.exponent(), series.exponent())
            && (s_exp as i64 - t_exp as i64) > wp as i64
        {
            tracing::trace!(k, "arb_bessel_y: Neumann series converged");
            break;
        }
    }
    let series_term = inv_pi
        .mul(&x_half.powi(n, wp, RM), wp, RM)
        .mul(&series, wp, RM)
        .neg();

    let y = finite.add(&log_term, wp, RM).add(&series_term, wp, RM);
    Ok(round_to(y, prec))
}

/// `n!` as an exact rational (an integer): a product of integers.  Not
/// `Ratio` arithmetic, whose every product is reduced by a gcd with the
/// denominator 1 — a binary gcd that takes a step per bit of the numerator,
/// `O(n²)` words for `n!`: before 0.34 `polygamma(1755, 1/10)` spent 5 s
/// in it in the fuzz build, `bessely(5000, 1)` 17 s.  The same value.
fn factorial_exact(n: usize) -> Ratio<BigInt> {
    Ratio::from_integer(crate::base::combinatorics::factorial(n as u64))
}

/// The coefficients `f_k = (n−k−1)!/k!`, `k = 0 … n−1`, of the finite sums of
/// `Y_n` and `K_n` (A&S 9.1.11, 9.6.11), as BigFloats: exactly the rounding of
/// the reduced fraction `ratio_to_bigfloat` made of the `Ratio` recurrence
/// `f_k = f_{k−1}/((n−k)·k)` before 0.34, without its gcds.  The reduced
/// fraction is an integer `(k+1)⋯(n−k−1)` while `k ≤ n−k−1` (`k!` divides
/// `(n−k−1)!`) and the unit fraction `1/((n−k)⋯k)` after: integer
/// recurrences with exact divisions and products.
struct NeumannCoefficients {
    n: usize,
    k: usize,
    /// `(n−k−1)!/k!` in the integer phase, the denominator in the other.
    value: BigInt,
}

impl NeumannCoefficients {
    fn new(n: usize) -> Self {
        let value = crate::base::combinatorics::factorial(n.saturating_sub(1) as u64);
        NeumannCoefficients { n, k: 0, value }
    }

    /// `f_k` at `wp` bits, then `k + 1`.
    fn next(&mut self, wp: usize) -> BigFloat {
        let (n, k) = (self.n, self.k);
        self.k += 1;
        if k > 0 {
            let (a, b) = (BigInt::from((n - k) as u64), BigInt::from(k as u64));
            if 2 * k < n {
                // (n−k)!/(k−1)! ÷ ((n−k)·k): exact.
                self.value = &self.value / (a * b);
            } else if 2 * k < n + 2 {
                // The first unit fraction: 1/k (n = 2k), 1/((k−1)·k) (n = 2k−1).
                self.value = (n - k..=k).map(|j| BigInt::from(j as u64)).product();
            } else {
                self.value = &self.value * a * b;
            }
        }
        if 2 * k < n {
            ratio_to_bigfloat(&Ratio::from_integer(self.value.clone()), wp)
        } else {
            ratio_to_bigfloat(&Ratio::new_raw(BigInt::from(1), self.value.clone()), wp)
        }
    }
}

/// `Y_n` or `K_n` of a large order where [`bessel_order::neumann_leading`]
/// does not apply (`x² > 2(n − 1)`): the full series would take `n` steps.
fn large_order_refused(what: &str, order: f64, x: f64) -> SymplexError {
    SymplexError::NotImplemented(format!(
        "{what} of integer order {order:e} at {x:e} (above sqrt(2n)): the Neumann series would \
         take n terms (no uniform asymptotic expansion)"
    ))
}

// ═══════════════════════════════════════════════════════════════════════════
// Catalan's constant
// ═══════════════════════════════════════════════════════════════════════════

/// Catalan's constant `G` at `prec` bits via Ramanujan's formula
///
/// ```text
/// G = (π/8)·ln(2 + √3) + (3/8)·Σ_{n≥0} 1 / ((2n+1)² · C(2n, n))
/// ```
///
/// The series gains ~2 bits per term (`1/C(2n,n) ~ 4^{-n}`), so about
/// `prec/2` terms are needed; every term is positive (no cancellation).
fn arb_catalan(w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    let wp = prec + 32;
    let one = BigFloat::from_i32(1, wp);
    let two = BigFloat::from_i32(2, wp);
    let three = BigFloat::from_i32(3, wp);
    let eight = BigFloat::from_i32(8, wp);

    // (π/8)·ln(2 + √3)
    let sqrt3 = three.sqrt(wp, RM);
    let ln_term = two.add(&sqrt3, wp, RM).ln(wp, RM, w.cc);
    let pi = w.cc.pi(wp, RM).clone();
    let first = pi.div(&eight, wp, RM).mul(&ln_term, wp, RM);

    // Σ b_n / (2n+1)² with b_n = 1/C(2n,n), b_{n+1} = b_n·(n+1)/(2(2n+1)).
    let mut b = one.clone();
    let mut sum = BigFloat::new(wp);
    let max_terms = wp / 2 + 64;
    for n in 0..max_terms {
        let two_n_plus_1 = BigFloat::from_i128(2 * n as i128 + 1, wp);
        let denom = two_n_plus_1.mul(&two_n_plus_1, wp, RM);
        let term = b.div(&denom, wp, RM);
        sum = sum.add(&term, wp, RM);
        if let (Some(t_exp), Some(s_exp)) = (term.exponent(), sum.exponent())
            && (s_exp as i64 - t_exp as i64) > wp as i64
        {
            break;
        }
        let n_plus_1 = BigFloat::from_i128(n as i128 + 1, wp);
        let two_2n_plus_1 = two.mul(&two_n_plus_1, wp, RM);
        b = b.mul(&n_plus_1, wp, RM).div(&two_2n_plus_1, wp, RM);
    }
    let second = three.div(&eight, wp, RM).mul(&sum, wp, RM);
    let mut g = first.add(&second, wp, RM);
    g.set_precision(prec, RM)
        .map_err(|e| SymplexError::ComputationFailed {
            operation: "catalan",
            reason: format!("precision adjustment failed: {e:?}"),
        })?;
    Ok(g)
}

// ═══════════════════════════════════════════════════════════════════════════
// Trigonometric / exponential / logarithmic integrals
// ═══════════════════════════════════════════════════════════════════════════

/// Number of guard bits needed to absorb the cancellation in an
/// alternating power series whose largest term is `~e^{|x|}`.
///
/// Saturating: `as usize` takes a huge or infinite `|x|·log₂e` to
/// `usize::MAX`, and the sum must stay there for the callers' cap to refuse
/// it.  Up to 0.46 the `+ 32` overflowed: a panic with overflow checks (the
/// CI fuzz builds: `besselj` of an argument near `2¹⁰⁰⁰` that the Hankel
/// expansion declined, met by `fuzz_roundtrip`), and in release builds a
/// wrap to 31 guard bits for a series that needed `10³⁰⁰`.
fn cancellation_guard_bits(x_abs: f64) -> usize {
    ((x_abs * std::f64::consts::LOG2_E).ceil() as usize).saturating_add(32)
}

/// [`cancellation_guard_bits`] for the power series of `J_ν` and `Y_n` at
/// `|x|`, refused beyond [`cancellation_cap`]: the series also takes about
/// `|x|` terms.  Before 0.30 nothing bounded them — `besselj(5/2, 10²⁵)`
/// asked for `1.4·10²⁵` bits and printed garbage (`3.8·10¹⁰⁵²⁷`).
fn series_guard_bits(x_abs: f64, prec: usize) -> Result<usize, SymplexError> {
    let guard = cancellation_guard_bits(x_abs);
    if !x_abs.is_finite() || guard > cancellation_cap(prec) {
        return Err(special_exhausted(prec));
    }
    Ok(guard)
}

/// Refuse a Bessel argument whose Hankel phase `x − νπ/2 − π/4` would
/// need more than [`cancellation_cap`] extra bits (`log₂|x|` of them, see
/// [`bessel_jy_hankel`]): `besselj(−5, erfi(1210))`, an argument of
/// `2^(2·10⁶)`, ran for minutes.
fn refuse_bessel_argument(x: &BigFloat, prec: usize) -> Result<(), SymplexError> {
    let cap = i64::try_from(cancellation_cap(prec)).unwrap_or(i64::MAX);
    if x.is_inf() || x.exponent().is_some_and(|e| i64::from(e) > cap) {
        return Err(special_exhausted(prec));
    }
    Ok(())
}

/// Auxiliary functions `f(x)`, `g(x)` of the asymptotic expansions
///
/// ```text
/// Si(x) = π/2 − f(x) cos x − g(x) sin x,   Ci(x) = f(x) sin x − g(x) cos x
/// f(x) ~ (1/x) Σ (−1)^k (2k)!/x^{2k},   g(x) ~ (1/x²) Σ (−1)^k (2k+1)!/x^{2k}
/// ```
///
/// for `x > 0`.  Uses optimal truncation (stop when terms grow); the
/// caller guarantees `x` is large enough that the smallest term is below
/// the working precision.
fn si_ci_asymptotic_fg(x: &BigFloat, wp: usize) -> (BigFloat, BigFloat) {
    let one = BigFloat::from_i32(1, wp);
    let inv_x = one.div(x, wp, RM);
    let inv_x2 = inv_x.mul(&inv_x, wp, RM);

    let mut f_sum = BigFloat::new(wp);
    let mut g_sum = BigFloat::new(wp);
    // t_f = (2k)!/x^{2k}, t_g = (2k+1)!/x^{2k}
    let mut t_f = one.clone();
    let mut t_g = one.clone();
    let mut prev_f: Option<BigFloat> = None;
    let max_terms = wp * 2 + 100;
    for k in 0..max_terms {
        if k > 0 {
            // (2k)! / (2k-2)! = (2k-1)(2k);  (2k+1)!/(2k-1)! = (2k)(2k+1)
            let a = BigFloat::from_i128((2 * k as i128 - 1) * (2 * k as i128), wp);
            let b = BigFloat::from_i128((2 * k as i128) * (2 * k as i128 + 1), wp);
            t_f = t_f.mul(&a, wp, RM).mul(&inv_x2, wp, RM);
            t_g = t_g.mul(&b, wp, RM).mul(&inv_x2, wp, RM);
        }
        // Optimal truncation: stop once terms start growing.
        if let Some(ref p) = prev_f
            && t_f.abs().cmp(p).is_some_and(|c| c > 0)
        {
            break;
        }
        if k % 2 == 0 {
            f_sum = f_sum.add(&t_f, wp, RM);
            g_sum = g_sum.add(&t_g, wp, RM);
        } else {
            f_sum = f_sum.sub(&t_f, wp, RM);
            g_sum = g_sum.sub(&t_g, wp, RM);
        }
        if let (Some(t_exp), Some(s_exp)) = (t_f.exponent(), f_sum.exponent())
            && (s_exp as i64 - t_exp as i64) > wp as i64
        {
            break;
        }
        prev_f = Some(t_f.abs());
    }
    (f_sum.mul(&inv_x, wp, RM), g_sum.mul(&inv_x2, wp, RM))
}

/// Sine and cosine integrals for real `x`.
///
/// * `want_si == true`: returns `(Si(x), 0)` — valid for all real `x`
///   (`Si` is odd).
/// * `want_si == false`: returns `(0, Ci(x))` — requires `x > 0`.
///
/// Power series for moderate `|x|` (with guard bits for the `e^{|x|}`
/// cancellation) and the Hankel-type asymptotic expansion when
/// `|x| > wp·ln 2` (where its optimal-truncation error `~e^{−x}` is below
/// the working precision).
fn arb_si_ci(
    x: &BigFloat,
    want_si: bool,
    w: &mut Wp<'_>,
) -> Result<(BigFloat, BigFloat), SymplexError> {
    let prec = w.bits();
    if x.is_nan() || x.is_inf_pos() || x.is_inf_neg() {
        return Err(SymplexError::Unevaluable {
            reason: "Si/Ci of special float value".into(),
        });
    }
    let x_f64 = f64_of(x)?;
    let x_abs = x_f64.abs();
    let base_wp = prec + 32;

    if x.is_zero() {
        return if want_si {
            Ok((BigFloat::new(prec), BigFloat::new(prec)))
        } else {
            Err(SymplexError::Unevaluable {
                reason: "Ci(0) is zoo".into(),
            })
        };
    }
    if !want_si && x.is_negative() {
        return Err(SymplexError::Unevaluable {
            reason: "Ci requires a positive argument here".into(),
        });
    }

    // Far out `f`, `g` are below `2^−wp` next to π/2: `Si(x) = ±π/2`; `Ci(x)
    // ≈ sin(x)/x` needs `x` reduced modulo 2π, refused beyond
    // `cancellation_cap` bits (before 0.30 both reduced any `x`, and
    // `Si(erfi(765))`, `x ≈ 2^844000`, hung).
    let e = usize::try_from(x.exponent().map_or(0, i64::from).max(0)).unwrap_or(0);
    if e > base_wp + 2 {
        if !want_si {
            if e > cancellation_cap(prec) {
                return Err(special_exhausted(prec));
            }
        } else {
            let half_pi =
                w.cc.pi(base_wp, RM)
                    .div(&BigFloat::from_i32(2, base_wp), base_wp, RM);
            let si = if x.is_negative() {
                half_pi.neg()
            } else {
                half_pi
            };
            return Ok((round_to(si, prec), BigFloat::new(prec)));
        }
    }
    let switch = (base_wp as f64) * std::f64::consts::LN_2;
    if x_abs > switch {
        // ── Asymptotic expansion ──
        let wp = base_wp;
        let ax = x.abs();
        let (f, g) = si_ci_asymptotic_fg(&ax, wp);
        let sin_x = ax.sin(wp, RM, w.cc);
        let cos_x = ax.cos(wp, RM, w.cc);
        let pi = w.cc.pi(wp, RM).clone();
        let two = BigFloat::from_i32(2, wp);
        let half_pi = pi.div(&two, wp, RM);
        let mut si =
            half_pi
                .sub(&f.mul(&cos_x, wp, RM), wp, RM)
                .sub(&g.mul(&sin_x, wp, RM), wp, RM);
        if x.is_negative() {
            si = si.neg();
        }
        let ci = f.mul(&sin_x, wp, RM).sub(&g.mul(&cos_x, wp, RM), wp, RM);
        return Ok((round_to(si, prec), round_to(ci, prec)));
    }

    // ── Power series ──
    let wp = base_wp + cancellation_guard_bits(x_abs);
    let mut xw = x.clone();
    let _ = xw.set_precision(wp, RM);
    let x2 = xw.mul(&xw, wp, RM);
    let max_terms = (x_abs * 1.5) as usize + wp + 40;

    let si = if want_si {
        // Si(x) = Σ (−1)^k t_k/(2k+1),  t_k = x^{2k+1}/(2k+1)!
        let mut t = xw.clone();
        let mut sum = BigFloat::new(wp);
        for k in 0..max_terms {
            if k > 0 {
                let d = BigFloat::from_i128((2 * k as i128) * (2 * k as i128 + 1), wp);
                t = t.mul(&x2, wp, RM).div(&d, wp, RM);
            }
            let term = t.div(&BigFloat::from_i128(2 * k as i128 + 1, wp), wp, RM);
            if k % 2 == 0 {
                sum = sum.add(&term, wp, RM);
            } else {
                sum = sum.sub(&term, wp, RM);
            }
            if let (Some(t_exp), Some(s_exp)) = (term.exponent(), sum.exponent())
                && (s_exp as i64 - t_exp as i64) > wp as i64
            {
                break;
            }
        }
        round_to(sum, prec)
    } else {
        BigFloat::new(prec)
    };

    let ci = if want_si {
        BigFloat::new(prec)
    } else {
        // Ci(x) = γ + ln x + Σ_{k≥1} (−1)^k u_k/(2k),  u_k = x^{2k}/(2k)!
        let gamma = arb_euler_gamma(&mut w.at(wp))?;
        let ln_x = xw.ln(wp, RM, w.cc);
        let mut sum = BigFloat::new(wp);
        let mut u = BigFloat::from_i32(1, wp);
        for k in 1..max_terms {
            let d = BigFloat::from_i128((2 * k as i128 - 1) * (2 * k as i128), wp);
            u = u.mul(&x2, wp, RM).div(&d, wp, RM);
            let term = u.div(&BigFloat::from_i128(2 * k as i128, wp), wp, RM);
            if k % 2 == 0 {
                sum = sum.add(&term, wp, RM);
            } else {
                sum = sum.sub(&term, wp, RM);
            }
            if let (Some(t_exp), Some(s_exp)) = (term.exponent(), sum.exponent())
                && (s_exp as i64 - t_exp as i64) > wp as i64
            {
                break;
            }
        }
        let ci = gamma.add(&ln_x, wp, RM).add(&sum, wp, RM);
        round_to(ci, prec)
    };

    Ok((si, ci))
}

/// Exponential integral `Ei(x)` for real `x ≠ 0`.
///
/// * `|x| ≤ wp·ln 2`: `Ei(x) = γ + ln|x| + Σ_{k≥1} xᵏ/(k·k!)`, with guard
///   bits for the alternating cancellation when `x < 0`;
/// * otherwise the asymptotic expansion `Ei(x) ~ (eˣ/x) Σ k!/xᵏ` with
///   optimal truncation (relative error `~e^{−|x|}`).
fn arb_ei(x: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    if x.is_nan() || x.is_inf_pos() || x.is_inf_neg() || x.is_zero() {
        return Err(SymplexError::Unevaluable {
            reason: "Ei of zero or special float value".into(),
        });
    }
    let x_f64 = f64_of(x)?;
    let x_abs = x_f64.abs();
    let base_wp = prec + 32;
    let switch = (base_wp as f64) * std::f64::consts::LN_2;

    if x_abs > switch {
        // ── Asymptotic: Ei(x) ~ (e^x / x) Σ_{k≥0} k! / x^k ──
        let wp = base_wp;
        let (inv_x, sum) = ei_asymptotic_series(x, wp);
        let ex = x.exp(wp, RM, w.cc);
        let r = ex.mul(&inv_x, wp, RM).mul(&sum, wp, RM);
        return Ok(round_to(r, prec));
    }

    // ── Power series ──
    // For x < 0 the terms reach `~e^{|x|}` and `Ei(x) ~ e^{−|x|}/|x|`: the
    // sum cancels about `2|x|·log₂e` bits, and the value near the zero
    // `x ≈ 0.3725` more.  The loss is measured (the largest term against
    // the value) and the series repeated with that many more bits, up to
    // `cancellation_cap`.  Before 0.30 the guard was `|x|·log₂e`, half
    // of it: `Ei(−196)` at 60 digits was wrong from its 24th digit.
    let cap = cancellation_cap(prec);
    let mut extra = if x.is_negative() {
        cancellation_guard_bits(2.0 * x_abs)
    } else {
        32
    };
    loop {
        let wp = prec + extra;
        let mut xw = x.clone();
        let _ = xw.set_precision(wp, RM);
        let gamma = arb_euler_gamma(&mut w.at(wp))?;
        let ln_abs_x = xw.abs().ln(wp, RM, w.cc);
        let mut largest = gamma.abs().max(&ln_abs_x.abs());
        let mut v = BigFloat::from_i32(1, wp); // x^k/k!
        let mut sum = BigFloat::new(wp);
        let max_terms = (x_abs * 3.0) as usize + wp + 40;
        for k in 1..max_terms {
            let k_bf = BigFloat::from_i128(k as i128, wp);
            v = v.mul(&xw, wp, RM).div(&k_bf, wp, RM);
            let term = v.div(&k_bf, wp, RM);
            sum = sum.add(&term, wp, RM);
            largest = largest.max(&term.abs()).max(&sum.abs());
            if let (Some(t_exp), Some(s_exp)) = (term.exponent(), sum.exponent())
                && (s_exp as i64 - t_exp as i64) > wp as i64
                && (k as f64) > x_abs
            {
                break;
            }
        }
        let r = gamma.add(&ln_abs_x, wp, RM).add(&sum, wp, RM);
        let lost = match (largest.exponent(), r.exponent()) {
            (Some(l), Some(v)) if !r.is_zero() => (i64::from(l) - i64::from(v)).max(0) as usize,
            _ => wp,
        };
        if lost + 32 <= extra {
            return Ok(round_to(r, prec));
        }
        if extra >= cap {
            return Err(special_exhausted(prec));
        }
        extra = (lost + 64).min(cap).max(extra + 32);
    }
}

/// The asymptotic series of `Ei(x) ~ (e^x/x)·Σ_{k≥0} k!/x^k` at `wp` bits,
/// truncated at its smallest term: `(1/x, Σ)`.  [`arb_ei`] multiplies them
/// by `e^x`; `extended` by `e^x` at its own scale, below the exponent range.
fn ei_asymptotic_series(x: &BigFloat, wp: usize) -> (BigFloat, BigFloat) {
    let one = BigFloat::from_i32(1, wp);
    let inv_x = one.div(x, wp, RM);
    let mut term = one.clone();
    let mut sum = BigFloat::new(wp);
    let mut prev: Option<BigFloat> = None;
    let max_terms = wp * 2 + 100;
    for k in 0..max_terms {
        if k > 0 {
            term = term
                .mul(&BigFloat::from_i128(k as i128, wp), wp, RM)
                .mul(&inv_x, wp, RM);
        }
        if let Some(ref p) = prev
            && term.abs().cmp(p).is_some_and(|c| c > 0)
        {
            break;
        }
        sum = sum.add(&term, wp, RM);
        if let (Some(t_exp), Some(s_exp)) = (term.exponent(), sum.exponent())
            && (s_exp as i64 - t_exp as i64) > wp as i64
        {
            break;
        }
        prev = Some(term.abs());
    }
    (inv_x, sum)
}

/// Round a `BigFloat` down to `prec` bits (no-op on failure).
fn round_to(mut v: BigFloat, prec: usize) -> BigFloat {
    let _ = v.set_precision(prec, RM);
    v
}

/// `b^e` for real `b > 0` (or integer `e`): [`crate::base::bigfloat::pow`],
/// which never calls the non-terminating `BigFloat::pow`.
fn bf_pow(b: &BigFloat, e: &BigFloat, w: &mut Wp<'_>) -> BigFloat {
    let wp = w.bits();
    crate::base::bigfloat::pow(b, e, wp, RM, w.cc)
}

// ═══════════════════════════════════════════════════════════════════════════
// Riemann zeta function
// ═══════════════════════════════════════════════════════════════════════════

/// `ζ(s)` for real `s ≠ 1`.
///
/// * `s > 0`: Borwein's algorithm (P. Borwein, *An efficient algorithm for
///   the Riemann zeta function*, 1991, Algorithm 2) — an accelerated
///   alternating series with error `≤ 3/(3+√8)ⁿ`, so `n ≈ 0.39·wp` terms.
/// * `s < 0`: functional equation
///   `ζ(s) = 2ˢ π^{s−1} sin(πs/2) Γ(1−s) ζ(1−s)`; trivial zeros at the
///   negative even integers are returned exactly.
/// * `s = 0`: `−1/2`.
fn arb_zeta(s: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    if s.is_nan() || s.is_inf_neg() {
        return Err(SymplexError::Unevaluable {
            reason: "zeta of special float value".into(),
        });
    }
    if s.is_inf_pos() {
        return Ok(BigFloat::from_i32(1, prec));
    }
    if s.is_zero() {
        return Ok(BigFloat::from_f64(-0.5, prec));
    }
    let s_f64 = f64_of(s)?;
    // The pole exactly (before 0.30 in `f64`, where `1 + 10⁻⁴⁵` is 1).
    if *s == BigFloat::from_i32(1, 64) {
        return Err(SymplexError::Unevaluable {
            reason: "zeta(1) is a pole".into(),
        });
    }
    let mut wp = prec + 32;

    if s.is_negative() {
        // `2^s`, `π^(s−1)` and `Γ(1 − s)` come from exponentials of
        // arguments of about `|s|·ln|s|`, whose absolute error is their
        // relative one: as many guard bits.
        wp += magnitude_bits(s_f64.abs() * (s_f64.abs().ln().max(0.0) + 2.0));
        // Trivial zeros (the `f64` test only filters; the exact difference
        // below decides).
        let s_round = s_f64.round();
        if (s_f64 - s_round).abs() < 1e-14 && (s_round as i64) % 2 == 0 {
            let mut sw = s.clone();
            let _ = sw.set_precision(wp, RM);
            let diff = sw.sub(&bigfloat::from_f64(s_round, wp), wp, RM);
            if diff.is_zero() {
                return Ok(BigFloat::new(prec));
            }
        }
        // An `s` whose unit in the last place at the working precision is
        // 2 or more has no known parity: `sin(πs/2)` is undetermined (and
        // `ζ(s)` is a trivial zero or far beyond the exponent range).  The
        // argument reduction of that `sin` for `s ≈ −2^(1.4·10⁶)` ran for
        // minutes (`zeta(erfi(−1000))`, found by fuzz_evalf in 0.31).
        if s.exponent().is_some_and(|e| i64::from(e) > wp as i64) {
            return Err(special_exhausted(prec));
        }
        // ζ(s) = 2^s π^{s−1} sin(π s/2) Γ(1−s) ζ(1−s), `1 − s` exact: next
        // to 0 the factors `sin(πs/2) ≈ πs/2` and `ζ(1 − s) ≈ −1/s` are both
        // relative to `s` (before 0.30 `1 − s` was rounded, and `ζ(−10⁻⁶⁰)`
        // printed `−0.499999999999999999141761477346` at 30 digits).
        let mut sw = s.clone();
        let _ = sw.set_precision(wp, RM);
        let one = BigFloat::from_i32(1, wp);
        let two = BigFloat::from_i32(2, wp);
        let pi = w.cc.pi(wp, RM).clone();
        let one_minus_s = BigFloat::from_i32(1, 64).sub(s, exact_bits(s, wp), RM);
        let two_pow_s = bf_pow(&two, &sw, &mut w.at(wp));
        let s_minus_1 = sw.sub(&one, wp, RM);
        let pi_pow = bf_pow(&pi, &s_minus_1, &mut w.at(wp));
        // sin(πs/2) = (−1)ⁿ·sin(πd) for s/2 = n + d, `d` exact: next to a
        // trivial zero `sin` of a rounded `πs/2` lost `log₂(1/|d|)` bits
        // (up to 0.43 `zeta(−2 − 2⁻¹⁵⁰)` at 40 digits was wrong from its
        // 33rd digit, `zeta(−70 − 2⁻¹³⁷)` from its 35th).
        let sin_term = sin_half_pi(s, &mut w.at(wp));
        let gamma_term = arb_gamma_real(&one_minus_s, &mut w.at(wp))?;
        let zeta_term = arb_zeta_borwein(&one_minus_s, &mut w.at(wp))?;
        let r = two_pow_s
            .mul(&pi_pow, wp, RM)
            .mul(&sin_term, wp, RM)
            .mul(&gamma_term, wp, RM)
            .mul(&zeta_term, wp, RM);
        return Ok(round_to(r, prec));
    }

    let r = arb_zeta_borwein(s, &mut w.at(wp))?;
    Ok(round_to(r, prec))
}

/// `sin(πs/2)` to `wp` bits relative for a finite real `s`: with `s/2 = n + d`,
/// `n` the nearest integer and `d` formed exactly, `(−1)ⁿ·sin(πd)` — exactly 0
/// at the even integers, and relative to `d` next to them.
fn sin_half_pi(s: &BigFloat, w: &mut Wp<'_>) -> BigFloat {
    let wp = w.bits();
    let exact = exact_bits(s, wp);
    let half = s.div(&BigFloat::from_i32(2, 64), exact, RM);
    let n = half.add(&BigFloat::from_f64(0.5, 64), exact, RM).floor();
    let d = half.sub(&n, exact, RM);
    if d.is_zero() {
        return BigFloat::new(wp);
    }
    let pi = w.cc.pi(wp + 8, RM).clone();
    let v = pi.mul(&d, wp + 8, RM).sin(wp + 8, RM, w.cc);
    let odd = !n
        .div(&BigFloat::from_i32(2, 64), exact, RM)
        .fract()
        .is_zero();
    round_to(if odd { v.neg() } else { v }, wp)
}

/// Borwein's Algorithm 2 for `ζ(s)`, real `s > 0`, `s ≠ 1`:
/// `ζ(s) = η(s) / (1 − 2^{1−s})` with `η` from [`arb_eta_borwein`].
fn arb_zeta_borwein(s: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let wp = w.bits();
    let eta = arb_eta_borwein(s, w)?;
    // 1 − 2^{1−s} = −expm1((1 − s)·ln 2), with `1 − s` exact: next to the
    // pole the difference `1 − 2^{1−s}` cancels `log₂(1/|1 − s|)` bits, and
    // before 0.30 both were rounded (`ζ(1 + 10⁻⁴⁵)` was "a pole").
    let one_minus_s = BigFloat::from_i32(1, 64).sub(s, exact_bits(s, wp), RM);
    let denom_factor = one_minus_pow2(&one_minus_s, w);
    if denom_factor.is_zero() {
        return Err(SymplexError::Unevaluable {
            reason: "zeta(1) is a pole".into(),
        });
    }
    Ok(eta.div(&denom_factor, wp, RM))
}

/// `1 − 2^t` to `wp` bits relative, also for a tiny `t` (where the
/// difference cancels): `−Σ_{k≥1} u^k/k!` with `u = t·ln 2` for `|u| < 1/4`.
fn one_minus_pow2(t: &BigFloat, w: &mut Wp<'_>) -> BigFloat {
    let wp = w.bits();
    if t.is_zero() {
        return BigFloat::new(wp);
    }
    let u = w.cc.ln_2(wp + 8, RM).mul(t, wp + 8, RM);
    if u.exponent().is_some_and(|e| e > -2) {
        return BigFloat::from_i32(1, wp).sub(&u.exp(wp + 8, RM, w.cc), wp, RM);
    }
    let mut term = u.clone();
    let mut sum = u.clone();
    for k in 2..(wp + 16) {
        term = term
            .mul(&u, wp + 8, RM)
            .div(&BigFloat::from_i128(k as i128, wp + 8), wp + 8, RM);
        sum = sum.add(&term, wp + 8, RM);
        if negligible(&term, &sum, wp + 8) {
            break;
        }
    }
    round_to(sum.neg(), wp)
}

/// Borwein's accelerated alternating sum for the Dirichlet eta function
/// `η(s) = Σ_{k≥0} (−1)ᵏ/(k+1)ˢ`, real `s > 0` (P. Borwein, *An efficient
/// algorithm for the Riemann zeta function*, 1991, Algorithm 2; error
/// `≤ 3/(3+√8)ⁿ`, so `n ≈ 0.39·wp` terms):
///
/// ```text
/// d_k = n Σ_{i=0}^{k} (n+i−1)! 4ⁱ / ((n−i)! (2i)!)
/// η(s) = − 1/d_n · Σ_{k=0}^{n−1} (−1)ᵏ (d_k − d_n)/(k+1)ˢ
/// ```
fn arb_eta_borwein(s: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let wp = w.bits();
    // n such that (3+√8)^{-n} < 2^{-wp}:  n > wp·ln2/ln(3+√8) ≈ 0.393·wp
    let n = ((wp as f64) * std::f64::consts::LN_2 / (3.0 + 8f64.sqrt()).ln()).ceil() as usize + 4;

    // Exact integer coefficients d_k via rational arithmetic.
    let mut d: Vec<Ratio<BigInt>> = Vec::with_capacity(n + 1);
    let mut acc = Ratio::<BigInt>::zero();
    // term_i = n · (n+i−1)! · 4^i / ((n−i)! (2i)!)
    // term_0 = n · (n−1)!/n! = 1
    let mut term = Ratio::from_integer(BigInt::from(1));
    for i in 0..=n {
        if i > 0 {
            // term_i / term_{i−1} = (n+i−1)·4·(n−i+1) / ((2i−1)(2i))
            let numer = BigInt::from((n + i - 1) as u64)
                * BigInt::from(4u64)
                * BigInt::from((n - i + 1) as u64);
            let denom = BigInt::from((2 * i - 1) as u64) * BigInt::from((2 * i) as u64);
            term *= Ratio::new(numer, denom);
        }
        acc += &term;
        d.push(acc.clone());
    }
    let d_n = ratio_to_bigfloat(&d[n], wp);

    // S = Σ_{k=0}^{n−1} (−1)^k (d_k − d_n) / (k+1)^s
    let mut sum = BigFloat::new(wp);
    let s_is_int = s.is_int();
    let s_int = if s_is_int {
        f64_of(s)?.round() as usize
    } else {
        0
    };
    // A non-integer power `m^s` costs a logarithm and an exponential, and
    // there are `0.39·wp` of them; `m^s = p^s·(m/p)^s` for the smallest
    // prime factor `p` of `m` leaves them to the primes (a seventh of
    // them at 4,000 bits, where `ζ` took seconds), each composite a product
    // of at most `log₂ m` roundings.
    let powers: Vec<BigFloat> = if s_is_int {
        Vec::new()
    } else {
        let mut spf = vec![0usize; n + 1];
        let mut table: Vec<BigFloat> = Vec::with_capacity(n + 1);
        table.push(BigFloat::new(wp));
        for m in 1..=n {
            if m >= 2 && spf[m] == 0 {
                for multiple in (m..=n).step_by(m) {
                    if spf[multiple] == 0 {
                        spf[multiple] = m;
                    }
                }
            }
            let v = if m == 1 {
                BigFloat::from_i32(1, wp)
            } else if spf[m] == m {
                bf_pow(&BigFloat::from_i128(m as i128, wp), s, &mut w.at(wp + 8))
            } else {
                table[spf[m]].mul(&table[m / spf[m]], wp + 8, RM)
            };
            table.push(v);
        }
        table
    };
    for k in 0..n {
        let coeff = ratio_to_bigfloat(&(&d[k] - &d[n]), wp);
        let kp1 = BigFloat::from_i128(k as i128 + 1, wp);
        let kp1_pow_s = if s_is_int {
            kp1.powi(s_int, wp, RM)
        } else {
            powers[k + 1].clone()
        };
        let term = coeff.div(&kp1_pow_s, wp, RM);
        if k % 2 == 0 {
            sum = sum.add(&term, wp, RM);
        } else {
            sum = sum.sub(&term, wp, RM);
        }
    }

    Ok(sum.div(&d_n, wp, RM).neg())
}

// ═══════════════════════════════════════════════════════════════════════════
// Polygamma
// ═══════════════════════════════════════════════════════════════════════════

/// `ψ⁽ⁿ⁾(x)` for integer `n ≥ 1` and real `x` (not a non-positive integer).
///
/// 1. Recurrence `ψ⁽ⁿ⁾(x) = ψ⁽ⁿ⁾(x+N) − (−1)ⁿ n! Σ_{k<N} 1/(x+k)^{n+1}`
///    shifts the argument to `x + N ≥ max(wp/3, n) + 10`.
/// 2. Asymptotic series (DLMF 5.15.8):
///    `ψ⁽ⁿ⁾(x) ~ (−1)^{n+1} [ (n−1)!/xⁿ + n!/(2x^{n+1}) + Σ_{k≥1} B₂ₖ (2k+n−1)!/((2k)! x^{2k+n}) ]`
///    with exact Bernoulli numbers, truncated when the terms fall below
///    the working precision (or start to diverge).
///
/// The shifted arguments `x + k` are exact, so a pole is only an integer
/// `x ≤ 0`; left of 0 the terms change sign and cancel near the zeros of
/// `ψ⁽ⁿ⁾` (for even `n`): the loss, the largest term over the value, is
/// measured and the evaluation repeated with that many more bits, up to
/// [`cancellation_cap`].  (Before 0.29 `x + k` was rounded at every step
/// and an `x + k` below `2^{−wp/2}` was taken for a pole:
/// `polygamma(1, −3 + √2·10⁻³⁰)` was refused at 16 digits.)
fn arb_polygamma(
    n: u32,
    x: &BigFloat,
    prec: usize,
    _cc: &mut Consts,
) -> Result<BigFloat, SymplexError> {
    if x.is_nan() || x.is_inf_pos() || x.is_inf_neg() {
        return Err(SymplexError::Unevaluable {
            reason: "polygamma of special float value".into(),
        });
    }
    let cap = cancellation_cap(prec);
    let mut extra = 32;
    loop {
        let (value, largest) = polygamma_wp(n, x, prec + extra)?;
        let lost = match (largest.exponent(), value.exponent()) {
            _ if value.is_zero() => prec + extra,
            (Some(l), Some(v)) => usize::try_from(i64::from(l) - i64::from(v)).unwrap_or(0),
            _ => 0,
        };
        if lost + 16 <= extra {
            return Ok(round_to(value, prec));
        }
        if extra >= cap {
            return Err(special_exhausted(prec));
        }
        extra = (lost + 32).max(2 * extra).min(cap);
    }
}

/// `q·r` for a reduced fraction `q` and an integer `r`: the reduced fraction
/// `Ratio`'s product gives, with gcds against the denominator of `q` only
/// (of `r` modulo it) — not a binary gcd of `r` with 1, which takes a step
/// per bit of `r` (see [`factorial_exact`]).
fn times_integer(q: &Ratio<BigInt>, r: &BigInt) -> Ratio<BigInt> {
    use num_integer::Integer;
    use num_traits::One;
    let mut num = r.clone();
    let mut den = q.denom().clone();
    loop {
        let g = den.gcd(&(&num % &den));
        if g.is_one() || g.is_zero() {
            break;
        }
        num /= &g;
        den /= &g;
    }
    Ratio::new_raw(q.numer() * num, den)
}

/// `ψ⁽ⁿ⁾(x)` at working precision `wp` (see [`arb_polygamma`]) and the
/// magnitude of the largest term summed.
fn polygamma_wp(n: u32, x: &BigFloat, wp: usize) -> Result<(BigFloat, BigFloat), SymplexError> {
    let exact = exact_bits(x, wp);
    let n_us = n as usize;
    let one = BigFloat::from_i32(1, wp);

    // n! and (n−1)! as BigFloats (exact integers first: see `factorial_exact`).
    let nm1_fact = factorial_exact(n_us.saturating_sub(1));
    let n_fact = Ratio::from_integer(nm1_fact.numer() * BigInt::from(n_us as u64));
    let n_fact_bf = ratio_to_bigfloat(&n_fact, wp);
    let nm1_fact_bf = ratio_to_bigfloat(&nm1_fact, wp);
    // (−1)^{n+1}
    let outer_sign_neg = n_us.is_multiple_of(2);

    // ── Step 1: shift x upward ──
    let threshold_val = ((wp / 3).max(n_us) + 10) as i128;
    let threshold = BigFloat::from_i128(threshold_val, wp);
    // Far left of 0 the shift would take `|x|` steps: refused up front (a
    // refusal, not a failure — before 0.30 a million steps were taken and
    // then `ComputationFailed` raised, for `polygamma(100, erfi(−5))`).
    let max_steps = 10 * wp + 1_000_000;
    if bf_lt(x, &BigFloat::from_i128(-(max_steps as i128), wp)) {
        return Err(SymplexError::NotImplemented(
            "polygamma of an argument far left of 0 (the argument shift would take more \
             than a million steps)"
                .into(),
        ));
    }
    let mut xw = x.clone();
    let mut shift_sum = BigFloat::new(wp); // Σ 1/(x+k)^{n+1}
    let mut largest_inv = BigFloat::new(wp);
    let mut steps: usize = 0;
    while xw.sub(&threshold, wp, RM).is_negative() {
        if xw.is_zero() {
            return Err(SymplexError::Unevaluable {
                reason: "polygamma at non-positive integer pole".into(),
            });
        }
        let p = xw.powi(n_us + 1, wp, RM);
        let inv = one.div(&p, wp, RM);
        if inv.abs() > largest_inv {
            largest_inv = inv.abs();
        }
        shift_sum = shift_sum.add(&inv, wp, RM);
        xw = xw.add(&one, exact, RM);
        steps += 1;
        if steps > max_steps {
            return Err(SymplexError::NotImplemented(
                "polygamma: the argument shift did not terminate".into(),
            ));
        }
    }
    // ψ⁽ⁿ⁾(x) = ψ⁽ⁿ⁾(x+N) + (−1)^{n+1} n! Σ 1/(x+k)^{n+1}
    let shift_term = n_fact_bf.mul(&shift_sum, wp, RM);

    // ── Step 2: asymptotic series at xw ──
    // bracket = (n−1)!/x^n + n!/(2 x^{n+1}) + Σ B_{2k} (2k+n−1)!/((2k)! x^{2k+n})
    let x_pow_n = xw.powi(n_us, wp, RM);
    let x_pow_np1 = x_pow_n.mul(&xw, wp, RM);
    let two = BigFloat::from_i32(2, wp);
    let mut bracket = nm1_fact_bf.div(&x_pow_n, wp, RM);
    bracket = bracket.add(&n_fact_bf.div(&two.mul(&x_pow_np1, wp, RM), wp, RM), wp, RM);

    let x2 = xw.mul(&xw, wp, RM);
    let mut x_pow = x_pow_n.mul(&x2, wp, RM); // x^{n+2} for k = 1
    // (2k+n−1)!/(2k)! = (2k+1)⋯(2k+n−1), an integer: start k=1 → (n+1)!/2!
    // (exact integer steps, the same values as the `Ratio` steps before
    // 0.34 without their gcds).
    let mut ratio_fact = n_fact.numer() * BigInt::from(n_us as u64 + 1) / BigInt::from(2);
    let mut prev_term_exp: Option<i32> = None;
    let max_terms = wp / 2 + 20;
    for k in 1..=max_terms {
        if k > 1 {
            // multiply by (2k+n−2)(2k+n−1) / ((2k−1)(2k))
            let a =
                BigInt::from((2 * k + n_us - 2) as u64) * BigInt::from((2 * k + n_us - 1) as u64);
            let b = BigInt::from((2 * k - 1) as u64) * BigInt::from((2 * k) as u64);
            ratio_fact = ratio_fact * a / b;
            x_pow = x_pow.mul(&x2, wp, RM);
        }
        let b2k = bernoulli::even(k);
        let coeff = ratio_to_bigfloat(&times_integer(&b2k, &ratio_fact), wp);
        let term = coeff.div(&x_pow, wp, RM);
        if let Some(t_exp) = term.exponent() {
            if let Some(prev) = prev_term_exp
                && t_exp > prev + 10
            {
                break; // diverging
            }
            prev_term_exp = Some(t_exp);
            if let Some(b_exp) = bracket.exponent()
                && (b_exp as i64 - t_exp as i64) > wp as i64
            {
                bracket = bracket.add(&term, wp, RM);
                break;
            }
        }
        bracket = bracket.add(&term, wp, RM);
    }

    let total = bracket.add(&shift_term, wp, RM);
    let result = if outer_sign_neg { total.neg() } else { total };
    let largest_shift = n_fact_bf.mul(&largest_inv, wp, RM);
    let largest = if largest_shift > bracket.abs() {
        largest_shift
    } else {
        bracket.abs()
    };
    Ok((result, largest))
}

// ═══════════════════════════════════════════════════════════════════════════
// Modified Bessel functions
// ═══════════════════════════════════════════════════════════════════════════

/// Ascending series for `I_ν(x)` at working precision `wp`:
/// `I_ν(x) = (x/2)^ν Σ_{k≥0} (x²/4)^k / (k! Γ(ν+k+1))`.
///
/// All terms are positive for `x > 0`, so no cancellation occurs; the
/// number of terms is `O(|x| + wp)`.
fn bessel_i_series(
    order: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let wp = w.bits();
    let two = BigFloat::from_i32(2, wp);
    let x_half = x.div(&two, wp, RM);
    let x_half_sq = x_half.mul(&x_half, wp, RM);
    let order_f64 = f64_of(order)?;
    let order_int = order_f64.round() as i64;
    // Exactly (see `arb_bessel_j`).
    let is_int_order = order.is_int();

    if x.is_zero() {
        return if is_int_order && order_int == 0 {
            Ok(BigFloat::from_i32(1, wp))
        } else if order_f64 > 0.0 {
            Ok(BigFloat::new(wp))
        } else {
            Err(SymplexError::Unevaluable {
                reason: "BesselI at x=0 with negative order".into(),
            })
        };
    }

    if !is_int_order && x.is_negative() {
        return Err(SymplexError::Unevaluable {
            reason: "BesselI of negative argument with non-integer order is complex".into(),
        });
    }
    let one = BigFloat::from_i32(1, wp);
    let x_f64 = f64_of(x)?.abs();
    // A large order (I₋ₙ = Iₙ): the factor (x/2)^ν/Γ(ν+1) at once, its
    // exponent to an absolute 2^(−wp) (`bessel_order`), the series from 1 —
    // refused when it would be long.  Before 0.34 an order beyond 2⁶³
    // saturated an `i64`, and `besseli(10¹², 10¹¹)` summed for ever.
    let large = order_f64.abs() >= bessel_order::LARGE_ORDER;
    let (prefix, eff_order, mut term) = if large {
        let eff_order = if is_int_order {
            order.abs()
        } else {
            order.clone()
        };
        let eff_f64 = f64_of(&eff_order)?;
        let bounds = bessel_order::series_bounds(eff_f64, x_f64, false);
        // Below the range: 0, which the accuracy layer holds as the
        // underflow of a nonzero `I_ν` (`underflowed`), as before 0.34.
        let Some(p) =
            bessel_order::power_over_gamma(&eff_order, &x_half.abs(), bounds, "besseli", w)?
        else {
            return Ok(BigFloat::new(wp));
        };
        bessel_order::refuse_long_i_series(eff_f64, x_f64)?;
        // I_n(−x) = (−1)ⁿ I_n(x).
        let p = if x.is_negative() && bessel_order::is_odd(order) {
            p.neg()
        } else {
            p
        };
        (p, eff_order, one.clone())
    } else {
        // (x/2)^ν
        let prefix = if is_int_order && order_int >= 0 {
            x_half.powi(order_int as usize, wp, RM)
        } else if is_int_order {
            // Negative integer order: I_{-n} = I_n.
            x_half.powi((-order_int) as usize, wp, RM)
        } else {
            let ln_xh = x_half.ln(wp, RM, w.cc);
            order.mul(&ln_xh, wp, RM).exp(wp, RM, w.cc)
        };
        let eff_order = if is_int_order && order_int < 0 {
            BigFloat::from_i128((-order_int) as i128, wp)
        } else {
            order.clone()
        };
        // term_0 = 1/Γ(ν+1)
        let gamma_nu1 = arb_gamma_real(&eff_order.add(&one, wp, RM), w)?;
        let term = one.div(&gamma_nu1, wp, RM);
        (prefix, eff_order, term)
    };

    // term_{k+1} = term_k · (x²/4) / ((k+1)(ν+k+1))
    let mut sum = term.clone();
    let max_terms = (x_f64 * 2.0) as usize + wp + 40;
    for k in 1..=max_terms {
        let k_bf = BigFloat::from_i128(k as i128, wp);
        let nu_plus_k = eff_order.add(&k_bf, wp, RM);
        let denom = k_bf.mul(&nu_plus_k, wp, RM);
        term = term.mul(&x_half_sq, wp, RM).div(&denom, wp, RM);
        sum = sum.add(&term, wp, RM);
        if let (Some(t_exp), Some(s_exp)) = (term.exponent(), sum.exponent())
            && (s_exp as i64 - t_exp as i64) > wp as i64
        {
            break;
        }
    }
    let r = prefix.mul(&sum, wp, RM);
    if large && (r.is_zero() || r.is_inf()) {
        return Err(SymplexError::Unevaluable {
            reason: "besseli of large order: the value leaves the arbitrary-precision exponent \
                     range"
                .into(),
        });
    }
    Ok(r)
}

/// Asymptotic expansion of `I_ν(x)` for large `x > 0` (DLMF 10.40.1):
///
/// ```text
/// I_ν(x) ~ e^x/√(2πx) · Σ_k (−1)^k a_k(ν)/x^k,
/// a_k(ν) = (4ν² − 1²)(4ν² − 3²)⋯(4ν² − (2k−1)²) / (k!·8^k),
/// ```
///
/// the coefficients of [`hankel_pq`].  The part it leaves out is `~e^{−2x}`
/// relative to the value, below `2^{−wp}` from [`hankel_threshold`] on;
/// the sum stops at a term below `2^{−wp}` relative (or ends, for a
/// half-integer order).  `None` when the terms start growing first (an
/// order large next to `x`).
fn bessel_i_asymptotic(order: &BigFloat, x: &BigFloat, w: &mut Wp<'_>) -> Option<BigFloat> {
    let wp = w.bits();
    let one = BigFloat::from_i32(1, wp);
    let eight = BigFloat::from_i32(8, wp);
    let four_nu_sq = BigFloat::from_i32(4, wp).mul(&order.mul(order, wp, RM), wp, RM);
    let inv_x = one.div(x, wp, RM);
    let mut a = one.clone(); // a_k/x^k
    let mut sum = one.clone();
    let mut prev = one.clone();
    let mut converged = false;
    for k in 1..(4 * wp + 100) {
        let two_km1 = BigFloat::from_i128(2 * k as i128 - 1, wp);
        let numer = four_nu_sq.sub(&two_km1.mul(&two_km1, wp, RM), wp, RM);
        let denom = eight.mul(&BigFloat::from_i128(k as i128, wp), wp, RM);
        a = a
            .mul(&numer, wp, RM)
            .div(&denom, wp, RM)
            .mul(&inv_x, wp, RM);
        if a.is_zero() {
            converged = true;
            break;
        }
        // A growing term, or one beyond the leading 1: an alternating sum
        // that would cancel bits the working precision does not carry.
        let size = a.abs();
        if size.cmp(&prev).is_some_and(|c| c > 0) || size.exponent().is_some_and(|e| e > 0) {
            break;
        }
        sum = if k % 2 == 1 {
            sum.sub(&a, wp, RM)
        } else {
            sum.add(&a, wp, RM)
        };
        if let (Some(t), Some(s)) = (a.exponent(), sum.exponent())
            && i64::from(s) - i64::from(t) > wp as i64
        {
            converged = true;
            break;
        }
        prev = size;
    }
    if !converged {
        return None;
    }
    let two_pi_x =
        w.cc.pi(wp, RM)
            .mul(&BigFloat::from_i32(2, wp), wp, RM)
            .mul(x, wp, RM);
    let amplitude = two_pi_x.sqrt(wp, RM).reciprocal(wp, RM);
    Some(
        x.exp(wp, RM, w.cc)
            .mul(&amplitude, wp, RM)
            .mul(&sum, wp, RM),
    )
}

/// Modified Bessel function of the first kind `I_ν(x)` (real `ν`, `x`).
///
/// * `|x|` from [`hankel_threshold`] on: the asymptotic expansion
///   ([`bessel_i_asymptotic`]), with `I_n(−x) = (−1)ⁿ I_n(x)`;
/// * otherwise, or where the expansion does not converge, the ascending
///   series ([`bessel_i_series`], about `|x| + wp` terms), refused beyond
///   [`cancellation_cap`] terms.
///
/// Before 0.30 the series served every `x`: `besseli(5/2, 5.2·10⁷⁰)`
/// printed `2.18·10⁴⁴⁸⁵⁷` (truly beyond the exponent range),
/// `besseli(2, 10¹⁶)` and `besseli(0, 32/15·10⁸)` hung.
fn arb_bessel_i(order: &BigFloat, x: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let classic = arb_bessel_i_classic(order, x, w);
    debye_fallback(bessel_debye::Kind::I, order, x, w, classic)
}

/// [`arb_bessel_i`] by the asymptotic expansion and the ascending series.
fn arb_bessel_i_classic(
    order: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    let wp = prec + 32;
    if x.is_inf() {
        return Err(special_exhausted(prec));
    }
    let x_abs = f64_of(x)?.abs();
    let order_f64 = f64_of(order)?;
    if x_abs >= hankel_threshold(wp, order_f64.abs()) {
        if x.is_negative() {
            if !order.is_int() {
                return Err(SymplexError::Unevaluable {
                    reason: "BesselI of negative argument with non-integer order is complex".into(),
                });
            }
            let v = arb_bessel_i(order, &x.abs(), w)?;
            return Ok(if bessel_order::is_odd(order) {
                v.neg()
            } else {
                v
            });
        }
        if let Some(v) = bessel_i_asymptotic(order, x, &mut w.at(wp)) {
            return Ok(round_to(v, prec));
        }
        if !x_abs.is_finite() || x_abs > cancellation_cap(prec) as f64 {
            return Err(special_exhausted(prec));
        }
    }
    let r = bessel_i_series(order, x, &mut w.at(wp))?;
    Ok(round_to(r, prec))
}

/// Asymptotic expansion for `K_ν(x)`, large `x > 0` (DLMF 10.40.2):
/// `K_ν(x) ~ √(π/(2x)) e^{−x} Σ_k a_k(ν)/x^k`,
/// `a_k(ν) = ∏_{j=1}^{k} (4ν² − (2j−1)²) / (k! 8^k)`.
///
/// While `(2k−1)² < 4ν²` the terms are positive and may grow (an order
/// large next to `x`); past that they alternate, and the first one that
/// grows marks the divergence of the expansion.  `None` when the terms do
/// not fall below `2^{−wp}` relative before that (the caller uses the
/// series); a half-integer order ends the expansion exactly.  Before 0.30
/// the first growing term stopped the sum, whatever the order, and the
/// partial sum was returned: `besselk(20, 60)` was `6.14·10⁻²⁷` (truly
/// `3.75·10⁻²⁶`), `besselk(50, 100)` `6.29·10⁻⁴⁴` (truly
/// `9.27·10⁻⁴⁰`), both certified.
fn bessel_k_asymptotic(order: &BigFloat, x: &BigFloat, w: &mut Wp<'_>) -> Option<BigFloat> {
    let wp = w.bits();
    let one = BigFloat::from_i32(1, wp);
    let four = BigFloat::from_i32(4, wp);
    let eight = BigFloat::from_i32(8, wp);
    let four_nu_sq = four.mul(&order.mul(order, wp, RM), wp, RM);
    let inv_x = one.div(x, wp, RM);
    let mut a_k = one.clone();
    let mut sum = one.clone();
    let mut prev: Option<BigFloat> = None;
    let mut converged = false;
    let max_terms = wp * 2 + 100;
    for k in 1..max_terms {
        let two_km1 = BigFloat::from_i128(2 * k as i128 - 1, wp);
        let two_km1_sq = two_km1.mul(&two_km1, wp, RM);
        let numer = four_nu_sq.sub(&two_km1_sq, wp, RM);
        let denom = eight.mul(&BigFloat::from_i128(k as i128, wp), wp, RM);
        a_k = a_k
            .mul(&numer, wp, RM)
            .div(&denom, wp, RM)
            .mul(&inv_x, wp, RM);
        if a_k.is_zero() {
            converged = true;
            break;
        }
        let alternating = two_km1_sq.cmp(&four_nu_sq).is_some_and(|c| c > 0);
        if alternating
            && let Some(ref p) = prev
            && a_k.abs().cmp(p).is_some_and(|c| c > 0)
        {
            break;
        }
        sum = sum.add(&a_k, wp, RM);
        if let (Some(t_exp), Some(s_exp)) = (a_k.exponent(), sum.exponent())
            && (s_exp as i64 - t_exp as i64) > wp as i64
        {
            converged = true;
            break;
        }
        prev = Some(a_k.abs());
    }
    if !converged {
        return None;
    }
    let pi = w.cc.pi(wp, RM).clone();
    let two = BigFloat::from_i32(2, wp);
    let amp = pi.div(&two.mul(x, wp, RM), wp, RM).sqrt(wp, RM);
    let e_neg_x = x.neg().exp(wp, RM, w.cc);
    Some(amp.mul(&e_neg_x, wp, RM).mul(&sum, wp, RM))
}

/// Modified Bessel function of the second kind `K_ν(x)`, `x > 0`.
///
/// * Large `x` (`2x > wp·ln 2`): asymptotic expansion.
/// * Integer order `n` (A&S 9.6.11):
///   `K_n(x) = ½(x/2)^{−n} Σ_{k<n} (n−k−1)!/k! (−x²/4)^k + (−1)^{n+1} ln(x/2) I_n(x)
///           + (−1)^n ½ (x/2)^n Σ_{k≥0} [ψ(k+1) + ψ(n+k+1)] (x²/4)^k/(k!(n+k)!)`
///   with `ψ(m+1) = −γ + H_m`.
/// * Non-integer order: `K_ν = (π/2)(I_{−ν} − I_ν)/sin(νπ)`.
///
/// Both series formulas suffer `e^{x}`-scale cancellation (the result is
/// `~e^{−x}`), which is absorbed by extra guard bits.
fn arb_bessel_k(order: &BigFloat, x: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let classic = arb_bessel_k_classic(order, x, w);
    debye_fallback(bessel_debye::Kind::K, order, x, w, classic)
}

/// [`arb_bessel_k`] by the asymptotic expansion and the series.
fn arb_bessel_k_classic(
    order: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    if x.is_zero() || x.is_negative() || x.is_nan() {
        return Err(SymplexError::Unevaluable {
            reason: "BesselK undefined at x ≤ 0".into(),
        });
    }
    let base_wp = prec + 32;
    let x_f64 = f64_of(x)?;
    let order_f64 = f64_of(order)?;
    let order_int = order_f64.round() as i64;
    // Exactly (see `arb_bessel_j`); a non-integer order near an integer is
    // paid for with precision below.
    let is_int_order = order.is_int();
    // |n| exactly (before 0.34 through a saturating `i64`).
    let order_abs = if is_int_order && order_f64.abs() < bessel_order::LARGE_ORDER {
        BigFloat::from_i128(order_int.unsigned_abs() as i128, base_wp)
    } else {
        order.abs()
    };

    // Large x: asymptotic (relative truncation error ~ e^{-2x}).
    if 2.0 * x_f64 > (base_wp as f64) * std::f64::consts::LN_2
        && let Some(r) = bessel_k_asymptotic(&order_abs, x, &mut w.at(base_wp))
    {
        return Ok(round_to(r, prec));
    }

    // Series: guard against e^{x} cancellation (two exponentially large
    // terms of opposite sign nearly cancel).
    let wp = base_wp + 2 * series_guard_bits(x_f64, prec)?;
    let mut xw = x.clone();
    let _ = xw.set_precision(wp, RM);

    if !is_int_order {
        // K_ν = (π/2)(I_{−ν} − I_ν)/sin(νπ), with sin(νπ) = (−1)ⁿ sin(dπ),
        // ν = n + d exactly.  Near an integer order the difference cancels
        // log₂(1/|d|) bits (and more against e^{x}): the loss is measured
        // and the series repeated with that many more bits, up to
        // `cancellation_cap`.  (Before 0.29 an order within 10⁻¹² of an
        // integer was taken for it — `besselk(1 + 10⁻¹⁴, 1)` was wrong from
        // its 15th digit — and one a little further lost its bits unseen.)
        let exact = exact_bits(order, wp);
        let n = order.add(&BigFloat::from_f64(0.5, 64), exact, RM).floor();
        let d = order.sub(&n, exact, RM);
        let odd = !n
            .div(&BigFloat::from_i32(2, 64), exact, RM)
            .fract()
            .is_zero();
        let neg_order = order.neg();
        let cap = cancellation_cap(prec);
        let mut extra = wp - prec;
        loop {
            let wq = prec + extra;
            let mut xq = x.clone();
            let _ = xq.set_precision(wq, RM);
            let i_neg = bessel_i_series(&neg_order, &xq, &mut w.at(wq))?;
            let i_pos = bessel_i_series(order, &xq, &mut w.at(wq))?;
            let diff = i_neg.sub(&i_pos, wq, RM);
            let big = if i_neg.abs() > i_pos.abs() {
                i_neg.abs()
            } else {
                i_pos.abs()
            };
            let lost = match (big.exponent(), diff.exponent()) {
                _ if diff.is_zero() => wq,
                (Some(b), Some(e)) => usize::try_from(i64::from(b) - i64::from(e)).unwrap_or(0),
                _ => 0,
            };
            if lost + 16 <= extra {
                let pi = w.cc.pi(wq, RM).clone();
                let two = BigFloat::from_i32(2, wq);
                let mut sin_nu_pi = d.mul(&pi, wq, RM).sin(wq, RM, w.cc);
                if odd {
                    sin_nu_pi = sin_nu_pi.neg();
                }
                let r = pi
                    .div(&two, wq, RM)
                    .mul(&diff, wq, RM)
                    .div(&sin_nu_pi, wq, RM);
                return Ok(round_to(r, prec));
            }
            if extra >= cap {
                return Err(special_exhausted(prec));
            }
            extra = (lost + 48).max(2 * extra).min(cap);
        }
    }

    // A large order (K₋ₙ = Kₙ): the finite part of the series alone where
    // the rest is negligible (`bessel_order::neumann_leading`).  Before 0.34
    // the series took `n` steps in exact rationals (`besselk(5000, 1)` never
    // finished) and an order beyond 2⁶³ saturated.
    if order_f64.abs() >= bessel_order::LARGE_ORDER {
        return bessel_order::neumann_leading(bessel_order::Second::K, &order_abs, x, w)
            .unwrap_or_else(|| Err(large_order_refused("besselk", order_f64, x_f64)));
    }
    let n = order_int.unsigned_abs() as usize;
    let one = BigFloat::from_i32(1, wp);
    let two = BigFloat::from_i32(2, wp);
    let half = one.div(&two, wp, RM);
    let x_half = xw.div(&two, wp, RM);
    let x_half_sq = x_half.mul(&x_half, wp, RM);
    let ln_x_half = x_half.ln(wp, RM, w.cc);
    let gamma = arb_euler_gamma(&mut w.at(wp))?;
    let n_bf = BigFloat::from_i128(n as i128, wp);
    let i_n = bessel_i_series(&n_bf, &xw, &mut w.at(wp))?;

    // Finite sum: ½ (x/2)^{-n} Σ_{k=0}^{n-1} (n−k−1)!/k! (−x²/4)^k
    let mut finite = BigFloat::new(wp);
    if n > 0 {
        // f_k = (n−k−1)!/k! ; f_0 = (n−1)!
        let mut f = NeumannCoefficients::new(n);
        let mut pow = one.clone(); // (−x²/4)^k
        let neg_x_half_sq = x_half_sq.neg();
        for k in 0..n {
            if k > 0 {
                pow = pow.mul(&neg_x_half_sq, wp, RM);
            }
            let coeff = f.next(wp);
            finite = finite.add(&coeff.mul(&pow, wp, RM), wp, RM);
        }
        let x_half_pow_neg_n = one.div(&x_half.powi(n, wp, RM), wp, RM);
        finite = half.mul(&x_half_pow_neg_n, wp, RM).mul(&finite, wp, RM);
    }

    // Log term: (−1)^{n+1} ln(x/2) I_n(x)
    let log_term = ln_x_half.mul(&i_n, wp, RM);
    let log_term = if n.is_multiple_of(2) {
        log_term.neg()
    } else {
        log_term
    };

    // Infinite sum: (−1)^n ½ (x/2)^n Σ_k [ψ(k+1) + ψ(n+k+1)] (x²/4)^k/(k!(n+k)!)
    // ψ(m+1) = −γ + H_m.
    let mut h_k = BigFloat::new(wp); // H_0 = 0
    let mut h_nk = BigFloat::new(wp); // H_n
    for m in 1..=n {
        h_nk = h_nk.add(
            &one.div(&BigFloat::from_i128(m as i128, wp), wp, RM),
            wp,
            RM,
        );
    }
    // 1/(k!(n+k)!) start: 1/n!
    let n_fact = factorial_exact(n);
    let mut inv_fact = one.div(&ratio_to_bigfloat(&n_fact, wp), wp, RM);
    let mut pow = one.clone();
    let mut series = BigFloat::new(wp);
    let two_gamma = gamma.mul(&two, wp, RM);
    let max_terms = (x_f64 * 2.0) as usize + wp + 40;
    for k in 0..=max_terms {
        if k > 0 {
            let k_bf = BigFloat::from_i128(k as i128, wp);
            let nk_bf = BigFloat::from_i128((n + k) as i128, wp);
            h_k = h_k.add(&one.div(&k_bf, wp, RM), wp, RM);
            h_nk = h_nk.add(&one.div(&nk_bf, wp, RM), wp, RM);
            inv_fact = inv_fact.div(&k_bf.mul(&nk_bf, wp, RM), wp, RM);
            pow = pow.mul(&x_half_sq, wp, RM);
        }
        let psi_sum = h_k.add(&h_nk, wp, RM).sub(&two_gamma, wp, RM);
        let term = psi_sum.mul(&pow, wp, RM).mul(&inv_fact, wp, RM);
        series = series.add(&term, wp, RM);
        if k > 2
            && let (Some(t_exp), Some(s_exp)) = (term.exponent(), series.exponent())
            && (s_exp as i64 - t_exp as i64) > wp as i64
        {
            break;
        }
    }
    let series_term = half
        .mul(&x_half.powi(n, wp, RM), wp, RM)
        .mul(&series, wp, RM);
    let series_term = if n.is_multiple_of(2) {
        series_term
    } else {
        series_term.neg()
    };

    let r = finite.add(&log_term, wp, RM).add(&series_term, wp, RM);
    Ok(round_to(r, prec))
}

// ═══════════════════════════════════════════════════════════════════════════
// Orthogonal polynomials
// ═══════════════════════════════════════════════════════════════════════════

/// Families of classical orthogonal polynomials evaluated numerically.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OrthoPoly {
    Legendre,
    ChebyshevT,
    ChebyshevU,
    Hermite,
    Laguerre,
}

/// Evaluate `P_n(x)` (or `T_n`, `U_n`, `H_n`, `L_n`) for any non-negative
/// integer `n` via the three-term recurrence:
///
/// * Legendre: `(k+1) P_{k+1} = (2k+1) x P_k − k P_{k−1}`
/// * Chebyshev T/U: `T_{k+1} = 2x T_k − T_{k−1}` (`T_1 = x`, `U_1 = 2x`)
/// * Hermite (physicists'): `H_{k+1} = 2x H_k − 2k H_{k−1}`
/// * Laguerre: `(k+1) L_{k+1} = (2k+1−x) L_k − k L_{k−1}`
fn arb_orthopoly(kind: OrthoPoly, n: u64, x: &BigFloat, prec: usize) -> BigFloat {
    let wp = prec + 32 + (n as f64).log2().ceil().max(0.0) as usize;
    let one = BigFloat::from_i32(1, wp);
    let two = BigFloat::from_i32(2, wp);
    let mut xw = x.clone();
    let _ = xw.set_precision(wp, RM);
    if n == 0 {
        return BigFloat::from_i32(1, prec);
    }
    let mut p_prev = one.clone();
    let mut p_curr = match kind {
        OrthoPoly::Legendre | OrthoPoly::ChebyshevT => xw.clone(),
        OrthoPoly::ChebyshevU | OrthoPoly::Hermite => two.mul(&xw, wp, RM),
        OrthoPoly::Laguerre => one.sub(&xw, wp, RM),
    };
    for k in 1..n {
        let k_bf = BigFloat::from_i128(k as i128, wp);
        let kp1 = BigFloat::from_i128(k as i128 + 1, wp);
        let next = match kind {
            OrthoPoly::Legendre => {
                let two_k_plus_1 = BigFloat::from_i128(2 * k as i128 + 1, wp);
                let a = two_k_plus_1.mul(&xw, wp, RM).mul(&p_curr, wp, RM);
                let b = k_bf.mul(&p_prev, wp, RM);
                a.sub(&b, wp, RM).div(&kp1, wp, RM)
            }
            OrthoPoly::ChebyshevT | OrthoPoly::ChebyshevU => two
                .mul(&xw, wp, RM)
                .mul(&p_curr, wp, RM)
                .sub(&p_prev, wp, RM),
            OrthoPoly::Hermite => {
                let a = two.mul(&xw, wp, RM).mul(&p_curr, wp, RM);
                let b = two.mul(&k_bf, wp, RM).mul(&p_prev, wp, RM);
                a.sub(&b, wp, RM)
            }
            OrthoPoly::Laguerre => {
                let two_k_plus_1 = BigFloat::from_i128(2 * k as i128 + 1, wp);
                let a = two_k_plus_1.sub(&xw, wp, RM).mul(&p_curr, wp, RM);
                let b = k_bf.mul(&p_prev, wp, RM);
                a.sub(&b, wp, RM).div(&kp1, wp, RM)
            }
        };
        p_prev = p_curr;
        p_curr = next;
    }
    round_to(p_curr, prec)
}

// ═══════════════════════════════════════════════════════════════════════════
// More special functions (0.9)
// ═══════════════════════════════════════════════════════════════════════════
//
// All routines take real arguments and return real results at `prec` bits;
// complex arguments and points outside the real domain of a function are
// refused with `Unevaluable`.  Working precision is `prec + 32` plus
// problem-dependent guard bits wherever an alternating series or a
// difference of nearly equal quantities loses digits.

fn unevaluable(reason: impl Into<String>) -> SymplexError {
    SymplexError::Unevaluable {
        reason: reason.into(),
    }
}

/// A series exhausted its term budget without meeting the negligibility
/// test; the partial sum is *not* returned.
fn series_did_not_converge(operation: &'static str, max_terms: usize) -> SymplexError {
    SymplexError::ComputationFailed {
        operation,
        reason: format!("series did not converge in {max_terms} terms"),
    }
}

/// The cached values of `args`, all required to be real.
fn real_args(name: &str, args: &[ExprId], cache: Cache<'_>) -> Result<Vec<BigFloat>, SymplexError> {
    let mut out = Vec::with_capacity(args.len());
    for &a in args {
        let v = get_cached(cache, a)?;
        if !v.1.is_zero() {
            return Err(unevaluable(format!(
                "{name} of complex argument not yet supported in evalf"
            )));
        }
        out.push(v.0.clone());
    }
    Ok(out)
}

/// `|a| < 2^{-bits} |b|` — has the series term `a` become negligible
/// against the accumulated sum `b`?
fn negligible(a: &BigFloat, b: &BigFloat, bits: usize) -> bool {
    if a.is_zero() {
        return true;
    }
    match (a.exponent(), b.exponent()) {
        (Some(ae), Some(be)) => (be as i64 - ae as i64) > bits as i64,
        _ => false,
    }
}

/// `a > b`?
fn bf_gt(a: &BigFloat, b: &BigFloat) -> bool {
    a.cmp(b).is_some_and(|c| c > 0)
}

/// `a < b`?
fn bf_lt(a: &BigFloat, b: &BigFloat) -> bool {
    a.cmp(b).is_some_and(|c| c < 0)
}

/// A `BigFloat` that is an integer exactly, as an `i64` when it fits.
///
/// Before 0.30 a value within `10⁻¹²` of an integer counted (in `f64`),
/// and `polylog(2 + 10⁻²⁰, 999/1000)` was `Li₂(999/1000)`, wrong from its
/// 20th digit at 30 and 60 digits (the degree and order callers already
/// asked for an exact integer as well).
fn bf_as_int(x: &BigFloat) -> Result<Option<i64>, SymplexError> {
    if !(x.is_zero() || x.is_int()) {
        return Ok(None);
    }
    let r = f64_of(x)?.round();
    Ok((r.abs() < 9.0e15).then_some(r as i64))
}

/// Numeric value of the library function `f` on `args` (of the arity `f`
/// declares) inside `eval_node`.
///
/// Exhaustive over [`LibFn`].  The integer sequences and combinatorial
/// counts have no arbitrary-precision routine: `eval` folds them exactly
/// for integer arguments before `evalf` runs, and anything left is an
/// error rather than a guess.
fn eval_lib_fn(
    f: LibFn,
    args: &[ExprId],
    cache: Cache<'_>,
    w: &mut Wp<'_>,
) -> Result<Complex, SymplexError> {
    let prec = w.bits();
    match f {
        // The orthogonal polynomials of a real non-integer degree
        // (`hypergeom.rs`; refused before 0.54).
        f if hypergeom::takes(f, args, cache) => hypergeom::eval(f, args, cache, w),
        LibFn::BesselJ => eval_bessel(arb_bessel_j, "BesselJ", args, cache, w),
        LibFn::BesselY => eval_bessel(arb_bessel_y, "BesselY", args, cache, w),
        LibFn::BesselI => eval_bessel(arb_bessel_i, "BesselI", args, cache, w),
        LibFn::BesselK => eval_bessel(arb_bessel_k, "BesselK", args, cache, w),
        LibFn::Legendre => eval_orthopoly(OrthoPoly::Legendre, f, args, cache, w),
        LibFn::ChebyshevT => eval_orthopoly(OrthoPoly::ChebyshevT, f, args, cache, w),
        LibFn::ChebyshevU => eval_orthopoly(OrthoPoly::ChebyshevU, f, args, cache, w),
        LibFn::Hermite => eval_orthopoly(OrthoPoly::Hermite, f, args, cache, w),
        LibFn::Laguerre => eval_orthopoly(OrthoPoly::Laguerre, f, args, cache, w),
        // A non-real argument: `(1 − 2^(1−s))·ζ(s)` and `ψ(z + 1) + γ`
        // (`zeta.rs`; refused before 0.44).
        LibFn::DirichletEta
            if args.len() == 1 && cache.get(&args[0]).is_some_and(|v| !v.1.is_zero()) =>
        {
            zeta::eta_complex(get_cached(cache, args[0])?, w)
        }
        LibFn::Harmonic
            if args.len() == 1 && cache.get(&args[0]).is_some_and(|v| !v.1.is_zero()) =>
        {
            zeta::harmonic_complex(get_cached(cache, args[0])?, w)
        }
        LibFn::Erfi if args.len() == 1 && cache.get(&args[0]).is_some_and(|v| !v.1.is_zero()) => {
            cerf::erf_family(cerf::Kind::Erfi, get_cached(cache, args[0])?, w)
        }
        // `K`, `E`, `F`, `Π` with a non-real value or parameter (`elliptic.rs`;
        // refused before 0.54: `elliptic_e(2)`, `elliptic_k(1 + 2i)`; `Π`
        // before 0.55: `elliptic_pi(1/2, 2)`).
        LibFn::EllipticK | LibFn::EllipticE | LibFn::EllipticF | LibFn::EllipticPi
            if elliptic::takes(f, args, cache) =>
        {
            elliptic::eval(f, args, cache, w)
        }
        LibFn::Erfi
        | LibFn::ErfInv
        | LibFn::ErfcInv
        | LibFn::ExpInt
        | LibFn::Shi
        | LibFn::Chi
        | LibFn::FresnelS
        | LibFn::FresnelC
        | LibFn::LowerGamma
        | LibFn::UpperGamma
        | LibFn::DirichletEta
        | LibFn::AiryAi
        | LibFn::AiryBi
        | LibFn::AiryAiPrime
        | LibFn::AiryBiPrime
        | LibFn::EllipticK
        | LibFn::EllipticE
        | LibFn::EllipticF
        | LibFn::EllipticPi
        | LibFn::Gegenbauer
        | LibFn::Jacobi
        | LibFn::AssocLegendre
        | LibFn::AssocLaguerre
        | LibFn::BetaInc
        | LibFn::BetaIncRegularized => eval_special_09(f, args, cache, w),
        LibFn::PolyLog => eval_polylog(args, cache, w),
        LibFn::RisingFactorial
        | LibFn::FallingFactorial
        | LibFn::Harmonic
        | LibFn::Catalan
        | LibFn::Fibonacci
        | LibFn::Lucas => {
            let v = real_args(f.name(), args, cache)?;
            debug!(prec, name = f.name(), "evalf: Gamma quotient / digamma");
            let r = match f {
                LibFn::RisingFactorial => factorials::rising_factorial(&v[0], &v[1], w)?,
                LibFn::FallingFactorial => factorials::falling_factorial(&v[0], &v[1], w)?,
                LibFn::Catalan => factorials::catalan(&v[0], w)?,
                LibFn::Fibonacci => factorials::fibonacci(&v[0], false, w)?,
                LibFn::Lucas => factorials::fibonacci(&v[0], true, w)?,
                _ => factorials::harmonic(&v[0], w)?,
            };
            Ok((r, BigFloat::new(prec)))
        }
        LibFn::Bernoulli => {
            let v = real_args(f.name(), args, cache)?;
            debug!(prec, "evalf: bernoulli(s) = −s·ζ(1 − s)");
            let r = bernoulli::bernoulli_function(&v[0], w)?;
            Ok((r, BigFloat::new(prec)))
        }
        LibFn::EulerNumber => {
            let v = real_args(f.name(), args, cache)?;
            debug!(prec, "evalf: euler(s) by Hurwitz zeta functions");
            let r = bernoulli::euler_function(&v[0], w)?;
            Ok((r, BigFloat::new(prec)))
        }
        // A real non-integer argument (`factorials.rs`; refused before 0.54).
        LibFn::Factorial2
            if cache
                .get(&args[0])
                .is_some_and(|v| v.1.is_zero() && !v.0.is_int()) =>
        {
            let v = real_args(f.name(), args, cache)?;
            Ok((factorials::factorial2(&v[0], w)?, BigFloat::new(prec)))
        }
        LibFn::Factorial2
        | LibFn::Subfactorial
        | LibFn::Bell
        | LibFn::Stirling1
        | LibFn::Stirling2
        | LibFn::PartitionCount => Err(unevaluable(format!(
            "evalf has no numerical routine for '{f}' (eval folds it exactly at integer arguments)"
        ))),
        // Evaluated from its node, which carries the branch index literal.
        LibFn::LambertW => Err(unevaluable("lambertw is evaluated from its node")),
    }
}

/// `polylog(s, z)` for a real order: `arb_polylog` for a real `z ∈ [−1, 1]`,
/// `polylog::polylog_general` for a non-real `z` or `|z| > 1` (before 0.31
/// these were "not yet supported" for every order: `polylog(3/2, −3/16 −
/// 5/18·i)`, `polylog(2, 3)`, `polylog(1/3, −5)`).
fn eval_polylog(
    args: &[ExprId],
    cache: Cache<'_>,
    w: &mut Wp<'_>,
) -> Result<Complex, SymplexError> {
    let prec = w.bits();
    let s = get_cached(cache, args[0])?;
    let z = get_cached(cache, args[1])?;
    if !s.1.is_zero() {
        return polylog::polylog_complex_order(s, z, w);
    }
    if polylog::off_unit_interval(z) {
        return polylog::polylog_general(&s.0, z, w);
    }
    debug!(prec, "evalf: polylog on [-1, 1]");
    Ok((arb_polylog(&s.0, &z.0, w)?, BigFloat::new(prec)))
}

/// `W_k(z)` of a Lambert W node (`LambertW(x)`, `lambertw(x, k)`) over ℂ,
/// on every integer branch (`evalf::lambertw`).
fn eval_lambertw_node(
    arena: &Arena,
    node: &ExprNode,
    cache: Cache<'_>,
    w: &mut Wp<'_>,
) -> Result<Complex, SymplexError> {
    let prec = w.bits();
    let (x, k) = lambertw::lambert_parts(arena, node)
        .unwrap_or_else(|| Err(unevaluable("not a Lambert W node")))?;
    let z = get_cached(cache, x)?;
    debug!(
        prec,
        k, "evalf: LambertW via Halley iteration (mpmath's starting values)"
    );
    lambertw::lambert_w(z, k, w)
}

/// A Bessel function `(order, x)` of real arguments through `arb`.
fn eval_bessel(
    arb: fn(&BigFloat, &BigFloat, &mut Wp<'_>) -> Result<BigFloat, SymplexError>,
    what: &str,
    args: &[ExprId],
    cache: Cache<'_>,
    w: &mut Wp<'_>,
) -> Result<Complex, SymplexError> {
    let prec = w.bits();
    let order = get_cached(cache, args[0])?;
    let arg = get_cached(cache, args[1])?;
    if !order.1.is_zero() || !arg.1.is_zero() {
        return Err(SymplexError::Unevaluable {
            reason: "Bessel of complex argument not yet supported in evalf".into(),
        });
    }
    tracing::debug!(prec, what, "evalf: Bessel function via series/asymptotic");
    let result = arb(&order.0, &arg.0, w)?;
    Ok((result, BigFloat::new(prec)))
}

/// A classical orthogonal polynomial `(n, x)` of real arguments by
/// recurrence, for a non-negative integer degree.
fn eval_orthopoly(
    kind: OrthoPoly,
    f: LibFn,
    args: &[ExprId],
    cache: Cache<'_>,
    w: &mut Wp<'_>,
) -> Result<Complex, SymplexError> {
    let prec = w.bits();
    let n_val = get_cached(cache, args[0])?;
    let x_val = get_cached(cache, args[1])?;
    if !n_val.1.is_zero() || !x_val.1.is_zero() {
        return Err(SymplexError::Unevaluable {
            reason: "orthogonal polynomial of complex arguments not supported".into(),
        });
    }
    let n_f = f64_of(&n_val.0)?;
    let n_round = n_f.round();
    // An integer exactly: before 0.29 a degree within 10⁻¹² of one was taken
    // for it, and `legendre(3 + 10⁻¹³, 1/3)` printed P₃(1/3) (wrong from the
    // 14th digit).
    if !n_val.0.is_int() || !(0.0..=1.0e7).contains(&n_round) {
        return Err(SymplexError::Unevaluable {
            reason: format!("{f}: degree must be a non-negative integer"),
        });
    }
    tracing::debug!(
        prec,
        n = n_round,
        "evalf: orthogonal polynomial via recurrence"
    );
    let result = arb_orthopoly(kind, n_round as u64, &x_val.0, prec);
    Ok((result, BigFloat::new(prec)))
}

/// Dispatch for the 0.9 special functions inside `eval_node`.
///
/// Exhaustive over [`LibFn`]; the functions outside this family are an
/// error here (they are dispatched by [`eval_lib_fn`] and never arrive).
fn eval_special_09(
    f: LibFn,
    args: &[ExprId],
    cache: Cache<'_>,
    w: &mut Wp<'_>,
) -> Result<Complex, SymplexError> {
    let prec = w.bits();
    let name = f.name();
    let v = real_args(name, args, cache)?;
    let real = |r: BigFloat| Ok((r, BigFloat::new(prec)));
    debug!(prec, name, "evalf: special function (0.9)");
    match f {
        LibFn::Erfi => real(arb_erfi(&v[0], w)?),
        LibFn::ErfInv => real(arb_erfinv(&v[0], w)?),
        LibFn::ErfcInv => real(arb_erfcinv(&v[0], w)?),
        LibFn::ExpInt => real(arb_expint(&v[0], &v[1], w)?),
        LibFn::Shi => real(arb_shi_chi(&v[0], true, w)?),
        LibFn::Chi => {
            if v[0].is_zero() {
                return Err(unevaluable("Chi(0) is zoo"));
            }
            let chi = arb_shi_chi(&v[0].abs(), false, w)?;
            if v[0].is_negative() {
                // Chi(−x) = Chi(x) + iπ
                Ok((chi, w.cc.pi(prec, RM).clone()))
            } else {
                real(chi)
            }
        }
        LibFn::FresnelS => real(arb_fresnel(&v[0], true, w)?),
        LibFn::FresnelC => real(arb_fresnel(&v[0], false, w)?),
        LibFn::LowerGamma => real(arb_lowergamma(&v[0], &v[1], w)?),
        LibFn::UpperGamma => real(arb_uppergamma(&v[0], &v[1], w)?),
        LibFn::DirichletEta => real(arb_dirichlet_eta(&v[0], w)?),
        LibFn::AiryAi => real(arb_airy(&v[0], AiryKind::Ai, w)?),
        LibFn::AiryBi => real(arb_airy(&v[0], AiryKind::Bi, w)?),
        LibFn::AiryAiPrime => real(arb_airy(&v[0], AiryKind::AiPrime, w)?),
        LibFn::AiryBiPrime => real(arb_airy(&v[0], AiryKind::BiPrime, w)?),
        LibFn::EllipticK => real(arb_elliptic_k(&v[0], w)?),
        LibFn::EllipticE => real(arb_elliptic_e(&v[0], w)?),
        LibFn::EllipticF => real(arb_elliptic_f(&v[0], &v[1], w)?),
        LibFn::EllipticPi => real(arb_elliptic_pi(&v[0], &v[1], w)?),
        LibFn::Gegenbauer => real(arb_gegenbauer(&v[0], &v[1], &v[2], w)?),
        LibFn::Jacobi => real(arb_jacobi(&v[0], &v[1], &v[2], &v[3], w)?),
        LibFn::AssocLegendre => real(arb_assoc_legendre(&v[0], &v[1], &v[2], w)?),
        LibFn::AssocLaguerre => real(arb_assoc_laguerre(&v[0], &v[1], &v[2], w)?),
        LibFn::BetaInc | LibFn::BetaIncRegularized => real(arb_betainc(
            &v[0],
            &v[1],
            &v[2],
            &v[3],
            f == LibFn::BetaIncRegularized,
            w,
        )?),
        LibFn::Factorial2
        | LibFn::Subfactorial
        | LibFn::RisingFactorial
        | LibFn::FallingFactorial
        | LibFn::Fibonacci
        | LibFn::Lucas
        | LibFn::Bernoulli
        | LibFn::Harmonic
        | LibFn::Catalan
        | LibFn::Bell
        | LibFn::EulerNumber
        | LibFn::Stirling1
        | LibFn::Stirling2
        | LibFn::PartitionCount
        | LibFn::LambertW
        | LibFn::PolyLog
        | LibFn::BesselJ
        | LibFn::BesselY
        | LibFn::BesselI
        | LibFn::BesselK
        | LibFn::Legendre
        | LibFn::ChebyshevT
        | LibFn::ChebyshevU
        | LibFn::Hermite
        | LibFn::Laguerre => Err(unevaluable(format!("cannot evaluate function '{name}'"))),
    }
}

// ── Error-function family ──────────────────────────────────────────────────────

/// `erfi(x) = (2/√π) Σ x^{2n+1}/(n!(2n+1))` (all terms of one sign, so no
/// cancellation) for `x² ≤ wp·ln 2`, otherwise the asymptotic expansion
/// `erfi(x) ~ e^{x²}/(x√π) Σ (2k−1)!!/(2x²)^k` (optimal truncation error
/// `~e^{−x²}` relative to the leading term).
fn arb_erfi(x: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    if x.is_zero() {
        return Ok(BigFloat::new(prec));
    }
    if x.is_nan() || x.is_inf() {
        return Err(unevaluable("erfi of special float value"));
    }
    let wp = prec + 32;
    let x_f = f64_of(x)?;
    let ax = x_f.abs();
    let two = BigFloat::from_i32(2, wp);
    let sqrt_pi = w.cc.pi(wp, RM).clone().sqrt(wp, RM);
    let x_sq = x.mul(x, wp, RM);

    if ax * ax > (wp as f64) * std::f64::consts::LN_2 + 4.0 {
        let two_x_sq = x_sq.mul(&two, wp, RM);
        let mut sum = BigFloat::from_i32(1, wp);
        let mut term = BigFloat::from_i32(1, wp);
        let mut prev: Option<BigFloat> = None;
        for k in 1..(wp * 2 + 100) {
            let f = BigFloat::from_i128(2 * k as i128 - 1, wp);
            term = term.mul(&f, wp, RM).div(&two_x_sq, wp, RM);
            if let Some(ref p) = prev
                && bf_gt(&term, p)
            {
                break;
            }
            sum = sum.add(&term, wp, RM);
            if negligible(&term, &sum, wp) {
                break;
            }
            prev = Some(term.clone());
        }
        let e = x_sq.exp(wp, RM, w.cc);
        let denom = x.abs().mul(&sqrt_pi, wp, RM);
        let r = e.mul(&sum, wp, RM).div(&denom, wp, RM);
        let r = if x.is_negative() { r.neg() } else { r };
        return Ok(round_to(r, prec));
    }

    let mut prod = x.clone();
    let _ = prod.set_precision(wp, RM);
    let mut sum = prod.clone();
    let max_terms = (3.0 * ax * ax) as usize + wp + 60;
    let mut converged = false;
    for n in 1..=max_terms {
        let n_bf = BigFloat::from_i128(n as i128, wp);
        prod = prod.mul(&x_sq, wp, RM).div(&n_bf, wp, RM);
        let d = BigFloat::from_i128(2 * n as i128 + 1, wp);
        let term = prod.div(&d, wp, RM);
        sum = sum.add(&term, wp, RM);
        if negligible(&term, &sum, wp) {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(series_did_not_converge("erfi", max_terms));
    }
    let r = sum.mul(&two, wp, RM).div(&sqrt_pi, wp, RM);
    Ok(round_to(r, prec))
}

/// `erfc(x)` with *relative* precision `prec` for `x > 0` (absolute
/// precision, via `1 − erf`, for `x ≤ 1/2`, where nothing cancels).
///
/// * `x² log₂ e ≥ wp + 8`: the asymptotic expansion alone (relative error
///   `≈ e^{−x²}`) already delivers `wp` bits.
/// * `x ≥ 5/4`: `erfc(x) = Γ(½, x²)/√π` by Legendre's continued fraction
///   ([`uppergamma_cf`], in its regime `x² ≥ max(1, ½ + 1)`), which cancels
///   nothing.  Before 0.30 this range took `1 − erf(x)` with `x² log₂ e`
///   extra bits, a Taylor series of some `e·x²` terms at that precision
///   (`erfc(18.5)` at 300 digits: 11 ms, now 2 ms).
/// * otherwise `1 − erf(x)` with `x² log₂ e + 8` extra bits to survive the
///   cancellation (`erfc(x) ≈ e^{−x²}/(x√π)`).
fn arb_erfc(x: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    let wp = prec + 32;
    let x_f = f64_of(x)?;
    if x_f <= 0.5 {
        let erf = arb_erf(x, w)?;
        let one = BigFloat::from_i32(1, wp);
        return Ok(round_to(one.sub(&erf, wp, RM), prec));
    }
    let x_sq_bits = (x_f * x_f * std::f64::consts::LOG2_E).ceil() as usize;
    let mut xw = x.clone();
    if x_sq_bits >= wp + 8 {
        let _ = xw.set_precision(wp, RM);
        return Ok(round_to(erfc_asymptotic(&xw, &mut w.at(wp)), prec));
    }
    if x_f >= 1.25 {
        // x² to a relative 2^−(wp + log₂ x² + 8): e^{−x²} to 2^−(wp + 8).
        let p = wp + magnitude_bits(x_f * x_f) + 8;
        let _ = xw.set_precision(p, RM);
        let x_sq = xw.mul(&xw, p, RM);
        let half = BigFloat::from_f64(0.5, 64);
        let g = uppergamma_cf(&half, &x_sq, &mut w.at(p))?;
        let sqrt_pi = w.cc.pi(wp, RM).clone().sqrt(wp, RM);
        return Ok(round_to(g.div(&sqrt_pi, wp, RM), prec));
    }
    let wp2 = wp + x_sq_bits + 8;
    let _ = xw.set_precision(wp2, RM);
    let erf = arb_erf(&xw, &mut w.at(wp2))?;
    let one = BigFloat::from_i32(1, wp2);
    Ok(round_to(one.sub(&erf, wp2, RM), prec))
}

/// Halley iteration for `erf(x) = y` (`solve_erfc == false`) or
/// `erfc(x) = y` (`solve_erfc == true`) from the starting point `x0`,
/// safeguarded by the bracket `[lo, hi]` that contains the root (both
/// functions are monotone): a step that leaves the bracket, or does not at
/// least halve it, is replaced by bisection, and the bracket shrinks with
/// every evaluation.  An iteration that does not converge is refused.
///
/// With `g = f/f'`, the Halley step for both equations is
/// `x ← x − g / (1 + x g)` (since `f'' = −2x f'`).
///
/// Before 0.30 there was no bracket and no convergence check: from a poor
/// start the iteration drifted to where `erfc` saturates at 2 and returned
/// its 200th iterate, `erfcinv(941/2³⁴) = −10573.05…` (truly `3.84317…`),
/// certified.
#[allow(clippy::too_many_arguments)]
fn erf_inverse_halley(
    y: &BigFloat,
    x0: f64,
    lo: f64,
    hi: f64,
    solve_erfc: bool,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    let wp = prec + 32;
    let mut yw = y.clone();
    let _ = yw.set_precision(wp, RM);
    let half_sqrt_pi =
        w.cc.pi(wp, RM)
            .clone()
            .sqrt(wp, RM)
            .div(&BigFloat::from_i32(2, wp), wp, RM);
    let one = BigFloat::from_i32(1, wp);
    let two = BigFloat::from_i32(2, wp);
    let (mut lo, mut hi) = (bigfloat::from_f64(lo, wp), bigfloat::from_f64(hi, wp));
    let mut x = bigfloat::from_f64(x0.clamp(lo_f(&lo), hi_f(&hi)), wp);
    fn lo_f(v: &BigFloat) -> f64 {
        f64_of(v).unwrap_or(f64::NEG_INFINITY)
    }
    fn hi_f(v: &BigFloat) -> f64 {
        f64_of(v).unwrap_or(f64::INFINITY)
    }
    for _ in 0..(4 * wp + 200) {
        let f = if solve_erfc {
            arb_erfc(&x, &mut w.at(wp))?.sub(&yw, wp, RM)
        } else {
            arb_erf(&x, &mut w.at(wp))?.sub(&yw, wp, RM)
        };
        if f.is_zero() {
            return Ok(round_to(x, prec));
        }
        // erf increases, erfc decreases: the root is above x when
        // erf(x) < y or erfc(x) > y.
        let root_above = f.is_negative() != solve_erfc;
        if root_above {
            lo = x.clone();
        } else {
            hi = x.clone();
        }
        // f' = ±(2/√π) e^{−x²}  ⇒  g = f/f' = ±f (√π/2) e^{x²}
        let x_sq = x.mul(&x, wp, RM);
        let g = f
            .mul(&half_sqrt_pi, wp, RM)
            .mul(&x_sq.exp(wp, RM, w.cc), wp, RM);
        let g = if solve_erfc { g.neg() } else { g };
        let denom = one.add(&x.mul(&g, wp, RM), wp, RM);
        let delta = if denom.is_zero() {
            g.clone()
        } else {
            g.div(&denom, wp, RM)
        };
        // Converged to `prec + 8` bits (the working precision carries 32
        // guard bits, of which the last few are rounding noise in `f`, so
        // demanding all of them never ends: `erfinv(erf(−9))` at 60
        // digits stalled on a one-ulp bracket).
        let tol = prec + 8;
        let width = hi.sub(&lo, wp, RM);
        let candidate = x.sub(&delta, wp, RM);
        // A step below the tolerance is convergence, before the bracket is
        // consulted: below an ulp of x the candidate rounds onto x, which is
        // now an end of the bracket, and "leaving the bracket" sent a
        // converged iteration into bisection from its last bracket down to
        // the tolerance (0.29: some 260 halvings, `erfcinv(sin(1 + 10⁻¹⁵⁰)
        // − sin(1))` at 60 digits evaluated `erfc` 1,104 times).
        if negligible(&delta, &x, tol) {
            return Ok(round_to(candidate, prec));
        }
        let inside = bf_gt(&candidate, &lo) && bf_lt(&candidate, &hi);
        let halves = bf_lt(&delta.abs().mul(&two, wp, RM), &width);
        if inside && halves {
            x = candidate;
            if negligible(&delta, &x, tol) {
                return Ok(round_to(x, prec));
            }
        } else {
            let mid = lo.add(&hi, wp, RM).div(&two, wp, RM);
            let stuck = mid == lo || mid == hi;
            x = mid;
            if stuck || negligible(&width, &x, tol) {
                return Ok(round_to(x, prec));
            }
        }
    }
    Err(SymplexError::computation_failed(
        "evalf",
        "the inverse error function iteration did not converge",
    ))
}

/// `erfinv(y)` for `|y| < 1`: Winitzki's closed-form approximation refined
/// by `f64` Newton steps, then Halley iteration on `erf` (`|y| ≤ 1/2`) or
/// on `erfc(x) = 1 − |y|` (`|y| > 1/2`, where `erf` alone would lose the
/// digits of `1 − y`).
fn arb_erfinv(y: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    if y.is_zero() {
        return Ok(BigFloat::new(prec));
    }
    let wp = prec + 32;
    let one = BigFloat::from_i32(1, wp);
    let ay = y.abs();
    let one_minus_ay = one.sub(&ay, wp, RM);
    if one_minus_ay.is_zero() || one_minus_ay.is_negative() {
        return Err(unevaluable("erfinv: argument must satisfy |y| < 1"));
    }
    // A tiny argument: erfinv(y) = (√π/2)·y·(1 + πy²/12 + …), whose
    // correction is below the working precision once |y| < 2^(−wp/2).  (The
    // bracketed iteration starts from (−½, ½) and cannot bisect down to
    // `erfinv(sech(8!)) ≈ 3·10⁻¹⁷⁵¹¹`.)
    if let Some(e) = y.exponent()
        && i64::from(e) < -(wp as i64) / 2 - 4
    {
        let half_sqrt_pi =
            w.cc.pi(wp, RM)
                .clone()
                .sqrt(wp, RM)
                .div(&BigFloat::from_i32(2, wp), wp, RM);
        return Ok(round_to(y.mul(&half_sqrt_pi, wp, RM), prec));
    }
    let y_f = f64_of(y)?;
    let r = if y_f.abs() > 0.5 {
        arb_erfcinv_small(&one_minus_ay, w)?
    } else {
        // Winitzki (a = 0.147), |relative error| < 2·10⁻³.
        let a = 0.147_f64;
        let l = (1.0 - y_f * y_f).ln();
        let t = 2.0 / (std::f64::consts::PI * a) + l / 2.0;
        let mut x0 = ((t * t - l / a).sqrt() - t).sqrt().copysign(y_f);
        for _ in 0..3 {
            let f = erf_f64(x0) - y_f;
            x0 -= f * (std::f64::consts::PI.sqrt() / 2.0) * (x0 * x0).exp();
        }
        // |y| ≤ 1/2 < erf(1/2): the root lies in (−1/2, 1/2).
        erf_inverse_halley(y, x0, -0.5, 0.5, false, w)?
    };
    Ok(if y.is_negative() && !r.is_negative() && y_f.abs() > 0.5 {
        r.neg()
    } else {
        r
    })
}

/// `erfcinv(u)` for `0 < u ≤ 1/2`, Halley iteration on `erfc`.  The initial
/// guess inverts the leading asymptotic `erfc(x) ≈ e^{−x²}/(x√π)`, which
/// also covers `u` far below the `f64` range; the root lies in
/// `(0, max(√L, 1)]` with `L = −ln u` (`erfc(0) = 1 > u`, and
/// `erfc(x) < e^{−x²}/(x√π) ≤ u` there).  (0.29 refined the guess by `f64`
/// Newton steps on `1 − erf_f64(x)`, a 50-term Taylor sum whose truncation
/// near `x = 4` is `10⁻⁶`: the steps diverged.)
fn arb_erfcinv_small(u: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    let wp = prec + 32;
    // L = −ln u  (computed in BigFloat so tiny u is fine)
    let l = u.ln(wp, RM, w.cc).neg();
    let l_f = f64_of(&l)?;
    let mut x0 = l_f.max(0.5).sqrt();
    for _ in 0..4 {
        // x² = L − ln(x√π)
        let v = l_f - (x0 * std::f64::consts::PI.sqrt()).ln();
        x0 = v.max(0.25).sqrt();
    }
    let hi = l_f.max(0.0).sqrt().max(1.0) * (1.0 + f64::EPSILON * 16.0);
    erf_inverse_halley(u, x0, 0.0, hi, true, w)
}

/// `erfcinv(u)` for `0 < u < 2`: `erfcinv(u) = −erfcinv(2 − u)` maps to
/// `u ≤ 1`, then `erfinv(1 − u)` for `u ≥ 1/2` and the `erfc` Halley
/// iteration for smaller `u`.
fn arb_erfcinv(u: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    let wp = prec + 32;
    let one = BigFloat::from_i32(1, wp);
    let two = BigFloat::from_i32(2, wp);
    if u.is_zero() || u.is_negative() || !bf_lt(u, &two) {
        return Err(unevaluable("erfcinv: argument must satisfy 0 < u < 2"));
    }
    if bf_gt(u, &one) {
        let v = two.sub(u, wp, RM);
        return Ok(arb_erfcinv(&v, w)?.neg());
    }
    let u_f = f64_of(u)?;
    if u_f >= 0.5 {
        let y = one.sub(u, wp, RM);
        return arb_erfinv(&y, w);
    }
    arb_erfcinv_small(u, w)
}

// ── Incomplete gamma / exponential integrals ───────────────────────────────────

/// `v > 0`: astro-float's `is_positive` is the sign bit alone, true for
/// `+0` (`expint(1 + 10⁻²⁰⁰, ⅓)` rounds its order to 1 and `s = 1 − n` to
/// `+0`, which must take the integer route, not `Γ(0)`).
fn bf_strictly_positive(v: &BigFloat) -> bool {
    is_strictly_positive(v)
}

/// `⌈log₂ v⌉` for `v > 1`, `0` for `v ≤ 1`, 1100 for a non-finite `v`: the
/// guard bits a quantity of magnitude `v` needs for its *absolute* error to
/// stay at the unit of the working precision (an exponent `s ln x ≈ 10⁹`
/// needs 30 bits more than the relative precision of its exponential).
fn magnitude_bits(v: f64) -> usize {
    if !v.is_finite() {
        1100
    } else if v > 1.0 {
        v.log2().ceil() as usize
    } else {
        0
    }
}

/// `⌈log₂(1/s)⌉` for `0 < s < 1`, from the binary exponent (so also for an
/// `s` below the `f64` range); `0` for `s ≥ 1`.
fn inverse_bits(s: &BigFloat) -> usize {
    match s.exponent() {
        Some(e) if e < 1 => usize::try_from(1 - i64::from(e)).unwrap_or(0),
        _ => 0,
    }
}

/// Guard bits for `ln Γ(s)`, `s > 0`, whose magnitude is below
/// `s·|ln s| + s + 2` for `s ≥ 1` and `|ln s| + 2` below.
fn lngamma_bits(s: &BigFloat, s_f: f64) -> usize {
    if s_f >= 1.0 {
        magnitude_bits(s_f * s_f.ln() + s_f + 2.0)
    } else {
        magnitude_bits(inverse_bits(s) as f64 * std::f64::consts::LN_2 + 2.0)
    }
}

/// The most bits of cancellation a special function pays for with working
/// precision before it refuses: `4·wp + 1024` (a shape of `10⁻³⁰⁰` costs
/// about 1000, and is evaluated; one of `10⁻¹⁰⁰⁰` is refused at 16 and 50
/// digits — `ln Γ` by Stirling's series at 3500 bits takes seconds).
fn cancellation_cap(wp: usize) -> usize {
    wp.saturating_mul(4).saturating_add(1024)
}

/// The refusal of a special function whose cancellation exceeds
/// [`cancellation_cap`]: [`SymplexError::PrecisionExhausted`] at the digits
/// the working precision `prec` stands for (the inverse of
/// `working_precision`).  The routines below `arb_*` raise it at their own
/// working precision; [`requested_at`] restates it at the caller's.
fn special_exhausted(prec: usize) -> SymplexError {
    SymplexError::PrecisionExhausted {
        requested: u32::try_from(prec.saturating_sub(64) * 10 / 34).unwrap_or(u32::MAX),
        achieved: 0,
    }
}

/// A [`SymplexError::PrecisionExhausted`] from inside a special function,
/// restated at the precision `prec` its caller asked for (other errors
/// pass through).
fn requested_at(prec: usize) -> impl Fn(SymplexError) -> SymplexError {
    move |e| match e {
        SymplexError::PrecisionExhausted { .. } => special_exhausted(prec),
        other => other,
    }
}

/// `v` at precision `p` (a copy; the value is unchanged when `p` is larger).
fn at_precision(v: &BigFloat, p: usize) -> BigFloat {
    let mut w = v.clone();
    let _ = w.set_precision(p, RM);
    w
}

/// `x^s e^{−x}` for `x > 0`, as `exp(s ln x − x)` with the exponent formed
/// to an absolute error of `2^−wp` (the relative error of the result): it
/// carries the bits of its magnitude as guard.  0.28 multiplied `e^{−x}` by
/// `x^s`, two factors near the ends of the exponent range for `s ≈ x ≈ 10⁸`.
fn gamma_prefactor(s: &BigFloat, x: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let wp = w.bits();
    let s_f = f64_of(s)?.abs();
    let x_f = f64_of(x)?;
    // |ln x| from the binary exponent when x is outside the f64 range.
    let ln_x = x
        .exponent()
        .map_or(0.0, |e| (f64::from(e).abs() + 1.0) * std::f64::consts::LN_2);
    let wpe = wp + magnitude_bits(s_f * ln_x + x_f) + 8;
    let e = s.mul(&x.ln(wpe, RM, w.cc), wpe, RM).sub(x, wpe, RM);
    Ok(e.exp(wp, RM, w.cc))
}

/// Legendre's continued fraction for `Γ(s, x)`, `x > 0` (modified Lentz):
///
/// ```text
/// Γ(s, x) = e^{−x} x^s / (x + 1 − s − 1·(1−s)/(x + 3 − s − 2·(2−s)/(x + 5 − s − …)))
/// ```
///
/// Converges for every `x > 0` and every real `s`; it is the method of
/// choice for `x ≥ max(1, s + 1)`.  At `x = s + 1` it needs about
/// `27·s^{1/3}` steps to 266 bits (573 at `s = 10⁴`, 5779 at `s = 10⁷`) and a
/// few hundred from `x = s + 10√s` on, so the budget is
/// `40·wp + 2000 + 64·s^{1/3}` (0.28's `40·wp + 2000` fell short near
/// `s = 10⁸`), the last term at most `2·10⁶` (seconds; beyond, from
/// `s ≈ 3·10¹⁷`, the fraction is refused rather than run for minutes); the
/// fraction is evaluated with `log₂(budget) + 8` guard bits for the
/// roundings of its steps.
fn uppergamma_cf(s: &BigFloat, x: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let wp = w.bits();
    let s_f = f64_of(s)?;
    let max_iter = (40 * wp + 2000).saturating_add((64.0 * s_f.abs().cbrt()).min(2e6) as usize);
    let wp = wp + magnitude_bits(max_iter as f64) + 8;
    let (s, x) = (&at_precision(s, wp), &at_precision(x, wp));
    let one = BigFloat::from_i32(1, wp);
    let two = BigFloat::from_i32(2, wp);
    // "tiny" replaces exact zeros in Lentz's algorithm.
    let mut tiny = BigFloat::from_i32(1, wp);
    tiny.set_exponent(-(4 * wp as i32));
    let mut b = x.add(&one, wp, RM).sub(s, wp, RM);
    let mut c = one.div(&tiny, wp, RM);
    let mut d = if b.is_zero() {
        one.div(&tiny, wp, RM)
    } else {
        one.div(&b, wp, RM)
    };
    let mut h = d.clone();
    let mut converged = false;
    for i in 1..=max_iter {
        let i_bf = BigFloat::from_i128(i as i128, wp);
        // a_i = −i (i − s)
        let an = i_bf.mul(&i_bf.sub(s, wp, RM), wp, RM).neg();
        b = b.add(&two, wp, RM);
        d = an.mul(&d, wp, RM).add(&b, wp, RM);
        if d.is_zero() {
            d = tiny.clone();
        }
        c = b.add(&an.div(&c, wp, RM), wp, RM);
        if c.is_zero() {
            c = tiny.clone();
        }
        d = one.div(&d, wp, RM);
        let del = d.mul(&c, wp, RM);
        h = h.mul(&del, wp, RM);
        let dev = del.sub(&one, wp, RM);
        if negligible(&dev, &one, wp) {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(SymplexError::ComputationFailed {
            operation: "evalf",
            reason: format!(
                "continued fraction for the incomplete gamma function did not converge in \
                 {max_iter} steps"
            ),
        });
    }
    let pref = gamma_prefactor(s, x, &mut w.at(wp))?;
    Ok(pref.mul(&h, wp, RM))
}

/// Series for `γ(s, x) = x^s e^{−x} Σ_{k≥0} x^k / (s (s+1) … (s+k))`,
/// `s > 0`, `x ≥ 0`: positive terms, decreasing from `k ≈ x − s` on.  Near
/// `x ≈ s` about `√(2s·wp·ln 2)` of them are needed (60 000 at `s = 10⁷`),
/// each term rounding its predecessor's error once more, so the sum carries
/// `log₂(budget) + 8` guard bits.  The budget `2x + wp + 50` is capped at
/// `10⁷` terms (a refusal beyond, for shapes near `10¹³` and above).
fn lowergamma_series(s: &BigFloat, x: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let wp = w.bits();
    if x.is_zero() {
        return Ok(BigFloat::new(wp));
    }
    let x_f = f64_of(x)?;
    let max_terms = ((2.0 * x_f).min(1e7) as usize).saturating_add(wp + 50);
    let wp = wp + magnitude_bits(max_terms as f64) + 8;
    let (s, x) = (&at_precision(s, wp), &at_precision(x, wp));
    let one = BigFloat::from_i32(1, wp);
    let mut term = one.div(s, wp, RM);
    let mut sum = term.clone();
    let mut converged = false;
    for k in 1..=max_terms {
        let s_plus_k = s.add(&BigFloat::from_i128(k as i128, wp), wp, RM);
        term = term.mul(x, wp, RM).div(&s_plus_k, wp, RM);
        sum = sum.add(&term, wp, RM);
        if negligible(&term, &sum, wp) {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(series_did_not_converge("lowergamma", max_terms));
    }
    let pref = gamma_prefactor(s, x, &mut w.at(wp))?;
    Ok(pref.mul(&sum, wp, RM))
}

/// `Γ(s, x)` at working precision `wp` for real `s`, `x > 0`, to a relative
/// error of about `2^−wp`:
///
/// * `x ≥ max(1, s + 1)`: the continued fraction [`uppergamma_cf`];
/// * `s > 0`, smaller `x`: `Γ(s) − γ(s, x)` ([`uppergamma_by_difference`]);
/// * `s ≤ 0`, `x < 1`: the downward recurrence ([`uppergamma_downward`]).
fn uppergamma_wp(s: &BigFloat, x: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let wp = w.bits();
    // |s| below the working precision, 0 < x < 2: Γ(s, x) = E₁(x) + δ with
    // |δ| = |∫_x^∞ (t^s − 1) e^{−t}/t dt| ≤ 2|s|((ln x)²/2 + 1) and E₁(x) ≥
    // E₁(2) = 0.0489, so δ is below 2^{−wp} of the value once
    // log₂(1/|s|) ≥ wp + 8 + 2·log₂(|ln x| + 1).  (0.28 reached this limit
    // only when s underflowed f64, by taking it for the integer 0.)
    if !s.is_zero()
        && f64_of(x)? < 2.0
        && inverse_bits(&s.abs()) >= wp + 8 + 2 * magnitude_bits(ln_magnitude(x) + 1.0)
    {
        return Ok(arb_ei(&x.neg(), w)?.neg());
    }
    if bf_strictly_positive(s) {
        return uppergamma_positive(s, x, w);
    }
    if f64_of(x)? >= 1.0 {
        return uppergamma_cf(s, x, w);
    }
    // Far left of 0 the downward recurrence takes |s| steps, while the
    // continued fraction's partial quotients are about −i/|s| for i ≪ |s|
    // and it converges in a few steps.  (0.28 ran the recurrence for any
    // s: `expint(10²⁰/3, 1/7)` overflowed `2·steps` — a panic in debug
    // builds, an endless loop in release ones.)
    if f64_of(s)? < -(DOWNWARD_MAX_STEPS as f64) {
        return uppergamma_cf(s, x, w);
    }
    uppergamma_downward(s, x, w)
}

/// The most steps of the downward recurrence for `Γ(s, x)`, `s ≤ 0`,
/// `x < 1`; further left the continued fraction is used.
const DOWNWARD_MAX_STEPS: usize = 256;

/// `Γ(s, x)` for `s > 0`, `x > 0`: the continued fraction for
/// `x ≥ s + 1`, [`uppergamma_by_difference`] below.
fn uppergamma_positive(
    s: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let wp = w.bits();
    let s_f = f64_of(s)?;
    // x ≥ s + 1 exactly (in f64, s + 1 is s from 2⁵³ on, and a huge s = x
    // went to the fraction, which needs ~27·s^{1/3} steps there).
    if !bf_lt(x, &s.add(&BigFloat::from_i32(1, 64), exact_bits(s, wp), RM)) {
        return uppergamma_cf(s, x, w);
    }
    uppergamma_by_difference(s, s_f, x, w)
}

/// A refusal when `Γ(s) > 2^{EXPONENT_MAX + 3}` (`s ≥ 1`): the incomplete
/// gamma functions `Γ(s, x)` for `x ≤ s + 1` (at least `e^{−2}·Γ(s)`,
/// see [`uppergamma_by_difference`]) and `γ(s, x)` for `x ≥ s` (at least
/// `Γ(s)/2`, `P(s, s) > ½`) overflow the arbitrary-precision exponent
/// range.  Before 0.30 `uppergamma(10²⁰/3, 10²⁰/3)` (about `10^(6.3·10²⁰)`)
/// ran the continued fraction for two million steps (20 s in a debug
/// build) and failed to converge.
fn incomplete_gamma_overflows(s: &BigFloat, what: &str) -> Result<(), SymplexError> {
    let Ok(s_f) = f64_of(s) else {
        return Ok(());
    };
    if s_f.is_nan() || s_f < 1.0 {
        return Ok(());
    }
    // log₂ Γ(s) ≥ ((s − ½) ln s − s)/ln 2 (Stirling, the rest positive).
    let lg = ((s_f - 0.5) * s_f.ln() - s_f) * std::f64::consts::LOG2_E;
    if lg - 3.0 > f64::from(astro_float::EXPONENT_MAX) {
        return Err(unevaluable(format!(
            "{what}: the value (about 2^{lg:.3e}) overflows the arbitrary-precision exponent range"
        )));
    }
    Ok(())
}

/// `Γ(s, x) = Γ(s) − γ(s, x)` for `s > 0`, `x < s + 1` (the region of the
/// series).  The difference cancels `log₂(1/Q(s, x))` bits, and there
/// `Q(s, x) ≥ Q(s, s + 1)`, which is at least `e^{−2}` for `s ≥ 1` (it rises
/// from `Q(1, 2) = e^{−2}` towards ½) and at least
/// `Γ(s, 2)/Γ(s) ≥ E₁(2)·s > 0.048·s` for `s < 1` (`t^{s−1} ≥ 1/t` on
/// `[2, ∞)`, and `Γ(s) ≤ 1/s`; mpmath finds `min Q(s, s + 1)/s = e^{−2}`
/// over `s ≤ 1`): at most `log₂(1/min(1, s)) + 5` bits, added to the working
/// precision up front, with the bits of `ln Γ(s)` for the relative accuracy
/// of `Γ(s)` itself.  0.28 assumed `Q` was never small here and added
/// guard bits for `x` instead — `1.44·x` of them (10⁷ bits, a hang, at
/// `x ≈ 7·10⁶`) and none for a small `s`: `uppergamma(10⁻¹⁰⁰, ½)` came out
/// `0.5649…`, truly `0.5597…`.
fn uppergamma_by_difference(
    s: &BigFloat,
    s_f: f64,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let wp = w.bits();
    if inverse_bits(s) > cancellation_cap(wp) {
        return Err(special_exhausted(wp));
    }
    let wp2 = wp + 8 + inverse_bits(s) + lngamma_bits(s, s_f);
    let sw = at_precision(s, wp2);
    let xw = at_precision(x, wp2);
    let g = arb_gamma_real(&sw, &mut w.at(wp2))?;
    let lower = lowergamma_series(&sw, &xw, &mut w.at(wp2))?;
    Ok(round_to(g.sub(&lower, wp2, RM), wp))
}

/// `Γ(s, x)` for `s ≤ 0`, `0 < x < 1`: from `Γ(s₀, x)` with
/// `s₀ = s + n ∈ (0, 1)` (or from `Γ(0, x) = E₁(x)` for an integer
/// `s = −n`) by `n` downward steps `Γ(σ − 1, x) = (t − Γ(σ, x))/(1 − σ)`,
/// `t = x^{σ−1}e^{−x}`.  Each step subtracts two positive numbers, and for
/// `x ≤ 1`
///
/// ```text
/// t − Γ(σ, x) = t·∫₀^∞ (1 − (1 + u/x)^{σ−1}) e^{−u} du ≥ 0.40·min(1, 1 − σ)·t
/// ```
///
/// (convexity in the exponent, and `∫₀^∞ u/(1 + u) e^{−u} du = 1 − e·E₁(1)
/// = 0.4037`), so a step loses at most `log₂(1/min(1, 1 − σ)) + 1.4` bits —
/// many in the first step when `s` sits just below an integer.  The losses
/// are measured as they occur and the evaluation is repeated with a larger
/// reserve when they exceed it.  0.28 reserved 16 bits per step and took an
/// `s` within `10⁻¹²` of an integer for that integer:
/// `expint(3 + 10⁻¹⁴, ⅓)` was wrong from its 14th digit,
/// `uppergamma(−2 + 10⁻¹⁵, ½)` from its 16th.
fn uppergamma_downward(
    s: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let wp = w.bits();
    let s_f = f64_of(s)?;
    let integer = s.is_int();
    let steps = if integer {
        (-s_f).round() as usize
    } else {
        (-s_f).ceil() as usize
    };
    let cap = cancellation_cap(wp);
    let mut reserve = steps.saturating_mul(2).saturating_add(32);
    loop {
        let wp2 = wp + reserve;
        let sw = at_precision(s, wp2);
        let xw = at_precision(x, wp2);
        // Base point: Γ(0, x) = E₁(x) = −Ei(−x) for an integer s; otherwise
        // Γ(s₀, x) with s₀ = s + steps ∈ (0, 1).
        let (mut cur_s, mut val) = if integer {
            (BigFloat::new(wp2), arb_ei(&xw.neg(), &mut w.at(wp2))?.neg())
        } else {
            let s0 = sw.add(&BigFloat::from_i128(steps as i128, wp2), wp2, RM);
            let v = uppergamma_positive(&s0, &xw, &mut w.at(wp2))?;
            (s0, v)
        };
        let one = BigFloat::from_i32(1, wp2);
        let mut lost = 0usize;
        for _ in 0..steps {
            let next_s = cur_s.sub(&one, wp2, RM);
            let t = gamma_prefactor(&next_s, &xw, &mut w.at(wp2))?;
            let diff = t.sub(&val, wp2, RM);
            match (t.exponent(), diff.exponent()) {
                (Some(et), Some(ed)) if bf_strictly_positive(&diff) => {
                    lost += usize::try_from(i64::from(et) - i64::from(ed)).unwrap_or(0);
                }
                _ => {
                    lost = wp2;
                    break;
                }
            }
            val = diff.div(&next_s.neg(), wp2, RM);
            cur_s = next_s;
        }
        if lost + 16 <= reserve {
            return Ok(round_to(val, wp));
        }
        if reserve >= cap {
            return Err(special_exhausted(wp));
        }
        reserve = (lost + 32).max(2 * reserve).min(cap);
    }
}

/// Upper incomplete gamma `Γ(s, x)` for real `s`, `x ≥ 0`.
fn arb_uppergamma(s: &BigFloat, x: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    if x.is_nan() || s.is_nan() || x.is_inf_neg() || s.is_inf() {
        return Err(unevaluable("uppergamma of special float value"));
    }
    if x.is_inf_pos() {
        return Ok(BigFloat::new(prec));
    }
    if x.is_negative() {
        return Err(unevaluable(
            "uppergamma of negative argument not yet supported in evalf",
        ));
    }
    let wp = prec + 32;
    let one = BigFloat::from_i32(1, 64);
    if bf_strictly_positive(s) && !bf_gt(x, &s.add(&one, exact_bits(s, wp), RM)) {
        incomplete_gamma_overflows(s, "uppergamma")?;
    }
    if x.is_zero() {
        if !bf_strictly_positive(s) {
            return Err(unevaluable("uppergamma(s, 0) diverges for s ≤ 0"));
        }
        let s_f = f64_of(s)?;
        let wp = wp + lngamma_bits(s, s_f);
        return Ok(round_to(arb_gamma_real(s, &mut w.at(wp))?, prec));
    }
    let r = uppergamma_wp(s, x, &mut w.at(wp)).map_err(requested_at(prec))?;
    Ok(round_to(r, prec))
}

/// Lower incomplete gamma `γ(s, x)` for real `s > 0`, `x ≥ 0`: the series
/// for `x < s + 1`, `Γ(s) − Γ(s, x)` beyond, where `P(s, x) ≥ P(s, s + 1)
/// > 0.51` (mpmath, over `s` from `10⁻²⁷` to `10³`, rising towards ½ + … for
/// larger `s`) and at most a bit cancels.
fn arb_lowergamma(s: &BigFloat, x: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    if x.is_nan() || s.is_nan() || s.is_inf() || x.is_inf_neg() {
        return Err(unevaluable("lowergamma of special float value"));
    }
    if !bf_strictly_positive(s) {
        return Err(unevaluable(
            "lowergamma requires s > 0 for numerical evaluation",
        ));
    }
    if x.is_negative() {
        return Err(unevaluable(
            "lowergamma of negative argument not yet supported in evalf",
        ));
    }
    let s_f = f64_of(s)?;
    let wp = prec + 32;
    if x.is_inf_pos() {
        let wp = wp + lngamma_bits(s, s_f);
        return Ok(round_to(arb_gamma_real(s, &mut w.at(wp))?, prec));
    }
    if !bf_lt(x, s) {
        incomplete_gamma_overflows(s, "lowergamma")?;
    }
    if bf_lt(x, &s.add(&BigFloat::from_i32(1, 64), exact_bits(s, wp), RM)) {
        let r = lowergamma_series(s, x, &mut w.at(wp)).map_err(requested_at(prec))?;
        return Ok(round_to(r, prec));
    }
    let wp = wp + 4 + lngamma_bits(s, s_f);
    let g = arb_gamma_real(s, &mut w.at(wp))?;
    let upper = uppergamma_cf(s, x, &mut w.at(wp))?;
    Ok(round_to(g.sub(&upper, wp, RM), prec))
}

/// Generalised exponential integral `E_n(x) = x^{n−1} Γ(1−n, x)` for real
/// `n` and `x > 0` (`E_n(0) = 1/(n−1)` for `n > 1`).
fn arb_expint(n: &BigFloat, x: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    if x.is_nan() || n.is_nan() || n.is_inf() {
        return Err(unevaluable("expint of special float value"));
    }
    if x.is_inf_pos() {
        return Ok(BigFloat::new(prec));
    }
    let wp = prec + 32;
    let one = BigFloat::from_i32(1, wp);
    if x.is_zero() {
        // `E_n(0) = +∞` for `n ≤ 1`: `E₁` has a logarithmic pole at 0.  (Up
        // to 0.49 `n − 1 = +0` counted as positive and `expint(1, 0)` came
        // out `1/0`, refused as "precision exhausted".)
        let n_minus_1 = n.sub(&one, wp, RM);
        if !is_strictly_positive(&n_minus_1) {
            return Err(unevaluable("expint(n, 0) diverges for n ≤ 1"));
        }
        return Ok(round_to(one.div(&n_minus_1, wp, RM), prec));
    }
    if x.is_negative() {
        return Err(unevaluable(
            "expint of negative argument is complex; not yet supported in evalf",
        ));
    }
    let s = one.sub(n, wp, RM);
    let g = uppergamma_wp(&s, x, &mut w.at(wp)).map_err(requested_at(prec))?;
    let x_pow = bf_pow(x, &n.sub(&one, wp, RM), &mut w.at(wp));
    Ok(round_to(x_pow.mul(&g, wp, RM), prec))
}

/// `Shi(x)` (`want_shi`) or `Chi(x)` for `x > 0` (`Shi` is odd and is
/// returned for any real `x`).
///
/// * `|x| ≤ 40`: `Shi(x) = Σ x^{2k+1}/((2k+1)(2k+1)!)`,
///   `Chi(x) = γ + ln x + Σ_{k≥1} x^{2k}/((2k)(2k)!)` — all terms positive.
/// * larger `|x|`: `Shi = (Ei(x) − Ei(−x))/2`, `Chi = (Ei(x) + Ei(−x))/2`.
fn arb_shi_chi(x: &BigFloat, want_shi: bool, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    if x.is_nan() || x.is_inf() {
        return Err(unevaluable("Shi/Chi of special float value"));
    }
    if x.is_zero() {
        return if want_shi {
            Ok(BigFloat::new(prec))
        } else {
            Err(unevaluable("Chi(0) is zoo"))
        };
    }
    let wp = prec + 32;
    let ax = x.abs();
    let x_f = f64_of(&ax)?;
    let r = if x_f > 40.0 {
        let ei_pos = arb_ei(&ax, &mut w.at(wp))?;
        let ei_neg = arb_ei(&ax.neg(), &mut w.at(wp))?;
        let two = BigFloat::from_i32(2, wp);
        if want_shi {
            ei_pos.sub(&ei_neg, wp, RM).div(&two, wp, RM)
        } else {
            ei_pos.add(&ei_neg, wp, RM).div(&two, wp, RM)
        }
    } else {
        let mut xw = ax.clone();
        let _ = xw.set_precision(wp, RM);
        let x2 = xw.mul(&xw, wp, RM);
        let max_terms = (x_f * 1.5) as usize + wp + 40;
        let name = if want_shi { "Shi" } else { "Chi" };
        let mut converged = false;
        let series = if want_shi {
            // t_k = x^{2k+1}/(2k+1)!
            let mut t = xw.clone();
            let mut sum = BigFloat::new(wp);
            for k in 0..max_terms {
                if k > 0 {
                    let d = BigFloat::from_i128((2 * k as i128) * (2 * k as i128 + 1), wp);
                    t = t.mul(&x2, wp, RM).div(&d, wp, RM);
                }
                let term = t.div(&BigFloat::from_i128(2 * k as i128 + 1, wp), wp, RM);
                sum = sum.add(&term, wp, RM);
                if negligible(&term, &sum, wp) {
                    converged = true;
                    break;
                }
            }
            sum
        } else {
            let gamma = arb_euler_gamma(&mut w.at(wp))?;
            let ln_x = xw.ln(wp, RM, w.cc);
            let mut u = BigFloat::from_i32(1, wp); // x^{2k}/(2k)!
            let mut sum = BigFloat::new(wp);
            for k in 1..max_terms {
                let d = BigFloat::from_i128((2 * k as i128 - 1) * (2 * k as i128), wp);
                u = u.mul(&x2, wp, RM).div(&d, wp, RM);
                let term = u.div(&BigFloat::from_i128(2 * k as i128, wp), wp, RM);
                sum = sum.add(&term, wp, RM);
                if negligible(&term, &sum, wp) {
                    converged = true;
                    break;
                }
            }
            gamma.add(&ln_x, wp, RM).add(&sum, wp, RM)
        };
        if !converged {
            return Err(series_did_not_converge(name, max_terms));
        }
        series
    };
    let r = if want_shi && x.is_negative() {
        r.neg()
    } else {
        r
    };
    Ok(round_to(r, prec))
}

/// Fresnel integrals `S(x)` (`want_s`) and `C(x)` for real `x` (both odd).
///
/// * `πx²/2 ≤ wp·ln 2`: Maclaurin series
///   `S = Σ (−1)^k (π/2)^{2k+1} x^{4k+3}/((2k+1)!(4k+3))`,
///   `C = Σ (−1)^k (π/2)^{2k} x^{4k+1}/((2k)!(4k+1))`, with guard bits for
///   the `e^{πx²/2}` cancellation.
/// * beyond: DLMF 7.12 auxiliary functions
///   `S = 1/2 − f cos(πx²/2) − g sin(πx²/2)`, `C = 1/2 + f sin(πx²/2) − g cos(πx²/2)`,
///   `f ~ (1/πx) Σ (−1)^m (4m−1)!!/(πx²)^{2m}`, `g ~ (1/πx) Σ (−1)^m (4m+1)!!/(πx²)^{2m+1}`
///   (optimal truncation error `~e^{−πx²/2}`).
fn arb_fresnel(x: &BigFloat, want_s: bool, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    if x.is_nan() || x.is_inf() {
        return Err(unevaluable("Fresnel integral of special float value"));
    }
    if x.is_zero() {
        return Ok(BigFloat::new(prec));
    }
    let base_wp = prec + 32;
    let ax = x.abs();
    let x_f = f64_of(&ax)?;
    let arg_f = std::f64::consts::FRAC_PI_2 * x_f * x_f;

    // `x = 2^(e−1)… 2^e`: the auxiliary functions are `~1/(πx)`, and the
    // phase `πx²/2` has `2e` bits before the point.
    let e = usize::try_from(ax.exponent().map_or(0, i64::from).max(0)).unwrap_or(0);
    let r = if e > base_wp + 2 {
        // `f cos + g sin` is below `2^−wp` next to 1/2: before 0.30 the
        // phase was reduced anyway (`fresnels(erfi(335))`, `x ≈ 2^161900`,
        // asked for π to 323,800 bits and hung).
        BigFloat::from_f64(0.5, base_wp)
    } else if arg_f > (base_wp as f64) * std::f64::consts::LN_2 {
        let wp = base_wp;
        let mut xw = ax.clone();
        let _ = xw.set_precision(wp, RM);
        let one = BigFloat::from_i32(1, wp);
        let pi = w.cc.pi(wp, RM).clone();
        let pi_x2 = pi.mul(&xw, wp, RM).mul(&xw, wp, RM);
        let inv = one.div(&pi_x2, wp, RM); // 1/(πx²)
        let inv2 = inv.mul(&inv, wp, RM);
        // f-series: (4m−1)!!/(πx²)^{2m}; g-series: (4m+1)!!/(πx²)^{2m+1}
        let mut tf = one.clone();
        let mut tg = inv.clone();
        let mut f_sum = BigFloat::new(wp);
        let mut g_sum = BigFloat::new(wp);
        let mut prev: Option<BigFloat> = None;
        for m in 0..(wp * 2 + 100) {
            if m > 0 {
                // (4m−1)!!/(4m−5)!! = (4m−3)(4m−1); (4m+1)!!/(4m−3)!! = (4m−1)(4m+1)
                let a = BigFloat::from_i128((4 * m as i128 - 3) * (4 * m as i128 - 1), wp);
                let b = BigFloat::from_i128((4 * m as i128 - 1) * (4 * m as i128 + 1), wp);
                tf = tf.mul(&a, wp, RM).mul(&inv2, wp, RM);
                tg = tg.mul(&b, wp, RM).mul(&inv2, wp, RM);
            }
            if let Some(ref p) = prev
                && bf_gt(&tf, p)
            {
                break;
            }
            if m % 2 == 0 {
                f_sum = f_sum.add(&tf, wp, RM);
                g_sum = g_sum.add(&tg, wp, RM);
            } else {
                f_sum = f_sum.sub(&tf, wp, RM);
                g_sum = g_sum.sub(&tg, wp, RM);
            }
            if negligible(&tf, &one, wp) {
                break;
            }
            prev = Some(tf.clone());
        }
        let pi_x = pi.mul(&xw, wp, RM);
        let f = f_sum.div(&pi_x, wp, RM);
        let g = g_sum.div(&pi_x, wp, RM);
        let two = BigFloat::from_i32(2, wp);
        // The phase to an absolute `2^−wp`: `2e` more bits.  Before 0.30 it
        // was formed at `wp` bits, an absolute error of `2^(2e−wp)` times
        // the amplitude `2^−e` — wrong digits once `e` passed the guard
        // bits: `fresnels(10⁴⁵)` at 60 digits ended `…816212`, truly
        // `…816209`.
        let wt = wp + 2 * e;
        let mut xt = ax.clone();
        let _ = xt.set_precision(wt, RM);
        let theta = w.cc.pi(wt, RM).mul(&xt, wt, RM).mul(&xt, wt, RM).div(
            &BigFloat::from_i32(2, wt),
            wt,
            RM,
        );
        let s = theta.sin(wp, RM, w.cc);
        let c = theta.cos(wp, RM, w.cc);
        let half = one.div(&two, wp, RM);
        if want_s {
            half.sub(&f.mul(&c, wp, RM), wp, RM)
                .sub(&g.mul(&s, wp, RM), wp, RM)
        } else {
            half.add(&f.mul(&s, wp, RM), wp, RM)
                .sub(&g.mul(&c, wp, RM), wp, RM)
        }
    } else {
        let wp = Prec::new(base_wp)
            .guarded(Guard::for_log2(arg_f * std::f64::consts::LOG2_E), Prec::MAX)
            .ok_or_else(|| special_exhausted(prec))?
            .plus(16)
            .bits();
        let mut xw = ax.clone();
        let _ = xw.set_precision(wp, RM);
        let pi = w.cc.pi(wp, RM).clone();
        let two = BigFloat::from_i32(2, wp);
        let half_pi = pi.div(&two, wp, RM);
        let x2 = xw.mul(&xw, wp, RM);
        // u = (π/2) x²;  terms: S_k = (−1)^k u^{2k+1} x /((2k+1)! (4k+3)),
        //                          C_k = (−1)^k u^{2k} x /((2k)! (4k+1))
        let u = half_pi.mul(&x2, wp, RM);
        let u2 = u.mul(&u, wp, RM);
        let mut t = if want_s {
            u.mul(&xw, wp, RM)
        } else {
            xw.clone()
        };
        let mut sum = BigFloat::new(wp);
        let max_terms = (arg_f * 2.0) as usize + wp + 40;
        let mut converged = false;
        for k in 0..max_terms {
            if k > 0 {
                let d = if want_s {
                    BigFloat::from_i128((2 * k as i128) * (2 * k as i128 + 1), wp)
                } else {
                    BigFloat::from_i128((2 * k as i128 - 1) * (2 * k as i128), wp)
                };
                t = t.mul(&u2, wp, RM).div(&d, wp, RM);
            }
            let div = BigFloat::from_i128(4 * k as i128 + if want_s { 3 } else { 1 }, wp);
            let term = t.div(&div, wp, RM);
            if k % 2 == 0 {
                sum = sum.add(&term, wp, RM);
            } else {
                sum = sum.sub(&term, wp, RM);
            }
            if negligible(&term, &sum, wp) {
                converged = true;
                break;
            }
        }
        if !converged {
            let name = if want_s { "fresnels" } else { "fresnelc" };
            return Err(series_did_not_converge(name, max_terms));
        }
        sum
    };
    let r = if x.is_negative() { r.neg() } else { r };
    Ok(round_to(r, prec))
}

// ── Incomplete beta ───────────────────────────────────────────────────────────────

/// `|ln v|` bounded from the binary exponent of `v > 0` (so for any
/// magnitude, and never below `ln 2`).
fn ln_magnitude(v: &BigFloat) -> f64 {
    v.exponent()
        .map_or(0.0, |e| (f64::from(e).abs() + 1.0) * std::f64::consts::LN_2)
}

/// A bound on `|ln Γ(s)|` for `s > 0` (see [`lngamma_bits`]).
fn lngamma_magnitude(s: &BigFloat, s_f: f64) -> f64 {
    if s_f >= 1.0 {
        s_f * s_f.ln() + s_f + 2.0
    } else {
        ln_magnitude(s) + 2.0
    }
}

/// `ln B(a, b) = ln Γ(a) + ln Γ(b) − ln Γ(a + b)` for `a, b > 0` to an
/// absolute error of about `2^−wp`: each `ln Γ` carries the bits of the
/// largest magnitude as guard (they are `≈ 1.7·10⁹` for shapes near `10⁸`
/// and cancel to the size of `ln B`), which also covers the rounding of
/// `a + b`.
fn ln_beta_wp(a: &BigFloat, b: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let wp = w.bits();
    let a_f = f64_of(a)?;
    let b_f = f64_of(b)?;
    let ab_f = a_f + b_f;
    let mag = lngamma_magnitude(a, a_f)
        .max(lngamma_magnitude(b, b_f))
        .max(ab_f * ab_f.ln().abs() + ab_f + 2.0);
    let p = wp + magnitude_bits(mag) + 8;
    let a_plus_b = a.add(b, p, RM);
    let la = stirling_log_gamma(a, &mut w.at(p))?;
    let lb = stirling_log_gamma(b, &mut w.at(p))?;
    let lab = stirling_log_gamma(&a_plus_b, &mut w.at(p))?;
    Ok(la.add(&lb, p, RM).sub(&lab, p, RM))
}

/// `B(a, b) = Γ(a)Γ(b)/Γ(a+b)` at working precision `wp` for `a, b > 0`, as
/// `exp(ln B(a, b))`: the three gamma values need not be representable
/// (`Γ(10⁸)` is beyond the exponent range, `B(10⁸, 10⁸) = 2^{−2·10⁸}` is
/// not).
fn beta_wp(a: &BigFloat, b: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let wp = w.bits();
    Ok(ln_beta_wp(a, b, w)?.exp(wp, RM, w.cc))
}

/// The continued fraction of the regularised incomplete beta function
/// (DLMF 8.17.22; evaluated by the modified Lentz algorithm of Thompson &
/// Barnett, J. Comput. Phys. 64 (1986) 490):
///
/// ```text
/// I_x(a, b) = x^a (1−x)^b / (a B(a, b)) · 1/(1 + d₁/(1 + d₂/(1 + …)))
/// d_{2m}   =  m (b−m) x / ((a+2m−1)(a+2m))
/// d_{2m+1} = −(a+m)(a+b+m) x / ((a+2m)(a+2m+1))
/// ```
///
/// Returns the fraction `1/(1 + d₁/…)` only (the prefactor is applied by
/// the caller).  Converges quickly for `x < (a+1)/(a+b+2)`.  Evaluated with
/// `log₂(2·budget) + 8` guard bits for the roundings of its steps; the
/// shape-dependent part of the budget, `10·√(a + b)`, is capped at `2·10⁶`
/// steps (seconds; shapes near `10¹⁴` still converge within it, and
/// beyond it the fraction is refused rather than run for minutes).
fn betainc_cf(
    a: &BigFloat,
    b: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let wp = w.bits();
    let ab_f = f64_of(&a.add(b, 64, RM))?;
    // The number of steps grows like √(max(a, b)) before the linear
    // convergence phase; budget generously.
    let max_iter = (40 * wp + 2000).saturating_add((10.0 * ab_f.abs().sqrt()).min(2e6) as usize);
    let wp = wp + magnitude_bits(2.0 * max_iter as f64) + 8;
    let (a, b, x) = (
        &at_precision(a, wp),
        &at_precision(b, wp),
        &at_precision(x, wp),
    );
    let one = BigFloat::from_i32(1, wp);
    // "tiny" replaces exact zeros in Lentz's algorithm.
    let mut tiny = BigFloat::from_i32(1, wp);
    tiny.set_exponent(-(4 * wp as i32));
    let qab = a.add(b, wp, RM);
    let qap = a.add(&one, wp, RM);
    let qam = a.sub(&one, wp, RM);
    let mut c = one.clone();
    let mut d = one.sub(&qab.mul(x, wp, RM).div(&qap, wp, RM), wp, RM);
    if d.is_zero() {
        d = tiny.clone();
    }
    d = one.div(&d, wp, RM);
    let mut h = d.clone();
    let mut converged = false;
    // One Lentz step with partial numerator `aa`; returns `del = d·c`.
    let step = |aa: &BigFloat, c: &mut BigFloat, d: &mut BigFloat| -> BigFloat {
        *d = one.add(&aa.mul(d, wp, RM), wp, RM);
        if d.is_zero() {
            *d = tiny.clone();
        }
        *c = one.add(&aa.div(c, wp, RM), wp, RM);
        if c.is_zero() {
            *c = tiny.clone();
        }
        *d = one.div(d, wp, RM);
        d.mul(c, wp, RM)
    };
    for m in 1..=max_iter {
        let m_bf = BigFloat::from_i128(m as i128, wp);
        let m2_bf = BigFloat::from_i128(2 * m as i128, wp);
        // Even step: d_{2m} = m (b−m) x / ((a+2m−1)(a+2m)).
        let num = m_bf.mul(&b.sub(&m_bf, wp, RM), wp, RM).mul(x, wp, RM);
        let den = qam.add(&m2_bf, wp, RM).mul(&a.add(&m2_bf, wp, RM), wp, RM);
        let aa = num.div(&den, wp, RM);
        let del = step(&aa, &mut c, &mut d);
        h = h.mul(&del, wp, RM);
        // Odd step: d_{2m+1} = −(a+m)(a+b+m) x / ((a+2m)(a+2m+1)).
        let num = a
            .add(&m_bf, wp, RM)
            .mul(&qab.add(&m_bf, wp, RM), wp, RM)
            .mul(x, wp, RM)
            .neg();
        let den = a.add(&m2_bf, wp, RM).mul(&qap.add(&m2_bf, wp, RM), wp, RM);
        let aa = num.div(&den, wp, RM);
        let del = step(&aa, &mut c, &mut d);
        h = h.mul(&del, wp, RM);
        let dev = del.sub(&one, wp, RM);
        if negligible(&dev, &one, wp) {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(SymplexError::ComputationFailed {
            operation: "evalf",
            reason: format!(
                "continued fraction for the incomplete beta function did not converge in \
                 {max_iter} steps"
            ),
        });
    }
    Ok(h)
}

/// `x^a y^b / (a B(a, b))` for `0 < x < 1`, `y = 1 − x`, as one
/// exponential of `a ln x + b ln y − ln a − ln B(a, b)`: the exponent is
/// formed with the bits of its magnitude as guard, so that its absolute
/// error — the relative error of the result — is `2^−wp`, and `ln B` comes
/// from [`ln_beta_wp`].  0.28 formed `x^a`, `y^b` and
/// `B(a, b) = Γ(a)Γ(b)/Γ(a + b)` separately: `Γ(10⁸)` is beyond the exponent
/// range, and `betainc_regularized(10⁸ + ⅓, 10⁸ + 1/7, 0, ½)` was refused.
fn betainc_prefactor(
    a: &BigFloat,
    b: &BigFloat,
    x: &BigFloat,
    y: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let wp = w.bits();
    let a_f = f64_of(a)?;
    let b_f = f64_of(b)?;
    let ab_f = a_f + b_f;
    let mag = a_f * ln_magnitude(x)
        + b_f * ln_magnitude(y)
        + ln_magnitude(a)
        + lngamma_magnitude(a, a_f)
        + lngamma_magnitude(b, b_f)
        + ab_f * ab_f.ln().abs()
        + ab_f
        + 2.0;
    let p = wp + magnitude_bits(mag) + 8;
    let e = a
        .mul(&x.ln(p, RM, w.cc), p, RM)
        .add(&b.mul(&y.ln(p, RM, w.cc), p, RM), p, RM)
        .sub(&a.ln(p, RM, w.cc), p, RM)
        .sub(&ln_beta_wp(a, b, &mut w.at(p))?, p, RM);
    Ok(e.exp(wp, RM, w.cc))
}

/// `I_x(a, b)` for `x ≤ (a+1)/(a+b+2)` (`one_minus_x = 1 − x` is passed in
/// so that it is exact): continued fraction times `x^a (1−x)^b / (a B(a, b))`.
fn betainc_reg_direct(
    a: &BigFloat,
    b: &BigFloat,
    x: &BigFloat,
    one_minus_x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let wp = w.bits();
    let h = betainc_cf(a, b, x, w)?;
    let pref = betainc_prefactor(a, b, x, one_minus_x, w)?;
    Ok(pref.mul(&h, wp, RM))
}

/// The switch point `(a+1)/(a+b+2)` of the continued fraction: below it
/// `I_x(a, b)` is computed directly, above it `I_{1−x}(b, a)`.
fn betainc_switch_point(a: &BigFloat, b: &BigFloat, wp: usize) -> BigFloat {
    let one = BigFloat::from_i32(1, wp);
    let two = BigFloat::from_i32(2, wp);
    let a_plus_1 = a.add(&one, wp, RM);
    let a_plus_b_plus_2 = a.add(b, wp, RM).add(&two, wp, RM);
    a_plus_1.div(&a_plus_b_plus_2, wp, RM)
}

/// `(I_x(a, b), I_{1−x}(b, a))` for `a, b > 0`, `0 < x < 1`: both tails of
/// the regularised incomplete beta function, each to a relative error of
/// about `2^−wp`.
///
/// The continued fraction converges fast on the side of the smaller
/// argument: it gives `I_x(a, b)` directly for `x ≤ (a+1)/(a+b+2)` and
/// `I_{1−x}(b, a)` above, and the other tail is `1 −` it.  That difference
/// cancels when the directly computed tail is close to 1: a small first
/// shape piles the mass against 0 (`I_{1/10}(10⁻⁶⁰, ⅓) = 1 − 4.8·10⁻⁶⁰`).
/// 0.28 assumed at most a bit was lost there and printed garbage as
/// certified digits: `betainc_regularized(⅓, 10⁻¹⁰⁰, 0, 9/10)` was
/// `1.28·10⁻⁸²`, truly `4.79·10⁻¹⁰⁰`.  The loss is measured instead: the
/// direct tail `d`, computed to a relative `2^−p`, leaves `1 − d` with an
/// absolute error of `2^−p`, i.e. `p − log₂(1/(1 − d))` correct bits, and
/// `d` is recomputed at a precision covering the loss (a first shape
/// `s < 1` costs about `log₂(1/s)` bits, which are added up front), up to
/// [`cancellation_cap`].
fn betainc_tails_wp(
    a: &BigFloat,
    b: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<(BigFloat, BigFloat), SymplexError> {
    let wp = w.bits();
    let direct_lower = !bf_gt(x, &betainc_switch_point(a, b, wp));
    let first = if direct_lower { a } else { b };
    let ab_bits = magnitude_bits(f64_of(&a.add(b, 64, RM))?);
    let cap = cancellation_cap(wp);
    // The loss is about log₂(1/s) for a first shape s < 1 (less by at most
    // log₂|ln x|): beyond the cap, refuse before computing at that precision.
    let mut extra = 16 + inverse_bits(first);
    if extra > cap {
        return Err(special_exhausted(wp));
    }
    loop {
        let p = wp + extra;
        // 1 − x: exact for x ≥ ½ (the argument's own bits); rounded far
        // below the unit of `b ln y` otherwise.
        let py = p + 64 + ab_bits;
        let y = BigFloat::from_i32(1, py).sub(x, py, RM);
        let d = if direct_lower {
            betainc_reg_direct(a, b, x, &y, &mut w.at(p))?
        } else {
            betainc_reg_direct(b, a, &y, x, &mut w.at(p))?
        };
        let c = BigFloat::from_i32(1, p).sub(&d, p, RM);
        let lost = match c.exponent() {
            Some(e) if bf_strictly_positive(&c) => usize::try_from(1 - i64::from(e)).unwrap_or(0),
            _ => p,
        };
        if lost + 8 <= extra {
            let (d, c) = (round_to(d, wp), round_to(c, wp));
            return Ok(if direct_lower { (d, c) } else { (c, d) });
        }
        if extra >= cap {
            return Err(special_exhausted(wp));
        }
        extra = (lost + 16).max(2 * extra).min(cap);
    }
}

/// Both tails at a limit `t ∈ [0, 1]` (exact at the ends).
fn betainc_tails_at(
    a: &BigFloat,
    b: &BigFloat,
    t: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<(BigFloat, BigFloat), SymplexError> {
    let wp = w.bits();
    let one = BigFloat::from_i32(1, wp);
    if t.is_zero() {
        return Ok((BigFloat::new(wp), one));
    }
    if t.cmp(&one) == Some(0) {
        return Ok((one, BigFloat::new(wp)));
    }
    betainc_tails_wp(a, b, t, w)
}

/// `I_{hi}(a, b) − I_{lo}(a, b)` for `0 ≤ lo < hi ≤ 1` at working precision
/// `wp`, together with the number of bits lost to cancellation in the
/// subtraction (`wp` when the difference vanished entirely).
///
/// Both tails are known at each limit ([`betainc_tails_at`]), so the
/// difference is `L(hi) − L(lo) = U(lo) − U(hi)` with `L(t) = I_t(a, b)`
/// and `U(t) = I_{1−t}(b, a)`, and the form whose larger term is the smaller
/// is taken: no `1 − …` is formed here, so a lower tail (`lo = 0`) or an
/// upper one (`hi = 1`) is a single directly computed tail, and limits near
/// 1 keep their relative accuracy.  What remains is the cancellation of
/// close limits, which the caller covers with precision.
fn betainc_reg_diff_wp(
    a: &BigFloat,
    b: &BigFloat,
    lo: &BigFloat,
    hi: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<(BigFloat, usize), SymplexError> {
    let wp = w.bits();
    let (l_lo, u_lo) = betainc_tails_at(a, b, lo, w)?;
    let (l_hi, u_hi) = betainc_tails_at(a, b, hi, w)?;
    let (p, q) = if bf_gt(&l_hi, &u_lo) {
        (u_lo, u_hi)
    } else {
        (l_hi, l_lo)
    };
    let diff = p.sub(&q, wp, RM);
    let lost = match (p.exponent(), q.exponent(), diff.exponent()) {
        _ if q.is_zero() => 0,
        _ if !bf_strictly_positive(&diff) => wp,
        (Some(ep), Some(eq), Some(ed)) => {
            usize::try_from(i64::from(ep.max(eq)) - i64::from(ed)).unwrap_or(0)
        }
        _ => 0,
    };
    Ok((diff, lost))
}

/// Generalised incomplete beta `B_{(x1, x2)}(a, b) = ∫_{x1}^{x2} t^{a−1}(1−t)^{b−1} dt`
/// (or `I_{(x1, x2)}(a, b) = B_{(x1, x2)}(a, b)/B(a, b)` when `regularized`)
/// for real `a, b > 0` and `0 ≤ x1, x2 ≤ 1`.
///
/// Computed as `I_{x2}(a, b) − I_{x1}(a, b)` ([`betainc_reg_diff_wp`]); when
/// the two values nearly cancel (`x1 ≈ x2`) the working precision is raised
/// by the number of bits lost and the difference recomputed, until the
/// loss is covered — or refused beyond [`cancellation_cap`].  0.28 returned
/// its third attempt whatever it had lost.
#[allow(clippy::too_many_arguments)]
fn arb_betainc(
    a: &BigFloat,
    b: &BigFloat,
    x1: &BigFloat,
    x2: &BigFloat,
    regularized: bool,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    let name = if regularized {
        "betainc_regularized"
    } else {
        "betainc"
    };
    if [a, b, x1, x2].iter().any(|v| v.is_nan() || v.is_inf()) {
        return Err(unevaluable(format!("{name} of special float value")));
    }
    if !bf_strictly_positive(a) || !bf_strictly_positive(b) {
        return Err(unevaluable(format!(
            "{name} requires a, b > 0 for numerical evaluation"
        )));
    }
    let one = BigFloat::from_i32(1, prec);
    if [x1, x2].iter().any(|x| x.is_negative() || bf_gt(x, &one)) {
        return Err(unevaluable(format!(
            "{name}: the limits x1, x2 must lie in [0, 1] for numerical evaluation"
        )));
    }
    // Orient the limits so that lo < hi; the sign is restored at the end.
    let (lo, hi, negate) = match x1.cmp(x2) {
        Some(c) if c < 0 => (x1, x2, false),
        Some(c) if c > 0 => (x2, x1, true),
        _ => return Ok(BigFloat::new(prec)),
    };
    let cap = prec + cancellation_cap(prec);
    let mut wp = prec + 32;
    let mut diff = loop {
        let (d, lost) =
            betainc_reg_diff_wp(a, b, lo, hi, &mut w.at(wp)).map_err(requested_at(prec))?;
        // Done once the guard bits cover the cancellation (16 to spare).
        if lost + prec + 16 <= wp {
            break d;
        }
        if wp >= cap {
            return Err(special_exhausted(prec));
        }
        wp = (prec + 32 + lost).min(cap);
    };
    if !regularized {
        let beta = beta_wp(a, b, &mut w.at(wp))?;
        diff = diff.mul(&beta, wp, RM);
    }
    if negate {
        diff = diff.neg();
    }
    Ok(round_to(diff, prec))
}

// ── Polylogarithm / Dirichlet eta ───────────────────────────────────────────────

/// `ζ(m)` for an integer `m ≤ 0`, exactly: `ζ(0) = −1/2`,
/// `ζ(−k) = −B_{k+1}/(k+1)`.
fn zeta_nonpositive_int(m: i64, wp: usize) -> BigFloat {
    if m == 0 {
        return BigFloat::from_f64(-0.5, wp);
    }
    let k = (-m) as usize;
    let b = crate::base::bernoulli::bernoulli(k + 1);
    if b.is_zero() {
        return BigFloat::new(wp);
    }
    let v = -b / Ratio::from_integer(BigInt::from(k as i64 + 1));
    ratio_to_bigfloat(&v, wp)
}

/// `k^s` for integer `k ≥ 1`.
fn int_pow_s(k: usize, s: &BigFloat, s_int: Option<i64>, w: &mut Wp<'_>) -> BigFloat {
    let wp = w.bits();
    let kb = BigFloat::from_i128(k as i128, wp);
    match s_int {
        Some(n) if n >= 0 => kb.powi(n as usize, wp, RM),
        Some(n) => BigFloat::from_i32(1, wp).div(&kb.powi((-n) as usize, wp, RM), wp, RM),
        None => bf_pow(&kb, s, w),
    }
}

/// Number of terms after which `|z|^k k^{−s}` has certainly fallen below
/// `2^{−wp}·|z|` (the first term): about `wp·ln 2 / (−ln|z|)` for `s ≥ 0`;
/// for `s < 0` the factor `k^{|s|}` first grows the terms (peak near
/// `k = |s|/(−ln|z|)`), so the bound is found by doubling from there.
fn polylog_series_terms(s_f: f64, z_f: f64, wp: usize) -> usize {
    if !(z_f > 0.0 && z_f < 1.0) {
        return wp * 2 + 60;
    }
    let a = -z_f.ln(); // > 0
    let target = (wp as f64) * std::f64::consts::LN_2 + a;
    let mut k = target / a;
    let m = (-s_f).max(0.0);
    if m > 0.0 {
        // k·a − m·ln k increases for k > m/a.
        k = k.max(m / a);
        for _ in 0..200 {
            if k * a - m * k.ln() >= target {
                break;
            }
            k *= 2.0;
        }
    }
    (k.ceil() as usize).saturating_add(20)
}

/// Direct series `Li_s(z) = Σ_{k≥1} z^k/k^s` for `|z| < 1` (used up to
/// [`polylog_series_limit`]); see [`polylog_series_terms`] for the term
/// bound.  For `z < 0` and `s < 0` the alternating terms can be far
/// larger than the sum; the bits lost to that cancellation are measured
/// and the sum recomputed once with that many guard bits.
fn polylog_series(
    s: &BigFloat,
    s_int: Option<i64>,
    z: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let wp = w.bits();
    let (sum, lost) = polylog_series_raw(s, s_int, z, w)?;
    // A few bits (an alternating series with s ≥ 0 loses about one) are
    // covered by the caller's guard bits.
    if lost <= 4 {
        return Ok(sum);
    }
    let wp2 = wp + lost + 8;
    let mut sw = s.clone();
    let _ = sw.set_precision(wp2, RM);
    let mut zw = z.clone();
    let _ = zw.set_precision(wp2, RM);
    let (sum, _) = polylog_series_raw(&sw, s_int, &zw, &mut w.at(wp2))?;
    Ok(round_to(sum, wp))
}

/// One pass of [`polylog_series`] at working precision `wp`; also returns
/// the number of bits by which the largest term exceeded the final sum.
fn polylog_series_raw(
    s: &BigFloat,
    s_int: Option<i64>,
    z: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<(BigFloat, usize), SymplexError> {
    let wp = w.bits();
    let z_f = f64_of(z).unwrap_or(0.5).abs();
    let s_f = f64_of(s).unwrap_or(0.0);
    let max_terms = polylog_series_terms(s_f, z_f, wp);
    let mut zk = z.clone();
    let mut sum = BigFloat::new(wp);
    let mut max_exp: Option<i64> = None;
    let mut converged = false;
    for k in 1..=max_terms {
        if k > 1 {
            zk = zk.mul(z, wp, RM);
        }
        let term = zk.div(&int_pow_s(k, s, s_int, w), wp, RM);
        if let Some(e) = term.exponent() {
            max_exp = Some(max_exp.map_or(e as i64, |m| m.max(e as i64)));
        }
        sum = sum.add(&term, wp, RM);
        if negligible(&term, &sum, wp) {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(series_did_not_converge("polylog", max_terms));
    }
    let lost = match (max_exp, sum.exponent()) {
        (Some(m), Some(se)) if m > se as i64 => (m - se as i64) as usize,
        _ => 0,
    };
    Ok((sum, lost))
}

/// The largest `n` of `Li₋ₙ(z)` by its finite sum ([`polylog_nonpositive`],
/// `polylog::nonpositive`): the row of `n + 1` Stirling numbers takes
/// `O(n²)` operations on integers of up to `n·log₂ n` bits (0.1 s at 1000).
const MAX_NONPOSITIVE_ORDER: u64 = 1000;

thread_local! {
    /// The last row of [`stirling2_row`]: the precision loops of `Li₋ₙ` ask
    /// for the same row at every pass.
    static STIRLING_ROW: std::cell::RefCell<Option<(u64, std::rc::Rc<Vec<BigInt>>)>> =
        const { std::cell::RefCell::new(None) };
}

/// The Stirling numbers of the second kind `S(m, j)`, `j = 0 … m`: one row
/// of the triangle `S(i, j) = j·S(i−1, j) + S(i−1, j−1)`, the integers
/// `crate::domains::combinatorics::stirling2` gives one at a time by the
/// same triangle (`O(m·j)` each).  Before 0.34 the row of `Li₋ₙ` took them
/// one at a time, `O(n³)`: `polylog(−601, 4/13)` 18 s, `polylog(−1401,
/// 4/13)` minutes.  Refused beyond [`MAX_NONPOSITIVE_ORDER`]` + 1`.
fn stirling2_row(m: u64) -> Result<std::rc::Rc<Vec<BigInt>>, SymplexError> {
    if m > MAX_NONPOSITIVE_ORDER + 1 {
        return Err(SymplexError::NotImplemented(format!(
            "polylog of an order below -{MAX_NONPOSITIVE_ORDER} (a finite sum of {m} terms with \
             Stirling numbers)"
        )));
    }
    if let Some(row) = STIRLING_ROW.with(|c| {
        c.borrow()
            .as_ref()
            .filter(|(k, _)| *k == m)
            .map(|(_, r)| std::rc::Rc::clone(r))
    }) {
        return Ok(row);
    }
    let mu = m as usize;
    let mut row = vec![BigInt::zero(); mu + 1];
    row[0] = BigInt::from(1);
    for i in 1..=mu {
        for j in (1..=i).rev() {
            let (lower, upper) = row.split_at_mut(j);
            upper[0] *= j as u64;
            upper[0] += &lower[j - 1];
        }
        row[0] = BigInt::zero();
    }
    let row = std::rc::Rc::new(row);
    STIRLING_ROW.with(|c| *c.borrow_mut() = Some((m, std::rc::Rc::clone(&row))));
    Ok(row)
}

/// `Li_{−n}(z)` for integer `n ≥ 0` and any real `z ≠ 1`:
/// `Σ_{k=0}^{n} k! S(n+1, k+1) (z/(1−z))^{k+1}`, at `wp` bits, and `log₂` of
/// its largest term.  For `z < 0` the terms alternate and cancel (see
/// [`polylog_nonpositive_measured`]).
fn polylog_nonpositive(n: usize, z: &BigFloat, wp: usize) -> Result<(BigFloat, f64), SymplexError> {
    let one = BigFloat::from_i32(1, wp);
    let one_minus_z = one.sub(z, wp, RM);
    if one_minus_z.is_zero() {
        return Err(unevaluable("polylog(s, 1) diverges for s ≤ 1"));
    }
    let w = z.div(&one_minus_z, wp, RM);
    let row = stirling2_row(n as u64 + 1)?;
    let mut k_fact = BigInt::from(1);
    let mut w_pow = w.clone();
    let mut sum = BigFloat::new(wp);
    let mut largest = f64::NEG_INFINITY;
    for k in 0..=n {
        if k > 0 {
            k_fact *= BigInt::from(k as u64);
            w_pow = w_pow.mul(&w, wp, RM);
        }
        let c = ratio_to_bigfloat(&Ratio::from_integer(&k_fact * &row[k + 1]), wp);
        let term = c.mul(&w_pow, wp, RM);
        if let Some(e) = term.exponent().filter(|_| !term.is_zero()) {
            largest = largest.max(f64::from(e));
        }
        sum = sum.add(&term, wp, RM);
    }
    Ok((sum, largest))
}

/// [`polylog_nonpositive`] to `prec` bits: the loss of the alternating sum
/// of a negative `z` (the largest term over the value) measured and made up
/// with more bits, up to [`cancellation_cap`]; the first pass, at `prec +
/// 32` bits, as before 0.34.  Before 0.34 it went unmeasured:
/// `polylog(−200, −1/2)` was `2.03·10²⁹⁶`, truly `−5.15·10²⁷²`.
fn polylog_nonpositive_measured(
    n: usize,
    z: &BigFloat,
    prec: usize,
) -> Result<BigFloat, SymplexError> {
    let cap = cancellation_cap(prec);
    let mut extra = 32;
    loop {
        let (value, largest) = polylog_nonpositive(n, z, prec + extra)?;
        let lost = match value.exponent() {
            Some(e) if !value.is_zero() && largest.is_finite() => {
                (largest - f64::from(e)).max(0.0).ceil() as usize
            }
            _ if largest.is_finite() => prec + extra,
            _ => 0,
        };
        if lost + 16 <= extra {
            return Ok(round_to(value, prec));
        }
        if extra >= cap || lost + 16 > cap {
            return Err(special_exhausted(prec));
        }
        extra = (lost + 32).max(2 * extra).min(cap);
    }
}

/// `Li_s(z)` for `1/2 < z < 1` via the expansion in `μ = ln z` (`|μ| < 2π`):
///
/// * integer `s = n ≥ 1`:
///   `Li_n(e^μ) = μ^{n−1}/(n−1)! (H_{n−1} − ln(−μ)) + Σ_{k≥0, k≠n−1} ζ(n−k) μ^k/k!`
/// * non-integer `s`: `Li_s(e^μ) = Γ(1−s)(−μ)^{s−1} + Σ_{k≥0} ζ(s−k) μ^k/k!`
///
/// For `s` next to an integer `n` the first term and the term `k = n − 1`
/// have opposite poles (`Γ(1 − s)` and `ζ(s − n + 1)`, about `1/|s − n|`)
/// that cancel: the loss, the largest term over the sum, is measured and
/// the expansion repeated with that many more bits (up to
/// [`cancellation_cap`]).  Before 0.30 it went unmeasured, and
/// `polylog(2 + 10⁻²⁰, 999/1000)` was wrong from its 20th digit at 60
/// digits (`polylog(1 + 10⁻³⁰, 95/100)` from its 29th).
fn polylog_near_one(
    s: &BigFloat,
    s_int: Option<i64>,
    z: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let wp = w.bits();
    // The caller's `wp` carries 32 guard bits; a loss within 24 of them
    // costs nothing.
    let cap = cancellation_cap(wp);
    let mut extra = 0usize;
    loop {
        let p = wp + extra;
        let (sum, largest) = polylog_near_one_at(s, s_int, z, &mut w.at(p))?;
        let lost = match (largest, sum.exponent()) {
            (Some(l), Some(e)) if !sum.is_zero() => usize::try_from(l - i64::from(e)).unwrap_or(0),
            (Some(_), _) => p,
            _ => 0,
        };
        if lost <= extra + 24 {
            return Ok(round_to(sum, wp));
        }
        if extra >= cap {
            return Err(special_exhausted(wp));
        }
        extra = (lost + 16).max(2 * extra).min(cap);
    }
}

/// `ζ(σ₁), ζ(σ₁ − 1), ζ(σ₁ − 2), …` for a non-integer `σ₁ < −1`: the
/// terms of the `μ`-expansion of [`polylog_near_one`] left of 0, where each
/// used to cost a whole `ζ` evaluation (a `Γ` by Stirling's series and a
/// Borwein sum with its exact coefficients rebuilt: `polylog(1/3, −1 +
/// 10⁻¹²)` spent 0.35 s on some 25 of them at 60 digits).  By the
/// functional equation `ζ(σ) = 2^σ π^{σ−1} sin(πσ/2) Γ(1−σ) ζ(1−σ)`
/// every factor follows from the previous one: `2^σ π^{σ−1}` divided by
/// `2π`, the sine a quarter period on, `Γ(t)` times `t` (`t = 1 − σ`), and
/// `ζ(t) = η(t)/(1 − 2^{1−t})` for `t ≥ 2` by one Borwein sum (Algorithm 2
/// of the paper cited at [`arb_eta_borwein`]) whose coefficients are built
/// once and whose powers `m^{−t}` are divided by `m` from one `t` to the
/// next.  The working precision carries 16 guard bits for the roundings
/// that accumulate over the steps.
struct ZetaLeft {
    wp: usize,
    /// `2^σ π^{σ−1}`.
    scale: BigFloat,
    inv_two_pi: BigFloat,
    /// `sin(πσ/2)`, `cos(πσ/2)`.
    sin: BigFloat,
    cos: BigFloat,
    /// `Γ(t)` and `t = 1 − σ`.
    gamma: BigFloat,
    t: BigFloat,
    /// Borwein's `(−1)^k (d_k − d_n)/d_n`, `k < n`.
    coeffs: Vec<BigFloat>,
    /// `m^{−t}`, `m = 1 … n` (index `m − 1`).
    powers: Vec<BigFloat>,
    /// `1/m`.
    inverses: Vec<BigFloat>,
    /// `2^{1−t}`.
    two_pow: BigFloat,
}

impl ZetaLeft {
    fn new(sigma: &BigFloat, w: &mut Wp<'_>) -> Result<Self, SymplexError> {
        let wp = w.bits();
        let wp = wp + 16;
        let one = BigFloat::from_i32(1, 64);
        let t = one.sub(sigma, exact_bits(sigma, wp), RM);
        let pi = w.cc.pi(wp, RM).clone();
        let two = BigFloat::from_i32(2, 64);
        let scale = bf_pow(&two, sigma, &mut w.at(wp)).mul(
            &bf_pow(&pi, &sigma.sub(&one, wp, RM), &mut w.at(wp)),
            wp,
            RM,
        );
        let half_pi_sigma = pi.mul(sigma, wp, RM).div(&two, wp, RM);
        let gamma = arb_gamma_real(&t, &mut w.at(wp))?;
        let n =
            ((wp as f64) * std::f64::consts::LN_2 / (3.0 + 8f64.sqrt()).ln()).ceil() as usize + 4;
        let mut d: Vec<Ratio<BigInt>> = Vec::with_capacity(n + 1);
        let mut acc = Ratio::<BigInt>::zero();
        let mut term = Ratio::from_integer(BigInt::from(1));
        for i in 0..=n {
            if i > 0 {
                let numer = BigInt::from((n + i - 1) as u64)
                    * BigInt::from(4u64)
                    * BigInt::from((n - i + 1) as u64);
                let denom = BigInt::from((2 * i - 1) as u64) * BigInt::from((2 * i) as u64);
                term *= Ratio::new(numer, denom);
            }
            acc += &term;
            d.push(acc.clone());
        }
        let coeffs = (0..n)
            .map(|k| {
                let c = ratio_to_bigfloat(&((&d[k] - &d[n]) / &d[n]), wp);
                if k % 2 == 0 { c } else { c.neg() }
            })
            .collect();
        // m^{−t}: a power for each prime, products for the composites.
        let neg_t = t.neg();
        let mut spf = vec![0usize; n + 1];
        let mut powers: Vec<BigFloat> = Vec::with_capacity(n);
        for m in 1..=n {
            if m >= 2 && spf[m] == 0 {
                for multiple in (m..=n).step_by(m) {
                    if spf[multiple] == 0 {
                        spf[multiple] = m;
                    }
                }
            }
            let v = if m == 1 {
                BigFloat::from_i32(1, wp)
            } else if spf[m] == m {
                bf_pow(&BigFloat::from_u64(m as u64, 64), &neg_t, &mut w.at(wp))
            } else {
                powers[spf[m] - 1].mul(&powers[m / spf[m] - 1], wp, RM)
            };
            powers.push(v);
        }
        let inverses = (1..=n)
            .map(|m| one.div(&BigFloat::from_u64(m as u64, 64), wp, RM))
            .collect();
        let two_pow = bf_pow(&two, &one.sub(&t, wp, RM), &mut w.at(wp));
        Ok(ZetaLeft {
            wp,
            scale,
            inv_two_pi: one.div(&pi.mul(&two, wp, RM), wp, RM),
            sin: half_pi_sigma.sin(wp, RM, w.cc),
            cos: half_pi_sigma.cos(wp, RM, w.cc),
            gamma,
            t,
            coeffs,
            powers,
            inverses,
            two_pow,
        })
    }

    /// `ζ(σ)` at the current `σ`, then a step to `σ − 1`.
    fn next(&mut self) -> BigFloat {
        let wp = self.wp;
        let mut eta = BigFloat::new(wp);
        for (c, p) in self.coeffs.iter().zip(self.powers.iter()) {
            eta = eta.add(&c.mul(p, wp, RM), wp, RM);
        }
        let one = BigFloat::from_i32(1, 64);
        let zeta_t = eta.neg().div(&one.sub(&self.two_pow, wp, RM), wp, RM);
        let value = self
            .scale
            .mul(&self.sin, wp, RM)
            .mul(&self.gamma, wp, RM)
            .mul(&zeta_t, wp, RM);
        // σ → σ − 1, t → t + 1.
        self.scale = self.scale.mul(&self.inv_two_pi, wp, RM);
        let (s, c) = (self.cos.neg(), self.sin.clone());
        self.sin = s;
        self.cos = c;
        self.gamma = self.gamma.mul(&self.t, wp, RM);
        self.t = self.t.add(&one, wp, RM);
        for (p, inv) in self.powers.iter_mut().zip(self.inverses.iter()) {
            *p = p.mul(inv, wp, RM);
        }
        self.two_pow = self.two_pow.div(&BigFloat::from_i32(2, 64), wp, RM);
        value
    }
}

/// One evaluation of [`polylog_near_one`] at working precision `wp`, and
/// the binary exponent of the largest term summed.
fn polylog_near_one_at(
    s: &BigFloat,
    s_int: Option<i64>,
    z: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<(BigFloat, Option<i64>), SymplexError> {
    let wp = w.bits();
    let mut largest: Option<i64> = None;
    let mut note = |v: &BigFloat| {
        if !v.is_zero()
            && let Some(e) = v.exponent()
        {
            largest = Some(largest.map_or(i64::from(e), |m| m.max(i64::from(e))));
        }
    };
    let one = BigFloat::from_i32(1, wp);
    let mu = z.ln(wp, RM, w.cc); // negative
    let neg_mu = mu.neg();
    // The terms decay like (|μ|/2π)^k (|ζ(−m)| ~ m!/(2π)^m), |μ| < ln 2.
    let max_terms = wp * 2 + 100;
    let not_converged = || series_did_not_converge("polylog", max_terms);
    let mut sum = BigFloat::new(wp);
    let mut mu_pow_over_fact = one.clone(); // μ^k/k!
    // ζ vanishes at the negative even integers, so a single negligible
    // term proves nothing: stop only after two in a row.
    let mut small_run = 0u8;
    let mut converged = false;
    match s_int {
        Some(n) if n >= 1 => {
            let n = n as usize;
            for k in 0..max_terms {
                if k > 0 {
                    mu_pow_over_fact = mu_pow_over_fact.mul(&mu, wp, RM).div(
                        &BigFloat::from_i128(k as i128, wp),
                        wp,
                        RM,
                    );
                }
                if k + 1 == n {
                    // μ^{n−1}/(n−1)! (H_{n−1} − ln(−μ))
                    let mut h = BigFloat::new(wp);
                    for j in 1..n {
                        h = h.add(
                            &one.div(&BigFloat::from_i128(j as i128, wp), wp, RM),
                            wp,
                            RM,
                        );
                    }
                    let ln_neg_mu = neg_mu.ln(wp, RM, w.cc);
                    let t = mu_pow_over_fact.mul(&h.sub(&ln_neg_mu, wp, RM), wp, RM);
                    note(&t);
                    sum = sum.add(&t, wp, RM);
                    continue;
                }
                let arg = n as i64 - k as i64;
                let zeta = if arg <= 0 {
                    zeta_nonpositive_int(arg, wp)
                } else {
                    arb_zeta(&BigFloat::from_i128(arg as i128, wp), w)?
                };
                let term = zeta.mul(&mu_pow_over_fact, wp, RM);
                note(&term);
                sum = sum.add(&term, wp, RM);
                if k >= n {
                    if negligible(&term, &sum, wp) {
                        small_run += 1;
                        if small_run >= 2 {
                            converged = true;
                            break;
                        }
                    } else {
                        small_run = 0;
                    }
                }
            }
            if !converged {
                return Err(not_converged());
            }
            Ok((sum, largest))
        }
        _ => {
            // `1 − s` exact, as `s − k` below: next to an integer the poles of
            // `Γ(1 − s)` and `ζ(s − n + 1)` cancel only for the same `s` (a
            // rounded `1 − s` moves `1/|s − n|` by its rounding over
            // `(s − n)²`).
            let one_minus_s = BigFloat::from_i32(1, 64).sub(s, exact_bits(s, wp), RM);
            let g = arb_gamma_real(&one_minus_s, w)?;
            let s_minus_1 = s.sub(&one, wp, RM);
            let lead = g.mul(&bf_pow(&neg_mu, &s_minus_1, w), wp, RM);
            note(&lead);
            let minus_one = BigFloat::from_i32(-1, 64);
            let mut left: Option<ZetaLeft> = None;
            for k in 0..max_terms {
                if k > 0 {
                    mu_pow_over_fact = mu_pow_over_fact.mul(&mu, wp, RM).div(
                        &BigFloat::from_i128(k as i128, wp),
                        wp,
                        RM,
                    );
                }
                let arg = s.sub(&BigFloat::from_i128(k as i128, wp), exact_bits(s, wp), RM);
                // Left of −1 from the sequence (see [`ZetaLeft`]).
                let zeta = if left.is_some() || bf_lt(&arg, &minus_one) {
                    if left.is_none() {
                        left = Some(ZetaLeft::new(&arg, w)?);
                    }
                    match left.as_mut() {
                        Some(l) => l.next(),
                        None => arb_zeta(&arg, w)?,
                    }
                } else {
                    arb_zeta(&arg, w)?
                };
                let term = zeta.mul(&mu_pow_over_fact, wp, RM);
                note(&term);
                sum = sum.add(&term, wp, RM);
                if k >= 2 {
                    if negligible(&term, &sum, wp) {
                        small_run += 1;
                        if small_run >= 2 {
                            converged = true;
                            break;
                        }
                    } else {
                        small_run = 0;
                    }
                }
            }
            if !converged {
                return Err(not_converged());
            }
            Ok((lead.add(&sum, wp, RM), largest))
        }
    }
}

/// Largest `|z|` for which the direct series is used.  For integer `s` the
/// `μ`-expansion is cheap (Bernoulli numbers), so it takes over at `1/2`;
/// for non-integer `s` its terms `ζ(s − k)` come from one incremental
/// sequence left of −1 ([`ZetaLeft`]), a few multiplications per Borwein
/// term, and it takes over at `3/4`, where the direct series needs
/// `wp·ln 2/ln(4/3)` terms of a non-integer power each.  (Before 0.30 every
/// `ζ(s − k)` was a full evaluation and the direct series ran up to `0.95`:
/// 1,800 powers for `polylog(1/3, −95/100)` at 60 digits, 0.57 s.)
fn polylog_series_limit(s_int: Option<i64>, wp: usize) -> BigFloat {
    if s_int.is_some() {
        BigFloat::from_f64(0.5, wp)
    } else {
        BigFloat::from_f64(0.75, wp)
    }
}

/// `Li_s(z)` for `0 < |z| < 1`.
fn polylog_unit_interval(
    s: &BigFloat,
    s_int: Option<i64>,
    z: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let wp = w.bits();
    let limit = polylog_series_limit(s_int, wp);
    let az = z.abs();
    // A large order: `k^(−s) < 2^(−wp)` from `k = 16` on, the direct series
    // ends after a few terms wherever `abs(z) < 1`.  Before 0.37 the expansion
    // in `ln z` took over beyond `abs(z) = ½` and summed `ζ(s − k)` for every
    // `k < s`: `polylog(1038, −9/10)` was refused ("did not converge").
    let large = f64_of(s).is_ok_and(|v| v >= wp as f64 / 4.0 + 8.0);
    if large || !bf_gt(&az, &limit) {
        return polylog_series(s, s_int, z, w);
    }
    if z.is_negative() {
        // Li_s(−x) = 2^{1−s} Li_s(x²) − Li_s(x).  The difference cancels
        // where Li_s(−x) is small next to its terms (next to a zero of
        // η(s) = −Li_s(−1), s ≈ −2, −4, …): the loss is measured and the
        // difference formed again with that many more bits.
        let cap = cancellation_cap(wp);
        let mut extra = 0usize;
        loop {
            let p = wp + extra;
            let one = BigFloat::from_i32(1, p);
            let x2 = az.mul(&az, 2 * p, RM);
            let a = polylog_unit_interval(s, s_int, &x2, &mut w.at(p))?;
            let b = polylog_near_one(s, s_int, &az, &mut w.at(p))?;
            let two = BigFloat::from_i32(2, p);
            let one_minus_s = one.sub(s, exact_bits(s, p), RM);
            let scale = bf_pow(&two, &one_minus_s, &mut w.at(p));
            let first = scale.mul(&a, p, RM);
            let r = first.sub(&b, p, RM);
            let largest = first.exponent().max(b.exponent()).map_or(0, i64::from);
            let lost = match r.exponent() {
                Some(e) if !r.is_zero() => usize::try_from(largest - i64::from(e)).unwrap_or(0),
                _ => p,
            };
            if lost <= extra + 24 {
                return Ok(round_to(r, wp));
            }
            if extra >= cap {
                return Err(special_exhausted(wp));
            }
            extra = (lost + 16).max(2 * extra).min(cap);
        }
    }
    polylog_near_one(s, s_int, z, w)
}

/// Polylogarithm `Li_s(z)` for real `s` and real `−1 ≤ z ≤ 1` (any real
/// `z ≠ 1` when `s` is a non-positive integer, where `Li_s` is rational).
/// A non-real `z` or `|z| > 1` is `polylog::polylog_general`.
fn arb_polylog(s: &BigFloat, z: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    if s.is_nan() || z.is_nan() || s.is_inf() || z.is_inf() {
        return Err(unevaluable("polylog of special float value"));
    }
    if z.is_zero() {
        return Ok(BigFloat::new(prec));
    }
    let wp = prec + 32;
    let one = BigFloat::from_i32(1, wp);
    let s_int = bf_as_int(s)?;
    if let Some(n) = s_int
        && n <= 0
    {
        // Beyond the Stirling sum: the poles of `polylog::nonpositive_poles`,
        // their cancellation measured by `polylog_general` (up to its
        // largest order; beyond, the Stirling sum refuses).
        if n.unsigned_abs() > MAX_NONPOSITIVE_ORDER && n.unsigned_abs() <= 1_000_000 {
            let v = polylog::polylog_general(s, &(z.clone(), BigFloat::new(prec)), w)?;
            return Ok(round_to(v.0, prec));
        }
        return polylog_nonpositive_measured(n.unsigned_abs() as usize, z, prec);
    }
    let az = z.abs();
    if bf_gt(&az, &one) {
        return Err(unevaluable(
            "polylog outside the unit disc not yet supported in evalf",
        ));
    }
    if az.sub(&one, wp, RM).is_zero() {
        // z = ±1
        if z.is_negative() {
            // Li_s(−1) = −η(s), entire in s.
            let eta = arb_eta(s, &mut w.at(wp))?;
            return Ok(round_to(eta.neg(), prec));
        }
        if !bf_gt(s, &one) {
            return Err(unevaluable("polylog(s, 1) diverges for s ≤ 1"));
        }
        let zeta = arb_zeta(s, &mut w.at(wp))?;
        return Ok(round_to(zeta, prec));
    }
    let r = polylog_unit_interval(s, s_int, z, &mut w.at(wp))?;
    Ok(round_to(r, prec))
}

/// Dirichlet eta `η(s)` at working precision `wp` for real `s`: Borwein's
/// alternating sum ([`arb_eta_borwein`]) for `s > 0` — `η` is entire, so
/// nothing special happens near `s = 1` (`η(1) = ln 2`) — and
/// `(1 − 2^{1−s}) ζ(s)` for `s ≤ 0`.
fn arb_eta(s: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let wp = w.bits();
    // Strictly: `+0` has the sign bit of a positive number, and Borwein's sum at
    // `s = 0` is not `η(0) = 1/2` (before 0.30 `dirichlet_eta` of an argument
    // that cancelled to `+0` printed `0.4913903202446295`).
    if bf_strictly_positive(s) {
        return arb_eta_borwein(s, w);
    }
    let one = BigFloat::from_i32(1, wp);
    let two = BigFloat::from_i32(2, wp);
    let one_minus_s = one.sub(s, wp, RM);
    let coeff = one.sub(&bf_pow(&two, &one_minus_s, w), wp, RM);
    let zeta = arb_zeta(s, w)?;
    Ok(coeff.mul(&zeta, wp, RM))
}

/// Dirichlet eta `η(s)` (see [`arb_eta`]), with `η(+∞) = 1`.
fn arb_dirichlet_eta(s: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    if s.is_nan() || s.is_inf_neg() {
        return Err(unevaluable("dirichlet_eta of special float value"));
    }
    if s.is_inf_pos() {
        return Ok(BigFloat::from_i32(1, prec));
    }
    let wp = prec + 32;
    let mut sw = s.clone();
    let _ = sw.set_precision(wp, RM);
    Ok(round_to(arb_eta(&sw, &mut w.at(wp))?, prec))
}

// ── Airy functions ─────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum AiryKind {
    Ai,
    Bi,
    AiPrime,
    BiPrime,
}

/// Airy functions for real `x`.
///
/// * `|x|^{3/2} ≤ (3/4)·wp·ln 2`: Maclaurin series (DLMF 9.4.1–9.4.4)
///   `Ai = c₁ f − c₂ g`, `Bi = √3 (c₁ f + c₂ g)` with
///   `f = Σ x^{3k}/∏_{j≤k}(3j−1)(3j)`, `g = Σ x^{3k+1}/∏_{j≤k}(3j)(3j+1)`,
///   `c₁ = 3^{−2/3}/Γ(2/3)`, `c₂ = 3^{−1/3}/Γ(1/3)`; the working precision
///   carries `(4/3)|x|^{3/2} log₂ e` guard bits for the cancellation.
/// * beyond, with `ζ = (2/3)|x|^{3/2}`: the asymptotic expansions
///   DLMF 9.7.5–9.7.12 (`u_k`, `v_k = −(6k+1)/(6k−1) u_k`), whose optimal
///   truncation error is `~e^{−2ζ}`.
fn arb_airy(x: &BigFloat, kind: AiryKind, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    if x.is_nan() || x.is_inf() {
        return Err(unevaluable("Airy function of special float value"));
    }
    let base_wp = prec + 32;
    let x_f = f64_of(x)?;
    let ax_f = x_f.abs();
    let zeta_f = 2.0 / 3.0 * ax_f.powf(1.5);

    if 2.0 * zeta_f > (base_wp as f64) * std::f64::consts::LN_2 {
        // The phase `(2/3)|x|^{3/2}` needs its bits before the point.
        let z_bits = 3 * x.exponent().map_or(0, i64::from).max(0) / 2;
        if !zeta_f.is_finite() || z_bits > i64::try_from(cancellation_cap(prec)).unwrap_or(i64::MAX)
        {
            return Err(special_exhausted(prec));
        }
        return Ok(round_to(airy_asymptotic(x, kind, &mut w.at(base_wp)), prec));
    }

    let wp = Prec::new(base_wp)
        .guarded(
            Guard::for_log2(2.0 * zeta_f * std::f64::consts::LOG2_E),
            Prec::MAX,
        )
        .ok_or_else(|| special_exhausted(prec))?
        .plus(16)
        .bits();
    let mut xw = x.clone();
    let _ = xw.set_precision(wp, RM);
    let one = BigFloat::from_i32(1, wp);
    let three = BigFloat::from_i32(3, wp);
    let third = one.div(&three, wp, RM);
    let two_thirds = third.mul(&BigFloat::from_i32(2, wp), wp, RM);
    let g13 = arb_gamma_real(&third, &mut w.at(wp))?;
    let g23 = arb_gamma_real(&two_thirds, &mut w.at(wp))?;
    let c1 = one.div(
        &bf_pow(&three, &two_thirds, &mut w.at(wp)).mul(&g23, wp, RM),
        wp,
        RM,
    );
    let c2 = one.div(
        &bf_pow(&three, &third, &mut w.at(wp)).mul(&g13, wp, RM),
        wp,
        RM,
    );

    // Series with term recurrences; the derivatives use
    // f'_k = 3k f_k / x, g'_k = (3k+1) g_k / x (k = 0 handled directly).
    let x3 = xw.mul(&xw, wp, RM).mul(&xw, wp, RM);
    let derivative = matches!(kind, AiryKind::AiPrime | AiryKind::BiPrime);
    let (mut f_sum, mut g_sum) = if derivative {
        (BigFloat::new(wp), one.clone()) // f'(0) = 0, g'(0) = 1
    } else {
        (one.clone(), xw.clone()) // f(0) = 1, g_0 = x
    };
    let mut fk = one.clone(); // x^{3k}/∏(3j−1)(3j)
    let mut gk = xw.clone(); // x^{3k+1}/∏(3j)(3j+1)
    // At x = 0 the k = 0 terms above are the exact values (every later
    // term vanishes, and the derivative recurrences would divide by x).
    let max_terms = if xw.is_zero() {
        0
    } else {
        (ax_f * 2.0) as usize + wp + 40
    };
    for k in 1..=max_terms {
        let k3 = 3 * k as i128;
        fk = fk
            .mul(&x3, wp, RM)
            .div(&BigFloat::from_i128((k3 - 1) * k3, wp), wp, RM);
        gk = gk
            .mul(&x3, wp, RM)
            .div(&BigFloat::from_i128(k3 * (k3 + 1), wp), wp, RM);
        let (tf, tg) = if derivative {
            // 3k x^{3k−1}/∏ = (3k/x) f_k ;  (3k+1) x^{3k}/∏ = ((3k+1)/x) g_k
            (
                fk.mul(&BigFloat::from_i128(k3, wp), wp, RM)
                    .div(&xw, wp, RM),
                gk.mul(&BigFloat::from_i128(k3 + 1, wp), wp, RM)
                    .div(&xw, wp, RM),
            )
        } else {
            (fk.clone(), gk.clone())
        };
        f_sum = f_sum.add(&tf, wp, RM);
        g_sum = g_sum.add(&tg, wp, RM);
        if negligible(&tf, &one, wp + 4) && negligible(&tg, &one, wp + 4) {
            break;
        }
    }
    let a = c1.mul(&f_sum, wp, RM);
    let b = c2.mul(&g_sum, wp, RM);
    let r = match kind {
        AiryKind::Ai | AiryKind::AiPrime => a.sub(&b, wp, RM),
        AiryKind::Bi | AiryKind::BiPrime => three.sqrt(wp, RM).mul(&a.add(&b, wp, RM), wp, RM),
    };
    Ok(round_to(r, prec))
}

/// Asymptotic expansions of the Airy functions for large `|x|`
/// (see [`arb_airy`]).  Returns the value at working precision `wp`.
fn airy_asymptotic(x: &BigFloat, kind: AiryKind, w: &mut Wp<'_>) -> BigFloat {
    let wp = w.bits();
    let one = BigFloat::from_i32(1, wp);
    let two = BigFloat::from_i32(2, wp);
    let ax = x.abs();
    let sqrt_ax = ax.sqrt(wp, RM);
    let x_quarter = sqrt_ax.sqrt(wp, RM); // |x|^{1/4}
    // `ζ = (2/3)|x|^{3/2}` enters `e^{±ζ}` and the phase `ζ + π/4` through
    // its absolute error: it is formed with the `Z` bits it has before the
    // point as guard (the caller refuses beyond `cancellation_cap`).
    // Before 0.30 it was formed at `wp` bits: `airyai(−467·10⁴⁰)` was wrong
    // from its 13th digit at 30, `airyai(−29/16·10⁷⁰)` entirely.
    let wz =
        wp + usize::try_from(3 * ax.exponent().map_or(0, i64::from).max(0) / 2 + 8).unwrap_or(0);
    let ax_z = at_precision(&ax, wz);
    let zeta = two.mul(&ax_z, wz, RM).mul(&ax_z.sqrt(wz, RM), wz, RM).div(
        &BigFloat::from_i32(3, wz),
        wz,
        RM,
    );
    let inv_zeta = one.div(&zeta, wp, RM);
    let sqrt_pi = w.cc.pi(wp, RM).clone().sqrt(wp, RM);
    let derivative = matches!(kind, AiryKind::AiPrime | AiryKind::BiPrime);

    // Coefficients c_k = u_k (or v_k) / ζ^k, u_k = u_{k−1}(6k−5)(6k−3)(6k−1)/(216 k (2k−1)),
    // v_k = −(6k+1)/(6k−1) u_k.  Optimal truncation: stop when |c_k| grows.
    let mut coeffs: Vec<BigFloat> = Vec::new();
    let mut u = one.clone();
    let mut zeta_pow = one.clone();
    let mut prev: Option<BigFloat> = None;
    for k in 0..(wp * 2 + 100) {
        if k > 0 {
            let k_i = k as i128;
            let num = BigFloat::from_i128((6 * k_i - 5) * (6 * k_i - 3) * (6 * k_i - 1), wp);
            let den = BigFloat::from_i128(216 * k_i * (2 * k_i - 1), wp);
            u = u.mul(&num, wp, RM).div(&den, wp, RM);
            zeta_pow = zeta_pow.mul(&inv_zeta, wp, RM);
        }
        let mut c = u.mul(&zeta_pow, wp, RM);
        if derivative {
            let k_i = k as i128;
            let ratio = BigFloat::from_i128(6 * k_i + 1, wp).div(
                &BigFloat::from_i128(6 * k_i - 1, wp),
                wp,
                RM,
            );
            c = c.mul(&ratio, wp, RM).neg();
        }
        if let Some(ref p) = prev
            && bf_gt(&c.abs(), p)
        {
            break;
        }
        let done = negligible(&c, &one, wp);
        prev = Some(c.abs());
        coeffs.push(c);
        if done {
            break;
        }
    }

    if !x.is_negative() {
        // Ai ~ e^{−ζ}/(2√π x^{1/4}) Σ (−1)^k u_k/ζ^k,   Bi ~ e^{ζ}/(√π x^{1/4}) Σ u_k/ζ^k
        // Ai' ~ −x^{1/4} e^{−ζ}/(2√π) Σ (−1)^k v_k/ζ^k, Bi' ~ x^{1/4} e^{ζ}/√π Σ v_k/ζ^k
        let mut sum = BigFloat::new(wp);
        let alternate = matches!(kind, AiryKind::Ai | AiryKind::AiPrime);
        for (k, c) in coeffs.iter().enumerate() {
            if alternate && k % 2 == 1 {
                sum = sum.sub(c, wp, RM);
            } else {
                sum = sum.add(c, wp, RM);
            }
        }
        let amp = match kind {
            AiryKind::Ai => zeta.neg().exp(wp, RM, w.cc).div(
                &two.mul(&sqrt_pi, wp, RM).mul(&x_quarter, wp, RM),
                wp,
                RM,
            ),
            AiryKind::Bi => zeta
                .exp(wp, RM, w.cc)
                .div(&sqrt_pi.mul(&x_quarter, wp, RM), wp, RM),
            AiryKind::AiPrime => x_quarter
                .mul(&zeta.neg().exp(wp, RM, w.cc), wp, RM)
                .div(&two.mul(&sqrt_pi, wp, RM), wp, RM)
                .neg(),
            AiryKind::BiPrime => x_quarter
                .mul(&zeta.exp(wp, RM, w.cc), wp, RM)
                .div(&sqrt_pi, wp, RM),
        };
        return amp.mul(&sum, wp, RM);
    }

    // x < 0: with θ = ζ + π/4, P = Σ (−1)^k c_{2k}, Q = Σ (−1)^k c_{2k+1}:
    // Ai  ~ (sinθ P − cosθ Q)/(√π x^{1/4}),  Bi  ~ (cosθ P + sinθ Q)/(√π x^{1/4})
    // Ai' ~ −x^{1/4}(cosθ P + sinθ Q)/√π,  Bi' ~  x^{1/4}(sinθ P − cosθ Q)/√π
    let mut p = BigFloat::new(wp);
    let mut q = BigFloat::new(wp);
    for (k, c) in coeffs.iter().enumerate() {
        let sign_neg = (k / 2) % 2 == 1;
        let target = if k % 2 == 0 { &mut p } else { &mut q };
        *target = if sign_neg {
            target.sub(c, wp, RM)
        } else {
            target.add(c, wp, RM)
        };
    }
    let pi = w.cc.pi(wz, RM).clone();
    let theta = zeta.add(&pi.div(&BigFloat::from_i32(4, wz), wz, RM), wz, RM);
    let s = theta.sin(wp, RM, w.cc);
    let c = theta.cos(wp, RM, w.cc);
    let sp_minus_cq = s.mul(&p, wp, RM).sub(&c.mul(&q, wp, RM), wp, RM);
    let cp_plus_sq = c.mul(&p, wp, RM).add(&s.mul(&q, wp, RM), wp, RM);
    match kind {
        AiryKind::Ai => sp_minus_cq.div(&sqrt_pi.mul(&x_quarter, wp, RM), wp, RM),
        AiryKind::Bi => cp_plus_sq.div(&sqrt_pi.mul(&x_quarter, wp, RM), wp, RM),
        AiryKind::AiPrime => x_quarter
            .mul(&cp_plus_sq, wp, RM)
            .div(&sqrt_pi, wp, RM)
            .neg(),
        AiryKind::BiPrime => x_quarter.mul(&sp_minus_cq, wp, RM).div(&sqrt_pi, wp, RM),
    }
}

// ── Elliptic integrals (Carlson symmetric forms) ────────────────────────────────

/// Are `a`, `b`, `c` (and `d`) equal to within `2^{−bits}` relative?
fn all_close(vals: &[&BigFloat], bits: usize, wp: usize) -> bool {
    let first = vals[0];
    vals[1..]
        .iter()
        .all(|v| negligible(&v.sub(first, wp, RM), first, bits))
}

/// Convergence threshold for the Carlson duplication iterations: the
/// remainder of `R_F`, `R_D`, `R_J` at arguments within relative spread `δ`
/// of their mean is `O(δ²)` (the Taylor expansions have no linear term), so
/// a spread of `2^{−wp/2}` already gives `wp` correct bits — and stopping
/// there avoids stalling on rounding noise in the last bits.
fn carlson_bits(wp: usize) -> usize {
    wp / 2 + 8
}

/// Carlson `R_C(x, y) = R_F(x, y, y)`, `x ≥ 0`, `y > 0`.
///
/// With `w = (y − x)/y`, `R_C = y^{−1/2} arcsin(√w)/√w = y^{−1/2} Σ_k C(2k,k) w^k / (4^k (2k+1))`
/// (the series is analytic through `w = 0`, covering `x > y` as well).
/// The series is used for `|w| < 1/10`: the closed forms
/// `arccos(√(x/y))/√(y−x)` / `arccosh(√(x/y))/√(x−y)` lose relative
/// precision like `1/w` there, which the `R_J` duplication (where
/// `y − x` shrinks quadratically) would amplify into a first-order error.
fn carlson_rc(x: &BigFloat, y: &BigFloat, wpc: &mut Wp<'_>) -> BigFloat {
    let wp = wpc.bits();
    let one = BigFloat::from_i32(1, wp);
    let diff = y.sub(x, wp, RM);
    let inv_sqrt_y = one.div(&y.sqrt(wp, RM), wp, RM);
    if diff.is_zero() {
        return inv_sqrt_y;
    }
    let w = diff.div(y, wp, RM);
    let w_f = f64_of(&w).unwrap_or(1.0);
    if w_f.abs() < 0.1 {
        // c_k = c_{k−1} (2k−1)² / ((2k)(2k+1)):  1, 1/6, 3/40, 5/112, …
        let mut sum = one.clone();
        let mut c = one.clone();
        let mut wk = one.clone();
        for k in 1..(wp * 2 + 100) {
            let k_i = k as i128;
            let num = BigFloat::from_i128((2 * k_i - 1) * (2 * k_i - 1), wp);
            let den = BigFloat::from_i128(2 * k_i * (2 * k_i + 1), wp);
            c = c.mul(&num, wp, RM).div(&den, wp, RM);
            wk = wk.mul(&w, wp, RM);
            let term = c.mul(&wk, wp, RM);
            sum = sum.add(&term, wp, RM);
            if negligible(&term, &sum, wp) {
                break;
            }
        }
        return inv_sqrt_y.mul(&sum, wp, RM);
    }
    let ratio = x.div(y, wp, RM).sqrt(wp, RM); // √(x/y)
    if is_sign_positive(&diff) {
        // y > x: arccos(√(x/y)) / √(y − x)
        ratio.acos(wp, RM, wpc.cc).div(&diff.sqrt(wp, RM), wp, RM)
    } else {
        // x > y: arccosh(√(x/y)) / √(x − y)
        ratio
            .acosh(wp, RM, wpc.cc)
            .div(&diff.neg().sqrt(wp, RM), wp, RM)
    }
}

/// Carlson `R_F(x, y, z)` (`x, y, z ≥ 0`, at most one zero) by the
/// duplication theorem iterated until the arguments coincide to working
/// precision, where `R_F(μ, μ, μ) = μ^{−1/2}`.
fn carlson_rf(
    x: &BigFloat,
    y: &BigFloat,
    z: &BigFloat,
    wp: usize,
) -> Result<BigFloat, SymplexError> {
    let four = BigFloat::from_i32(4, wp);
    let (mut x, mut y, mut z) = (x.clone(), y.clone(), z.clone());
    for _ in 0..(wp * 2 + 200) {
        if all_close(&[&x, &y, &z], carlson_bits(wp), wp) {
            // R_F(μ, μ, μ) = μ^{−1/2}, μ = (x + y + z)/3
            let one = BigFloat::from_i32(1, wp);
            let mu = x
                .add(&y, wp, RM)
                .add(&z, wp, RM)
                .div(&BigFloat::from_i32(3, wp), wp, RM);
            return Ok(one.div(&mu.sqrt(wp, RM), wp, RM));
        }
        let (sx, sy, sz) = (x.sqrt(wp, RM), y.sqrt(wp, RM), z.sqrt(wp, RM));
        let lambda =
            sx.mul(&sy, wp, RM)
                .add(&sy.mul(&sz, wp, RM), wp, RM)
                .add(&sz.mul(&sx, wp, RM), wp, RM);
        x = x.add(&lambda, wp, RM).div(&four, wp, RM);
        y = y.add(&lambda, wp, RM).div(&four, wp, RM);
        z = z.add(&lambda, wp, RM).div(&four, wp, RM);
    }
    Err(SymplexError::ComputationFailed {
        operation: "evalf",
        reason: "Carlson R_F did not converge".into(),
    })
}

/// Carlson `R_D(x, y, z) = R_J(x, y, z, z)` (`z > 0`) by duplication:
/// `R_D(x,y,z) = R_D(x',y',z')/4 + 3/(√z (z + λ))`, `R_D(μ,μ,μ) = μ^{−3/2}`.
fn carlson_rd(
    x: &BigFloat,
    y: &BigFloat,
    z: &BigFloat,
    wp: usize,
) -> Result<BigFloat, SymplexError> {
    let one = BigFloat::from_i32(1, wp);
    let three = BigFloat::from_i32(3, wp);
    let four = BigFloat::from_i32(4, wp);
    let (mut x, mut y, mut z) = (x.clone(), y.clone(), z.clone());
    let mut acc = BigFloat::new(wp);
    let mut scale = one.clone(); // 4^{−k}
    for _ in 0..(wp * 2 + 200) {
        if all_close(&[&x, &y, &z], carlson_bits(wp), wp) {
            // R_D(μ, μ, μ) = μ^{−3/2}, μ = (x + y + 3z)/5
            let mu = x.add(&y, wp, RM).add(&three.mul(&z, wp, RM), wp, RM).div(
                &BigFloat::from_i32(5, wp),
                wp,
                RM,
            );
            let tail = one.div(&mu.mul(&mu.sqrt(wp, RM), wp, RM), wp, RM);
            return Ok(acc.add(&scale.mul(&tail, wp, RM), wp, RM));
        }
        let (sx, sy, sz) = (x.sqrt(wp, RM), y.sqrt(wp, RM), z.sqrt(wp, RM));
        let lambda =
            sx.mul(&sy, wp, RM)
                .add(&sy.mul(&sz, wp, RM), wp, RM)
                .add(&sz.mul(&sx, wp, RM), wp, RM);
        let term = three.div(&sz.mul(&z.add(&lambda, wp, RM), wp, RM), wp, RM);
        acc = acc.add(&scale.mul(&term, wp, RM), wp, RM);
        scale = scale.div(&four, wp, RM);
        x = x.add(&lambda, wp, RM).div(&four, wp, RM);
        y = y.add(&lambda, wp, RM).div(&four, wp, RM);
        z = z.add(&lambda, wp, RM).div(&four, wp, RM);
    }
    Err(SymplexError::ComputationFailed {
        operation: "evalf",
        reason: "Carlson R_D did not converge".into(),
    })
}

/// Carlson `R_J(x, y, z, p)` for `p > 0` by duplication:
/// `R_J = R_J(x',y',z',p')/4 + 3 R_C(α², β²)` with
/// `α = p(√x + √y + √z) + √x√y√z`, `β = √p (p + λ)`, `R_J(μ,μ,μ,μ) = μ^{−3/2}`.
fn carlson_rj(
    x: &BigFloat,
    y: &BigFloat,
    z: &BigFloat,
    p: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let wp = w.bits();
    if !is_sign_positive(p) {
        return Err(unevaluable(
            "elliptic_pi: the characteristic n must satisfy n < 1 for a real value",
        ));
    }
    let one = BigFloat::from_i32(1, wp);
    let three = BigFloat::from_i32(3, wp);
    let four = BigFloat::from_i32(4, wp);
    let (mut x, mut y, mut z, mut p) = (x.clone(), y.clone(), z.clone(), p.clone());
    let mut acc = BigFloat::new(wp);
    let mut scale = one.clone();
    for _ in 0..(wp * 2 + 200) {
        if all_close(&[&x, &y, &z, &p], carlson_bits(wp), wp) {
            // R_J(μ, μ, μ, μ) = μ^{−3/2}, μ = (x + y + z + 2p)/5
            let two_p = p.mul(&BigFloat::from_i32(2, wp), wp, RM);
            let mu = x.add(&y, wp, RM).add(&z, wp, RM).add(&two_p, wp, RM).div(
                &BigFloat::from_i32(5, wp),
                wp,
                RM,
            );
            let tail = one.div(&mu.mul(&mu.sqrt(wp, RM), wp, RM), wp, RM);
            return Ok(acc.add(&scale.mul(&tail, wp, RM), wp, RM));
        }
        let (sx, sy, sz, sp) = (
            x.sqrt(wp, RM),
            y.sqrt(wp, RM),
            z.sqrt(wp, RM),
            p.sqrt(wp, RM),
        );
        let lambda =
            sx.mul(&sy, wp, RM)
                .add(&sy.mul(&sz, wp, RM), wp, RM)
                .add(&sz.mul(&sx, wp, RM), wp, RM);
        let alpha = p.mul(&sx.add(&sy, wp, RM).add(&sz, wp, RM), wp, RM).add(
            &sx.mul(&sy, wp, RM).mul(&sz, wp, RM),
            wp,
            RM,
        );
        let beta = sp.mul(&p.add(&lambda, wp, RM), wp, RM);
        let rc = carlson_rc(&alpha.mul(&alpha, wp, RM), &beta.mul(&beta, wp, RM), w);
        acc = acc.add(&scale.mul(&three.mul(&rc, wp, RM), wp, RM), wp, RM);
        scale = scale.div(&four, wp, RM);
        x = x.add(&lambda, wp, RM).div(&four, wp, RM);
        y = y.add(&lambda, wp, RM).div(&four, wp, RM);
        z = z.add(&lambda, wp, RM).div(&four, wp, RM);
        p = p.add(&lambda, wp, RM).div(&four, wp, RM);
    }
    Err(SymplexError::ComputationFailed {
        operation: "evalf",
        reason: "Carlson R_J did not converge".into(),
    })
}

/// `1 − m` with a domain check (`m < 1` required for a real `K`, `E`, `Π`).
fn one_minus_m(m: &BigFloat, wp: usize, what: &str) -> Result<BigFloat, SymplexError> {
    let one = BigFloat::from_i32(1, wp);
    let v = one.sub(m, wp, RM);
    if v.is_negative() {
        return Err(unevaluable(format!(
            "{what}: parameter m must satisfy m ≤ 1 for a real value"
        )));
    }
    Ok(v)
}

/// Complete elliptic integral of the first kind `K(m) = R_F(0, 1−m, 1)`.
fn arb_elliptic_k(m: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    if m.is_nan() || m.is_inf() {
        return Err(unevaluable("elliptic_k of special float value"));
    }
    let wp = prec + 32;
    let y = one_minus_m(m, wp, "elliptic_k")?;
    if y.is_zero() {
        return Err(unevaluable("elliptic_k(1) is infinite"));
    }
    let _ = w.cc;
    let r = carlson_rf(&BigFloat::new(wp), &y, &BigFloat::from_i32(1, wp), wp)?;
    Ok(round_to(r, prec))
}

/// Complete elliptic integral of the second kind
/// `E(m) = R_F(0, 1−m, 1) − (m/3) R_D(0, 1−m, 1)`.
fn arb_elliptic_e(m: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    if m.is_nan() || m.is_inf() {
        return Err(unevaluable("elliptic_e of special float value"));
    }
    let wp = prec + 32;
    let one = BigFloat::from_i32(1, wp);
    let y = one_minus_m(m, wp, "elliptic_e")?;
    if y.is_zero() {
        return Ok(BigFloat::from_i32(1, prec));
    }
    let _ = w.cc;
    let zero = BigFloat::new(wp);
    let rf = carlson_rf(&zero, &y, &one, wp)?;
    let rd = carlson_rd(&zero, &y, &one, wp)?;
    let three = BigFloat::from_i32(3, wp);
    let r = rf.sub(&m.mul(&rd, wp, RM).div(&three, wp, RM), wp, RM);
    Ok(round_to(r, prec))
}

/// Incomplete elliptic integral of the first kind
/// `F(φ | m) = sinφ · R_F(cos²φ, 1 − m sin²φ, 1)` for `|φ| ≤ π/2`, extended
/// by `F(φ + kπ | m) = F(φ | m) + 2k K(m)`.
fn arb_elliptic_f(phi: &BigFloat, m: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    if phi.is_nan() || phi.is_inf() || m.is_nan() || m.is_inf() {
        return Err(unevaluable("elliptic_f of special float value"));
    }
    if phi.is_zero() {
        return Ok(BigFloat::new(prec));
    }
    let wp = prec + 32;
    let one = BigFloat::from_i32(1, wp);
    let pi = w.cc.pi(wp, RM).clone();
    // Reduce φ to [−π/2, π/2]: φ = φ₀ + kπ.
    let k_bf = phi
        .div(&pi, wp, RM)
        .add(&BigFloat::from_f64(0.5, wp), wp, RM)
        .floor();
    let phi0 = phi.sub(&k_bf.mul(&pi, wp, RM), wp, RM);
    let s = phi0.sin(wp, RM, w.cc);
    let c = phi0.cos(wp, RM, w.cc);
    let s2 = s.mul(&s, wp, RM);
    let c2 = c.mul(&c, wp, RM);
    let y = one.sub(&m.mul(&s2, wp, RM), wp, RM);
    if !is_sign_positive(&y) {
        return Err(unevaluable(
            "elliptic_f: 1 − m sin²φ must be positive for a real value",
        ));
    }
    let mut r = s.mul(&carlson_rf(&c2, &y, &one, wp)?, wp, RM);
    if !k_bf.is_zero() {
        let k = arb_elliptic_k(m, &mut w.at(wp))?;
        let two = BigFloat::from_i32(2, wp);
        r = r.add(&two.mul(&k_bf, wp, RM).mul(&k, wp, RM), wp, RM);
    }
    Ok(round_to(r, prec))
}

/// Complete elliptic integral of the third kind
/// `Π(n | m) = R_F(0, 1−m, 1) + (n/3) R_J(0, 1−m, 1, 1−n)` for `n < 1`, `m < 1`.
fn arb_elliptic_pi(n: &BigFloat, m: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    if n.is_nan() || n.is_inf() || m.is_nan() || m.is_inf() {
        return Err(unevaluable("elliptic_pi of special float value"));
    }
    let wp = prec + 32;
    let one = BigFloat::from_i32(1, wp);
    let zero = BigFloat::new(wp);
    let y = one_minus_m(m, wp, "elliptic_pi")?;
    if y.is_zero() {
        return Err(unevaluable("elliptic_pi(n, 1) is infinite"));
    }
    let p = one.sub(n, wp, RM);
    let rf = carlson_rf(&zero, &y, &one, wp)?;
    if n.is_zero() {
        return Ok(round_to(rf, prec));
    }
    let rj = carlson_rj(&zero, &y, &one, &p, &mut w.at(wp))?;
    let three = BigFloat::from_i32(3, wp);
    let r = rf.add(&n.mul(&rj, wp, RM).div(&three, wp, RM), wp, RM);
    Ok(round_to(r, prec))
}

// ── Orthogonal polynomials with parameters ───────────────────────────────────────

/// The degree of an orthogonal polynomial as a non-negative integer.
fn poly_degree(n: &BigFloat, what: &str) -> Result<u64, SymplexError> {
    // An integer exactly: before 0.29 a degree within 10⁻¹² of one was
    // taken for it (`jacobi(3 + 10⁻¹³, …)` evaluated the cubic).
    match bf_as_int(n)? {
        Some(d) if n.is_int() && (0..=10_000_000).contains(&d) => Ok(d as u64),
        _ => Err(unevaluable(format!(
            "{what}: degree must be a non-negative integer"
        ))),
    }
}

/// Gegenbauer `C_n^{(a)}(x)`: `(k+1) C_{k+1} = 2(k+a) x C_k − (k+2a−1) C_{k−1}`.
fn arb_gegenbauer(
    n: &BigFloat,
    a: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    let deg = poly_degree(n, "gegenbauer")?;
    let wp = prec + 32 + (deg as f64).log2().ceil().max(0.0) as usize;
    let one = BigFloat::from_i32(1, wp);
    let two = BigFloat::from_i32(2, wp);
    if deg == 0 {
        return Ok(BigFloat::from_i32(1, prec));
    }
    let mut prev = one.clone();
    let mut curr = two.mul(a, wp, RM).mul(x, wp, RM);
    for k in 1..deg {
        let k_bf = BigFloat::from_i128(k as i128, wp);
        let t1 = two
            .mul(&k_bf.add(a, wp, RM), wp, RM)
            .mul(x, wp, RM)
            .mul(&curr, wp, RM);
        let coeff = k_bf.add(&two.mul(a, wp, RM), wp, RM).sub(&one, wp, RM);
        let t2 = coeff.mul(&prev, wp, RM);
        let next = t1
            .sub(&t2, wp, RM)
            .div(&BigFloat::from_i128(k as i128 + 1, wp), wp, RM);
        prev = curr;
        curr = next;
    }
    Ok(round_to(curr, prec))
}

/// The largest degree of [`arb_jacobi`]: its sum takes `O(n)` operations at
/// about `n·log₂(1 + √(1 − x²))` bits: 0.12 s at 2500, 0.6 s at 5000
/// (release), and `O(n²)` with the precision beyond.
const MAX_JACOBI_DEGREE: usize = 5000;

/// Jacobi `P_n^{(a,b)}(x)` via the explicit sum
/// `Σ_s C(n+a, n−s) C(n+b, s) ((x−1)/2)^s ((x+1)/2)^{n−s}`, its
/// cancellation measured: the largest term over the value, made up with as
/// many more bits (mpmath's `hypsum` does the same for the `₂F₁` of its
/// `jacobi`).  The first pass takes `2n` guard bits up to degree 2000 (the
/// same digits as before 0.35, which took them unmeasured — a value next to
/// a zero of `P_n` could lose more), `n·log₂(1 + √(1 − x²))` beyond (the
/// terms' sum over the value's scale for `abs(x) < 1`).  Before 0.35 a degree
/// beyond 2000 was refused: `jacobi(2500, 1/3, 1/5, 1/7)`.
///
/// The coefficients are updated term by term: `∏_{j>s}(a + j)` divided by
/// `a + s` (from the first `s` at which it is nonzero: a negative integer
/// `a = −m` makes the terms below `m` zero), `∏_{j>n−s}(b + j)` multiplied by
/// `b + n − s + 1`, `1/((n−s)!·s!)` by `(n − s + 1)/s`, the powers by
/// `(x−1)/2` and divided by `(x+1)/2`.  Before 0.34 each term recomputed its
/// products and factorials (`O(n²)` operations and exact factorials):
/// `jacobi(1000, 1/3, 1/5, 1/3)` 1.3 s, `jacobi(2000, …)` and up never
/// finished.  Refused beyond [`MAX_JACOBI_DEGREE`].
fn arb_jacobi(
    n: &BigFloat,
    a: &BigFloat,
    b: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    let deg = poly_degree(n, "jacobi")? as usize;
    if deg > MAX_JACOBI_DEGREE {
        return Err(SymplexError::NotImplemented(format!(
            "jacobi of degree {deg} (at most {MAX_JACOBI_DEGREE} in evalf)"
        )));
    }
    if deg == 0 {
        return Ok(BigFloat::from_i32(1, prec));
    }
    // `P_n^(a,b)(−x) = (−1)ⁿ P_n^(b,a)(x)`: 0 at `x = 0` for `a = b` and an
    // odd `n` (whose terms cancel exactly, which no measure of the loss
    // certifies).
    if x.is_zero() && deg % 2 == 1 && a.cmp(b) == Some(0) {
        return Ok(BigFloat::new(prec));
    }
    // The rounding of the sum: each term carries at most `12(n + 2)` roundings
    // relative (running products, powers), the additions `n` of the largest
    // partial sum: below `2^(4 + 2·log₂(n + 2) − wp)` of the largest term.
    let rounding = 4 + 2 * (usize::BITS - (deg + 2).leading_zeros()) as usize;
    // The first pass: `2n` guard bits up to the old maximum (the same digits
    // as before 0.35), beyond the cancellation of `abs(x) < 1` estimated as
    // `n·log₂(1 + √(1 − x²))` (the terms' sum over the value's scale).
    let first_extra = if deg <= 2000 {
        32 + 2 * deg
    } else {
        let xf = f64_of(x)?.abs();
        let est = if xf < 1.0 {
            deg as f64 * (1.0 + (1.0 - xf * xf).sqrt()).log2()
        } else {
            0.0
        };
        40 + rounding + est.ceil() as usize
    };
    // At most the `2n` guard bits of before 0.35 beyond `cancellation_cap`.
    let cap = cancellation_cap(prec) + 2 * deg + 64;
    let mut extra = first_extra;
    loop {
        let wp = prec + extra;
        let (sum, largest) = jacobi_sum(deg, a, b, x, &mut w.at(wp))?;
        // The loss: the largest term over the value.
        let lost = match (sum.exponent(), largest) {
            (Some(e), Some(l)) if !sum.is_zero() => (i64::from(l) - i64::from(e)).max(0) as usize,
            (_, None) => 0,
            _ => wp,
        };
        if lost + rounding + 8 <= extra {
            return Ok(round_to(sum, prec));
        }
        if extra >= cap {
            return Err(special_exhausted(prec));
        }
        extra = (lost + rounding + 40).max(extra + 64).min(cap);
    }
}

/// The sum of [`arb_jacobi`] at `wp` bits, and the exponent of its largest
/// term (`None` when all are 0).
fn jacobi_sum(
    deg: usize,
    a: &BigFloat,
    b: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<(BigFloat, Option<i32>), SymplexError> {
    let wp = w.bits();
    let one = BigFloat::from_i32(1, wp);
    let two = BigFloat::from_i32(2, wp);
    let lo = x.sub(&one, wp, RM).div(&two, wp, RM);
    let hi = x.add(&one, wp, RM).div(&two, wp, RM);
    let int = |j: usize| BigFloat::from_u64(j as u64, 64);
    // ∏_{j=from}^{deg} (p + j).
    let shifted = |p: &BigFloat, from: usize| -> BigFloat {
        let mut acc = one.clone();
        for j in from..=deg {
            acc = acc.mul(&p.add(&int(j), wp, RM), wp, RM);
        }
        acc
    };
    // `a + j` vanishes at `j = −a` for a negative integer `a`: the terms
    // `s < −a` (whose product holds that factor) are 0.
    let first = match bf_as_int(a)? {
        Some(m) if a.is_int() && m < 0 && m.unsigned_abs() as usize <= deg => {
            m.unsigned_abs() as usize
        }
        _ => 0,
    };
    // The sum over `s ≥ first`: (x+1)/2 = 0 leaves the term `s = n` alone.
    let hi_zero = hi.is_zero();
    let ratio = if hi_zero {
        BigFloat::new(wp)
    } else {
        lo.div(&hi, wp, RM)
    };
    let mut pa = shifted(a, first + 1);
    let mut pb = one.clone();
    for s in 1..=first {
        pb = pb.mul(&b.add(&int(deg - s + 1), wp, RM), wp, RM);
    }
    // 1/((n − first)!·first!).
    let f0 = factorial_bigint(deg - first) * factorial_bigint(first);
    let mut inv_fact = one.div(&ratio_to_bigfloat(&f0, wp), wp, RM);
    // lo^first·hi^(n − first).
    let direct_pow = |s: usize| lo.powi(s, wp, RM).mul(&hi.powi(deg - s, wp, RM), wp, RM);
    let mut pow = if hi_zero {
        BigFloat::new(wp)
    } else {
        direct_pow(first)
    };
    // A first power beyond the exponent range (`x` within `2^(−wp/n)`-ish of
    // −1): every power directly, as before 0.34.
    let direct = !hi_zero && !lo.is_zero() && (pow.is_zero() || pow.is_inf());
    let mut sum = BigFloat::new(wp);
    let mut largest: Option<i32> = None;
    for s in first..=deg {
        if s > first {
            pa = pa.div(&a.add(&int(s), wp, RM), wp, RM);
            pb = pb.mul(&b.add(&int(deg - s + 1), wp, RM), wp, RM);
            inv_fact = inv_fact.mul(&int(deg - s + 1), wp, RM).div(&int(s), wp, RM);
            pow = if direct {
                direct_pow(s)
            } else {
                pow.mul(&ratio, wp, RM)
            };
        }
        if hi_zero && s == deg {
            pow = lo.powi(deg, wp, RM);
        }
        let term = pa.mul(&pb, wp, RM).mul(&inv_fact, wp, RM).mul(&pow, wp, RM);
        if let Some(e) = term.exponent().filter(|_| !term.is_zero()) {
            largest = Some(largest.map_or(e, |l| l.max(e)));
        }
        sum = sum.add(&term, wp, RM);
    }
    Ok((sum, largest))
}

/// `n!` as a rational.
fn factorial_bigint(n: usize) -> Ratio<BigInt> {
    let mut f = BigInt::from(1);
    for i in 2..=n {
        f *= BigInt::from(i as u64);
    }
    Ratio::from_integer(f)
}

/// Associated Legendre `P_n^m(x)` (Condon–Shortley phase) for integer
/// `n ≥ 0`, integer `m`, real `x` (`|x| ≤ 1` when `m` is odd).
fn arb_assoc_legendre(
    n: &BigFloat,
    m: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    let deg = poly_degree(n, "assoc_legendre")? as i64;
    let order = bf_as_int(m)?
        .filter(|_| m.is_int())
        .ok_or_else(|| unevaluable("assoc_legendre: order must be an integer"))?;
    if order.abs() > deg {
        return Ok(BigFloat::new(prec));
    }
    let wp = prec + 32 + 2 * (deg as f64).log2().ceil().max(0.0) as usize;
    let one = BigFloat::from_i32(1, wp);
    let mu = order.unsigned_abs() as usize;
    let x2 = x.mul(x, wp, RM);
    let one_minus_x2 = one.sub(&x2, wp, RM);
    // (1 − x²)^{μ/2}
    let root_pow = if mu.is_multiple_of(2) {
        one_minus_x2.powi(mu / 2, wp, RM)
    } else {
        if one_minus_x2.is_negative() {
            return Err(unevaluable(
                "assoc_legendre with odd order is complex for |x| > 1",
            ));
        }
        one_minus_x2
            .sqrt(wp, RM)
            .mul(&one_minus_x2.powi(mu / 2, wp, RM), wp, RM)
    };
    // P_μ^μ = (−1)^μ (2μ−1)!! (1−x²)^{μ/2}
    let mut dfact = BigInt::from(1);
    for j in 1..=mu {
        dfact *= BigInt::from((2 * j - 1) as u64);
    }
    if mu % 2 == 1 {
        dfact = -dfact;
    }
    let mut prev = ratio_to_bigfloat(&Ratio::from_integer(dfact), wp).mul(&root_pow, wp, RM);
    let deg_u = deg as usize;
    let mut curr = if deg_u > mu {
        BigFloat::from_i128(2 * mu as i128 + 1, wp)
            .mul(x, wp, RM)
            .mul(&prev, wp, RM)
    } else {
        prev.clone()
    };
    for k in (mu + 1)..deg_u {
        let t1 = BigFloat::from_i128(2 * k as i128 + 1, wp)
            .mul(x, wp, RM)
            .mul(&curr, wp, RM);
        let t2 = BigFloat::from_i128((k + mu) as i128, wp).mul(&prev, wp, RM);
        let next = t1
            .sub(&t2, wp, RM)
            .div(&BigFloat::from_i128((k - mu + 1) as i128, wp), wp, RM);
        prev = curr;
        curr = next;
    }
    if order < 0 {
        // P_n^{−μ} = (−1)^μ (n−μ)!/(n+μ)! P_n^μ
        let mut ratio = factorial_bigint(deg_u - mu) / factorial_bigint(deg_u + mu);
        if mu % 2 == 1 {
            ratio = -ratio;
        }
        curr = curr.mul(&ratio_to_bigfloat(&ratio, wp), wp, RM);
    }
    Ok(round_to(curr, prec))
}

/// Generalised Laguerre `L_n^{(a)}(x)`:
/// `(k+1) L_{k+1} = (2k+1+a−x) L_k − (k+a) L_{k−1}`.
fn arb_assoc_laguerre(
    n: &BigFloat,
    a: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    let deg = poly_degree(n, "assoc_laguerre")?;
    let wp = prec + 32 + (deg as f64).log2().ceil().max(0.0) as usize;
    let one = BigFloat::from_i32(1, wp);
    if deg == 0 {
        return Ok(BigFloat::from_i32(1, prec));
    }
    let mut prev = one.clone();
    let mut curr = one.add(a, wp, RM).sub(x, wp, RM);
    for k in 1..deg {
        let k_bf = BigFloat::from_i128(k as i128, wp);
        let coeff = BigFloat::from_i128(2 * k as i128 + 1, wp)
            .add(a, wp, RM)
            .sub(x, wp, RM);
        let t1 = coeff.mul(&curr, wp, RM);
        let t2 = k_bf.add(a, wp, RM).mul(&prev, wp, RM);
        let next = t1
            .sub(&t2, wp, RM)
            .div(&BigFloat::from_i128(k as i128 + 1, wp), wp, RM);
        prev = curr;
        curr = next;
    }
    Ok(round_to(curr, prec))
}

/// The reason of the error [`get_cached`] raises for a sub-expression
/// with no value ([`root_cause`] replaces it by the sub-expression's own
/// error).
fn not_evaluated_reason(id: ExprId) -> String {
    format!("sub-expression {id:?} has no numerical value")
}

fn get_cached(cache: Cache<'_>, id: ExprId) -> Result<&Complex, SymplexError> {
    cache.get(&id).ok_or_else(|| SymplexError::Unevaluable {
        reason: not_evaluated_reason(id),
    })
}

/// Convert a `Ratio<BigInt>` to a `BigFloat` with the given precision in
/// bits (see [`crate::base::numeric::ratio_to_bigfloat`]).
fn ratio_to_bigfloat(r: &Ratio<BigInt>, prec: usize) -> BigFloat {
    crate::base::numeric::ratio_to_bigfloat(r, prec, RM)
}

/// Try to extract a small integer exponent (fits in i32) from an ExprId.
fn try_as_small_int(arena: &Arena, id: ExprId) -> Option<i32> {
    let r = arena.as_num(id)?;
    if !r.is_integer() {
        return None;
    }
    let n = r.to_integer();
    let val: i64 = n.try_into().ok()?;
    if val >= i32::MIN as i64 && val <= i32::MAX as i64 {
        Some(val as i32)
    } else {
        None
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Decimal / complex formatting
// ═══════════════════════════════════════════════════════════════════════════

/// Determine whether `part` is negligibly small compared to `other`.
///
/// Returns `true` if `part` is zero, or if its binary exponent is more
/// than `digits * log2(10)` bits below `other`'s exponent (meaning it
/// is below the requested output precision).
fn is_negligible_part(part: &BigFloat, other: &BigFloat, digits: u32) -> bool {
    if part.is_zero() {
        return true;
    }
    if other.is_zero() {
        return false;
    }
    match (part.exponent(), other.exponent()) {
        (Some(p_exp), Some(o_exp)) => {
            let bit_threshold = (digits as i64) * 34 / 10 + 4;
            (o_exp as i64 - p_exp as i64) > bit_threshold
        }
        _ => false,
    }
}

/// Format a complex result as a string.
///
/// A part that is 0 is not printed: a real number, a pure imaginary
/// number, or `a + b*i` / `a - b*i`.  [`settle`] has set to 0 every part
/// whose ball contains 0 (it is below the printed digits of the value)
/// and certified the digits of every other part, however small next to
/// the other one, so a part certified nonzero is always printed (SymPy's
/// `N` agrees: `Chi(−60)` is `9.68091107·10²³ + 3.141592654i` at 10
/// digits).  Up to 0.52 a part negligible next to the other
/// ([`is_negligible_part`]) was dropped whatever its ball: `Chi(−60)`
/// printed `9.68091107e23`, `10³⁰ + i` printed `1e30` and `1 + 10³⁰·i`
/// printed `1e30*i`.
fn format_complex(z: &Complex, digits: u32) -> Result<String, SymplexError> {
    let re = &z.0;
    let im = &z.1;

    if re.is_zero() && im.is_zero() {
        return Ok("0".to_string());
    }

    if im.is_zero() {
        return format_decimal(re, digits);
    }

    if re.is_zero() {
        // Pure imaginary.
        let im_str = format_decimal(im, digits)?;
        return if im_str == "1" {
            Ok("i".to_string())
        } else if im_str == "-1" {
            Ok("-i".to_string())
        } else {
            Ok(format!("{im_str}*i"))
        };
    }

    // Both parts present.
    let re_str = format_decimal(re, digits)?;
    if im.is_negative() {
        let im_abs_str = format_decimal(&im.abs(), digits)?;
        if im_abs_str == "1" {
            Ok(format!("{re_str} - i"))
        } else {
            Ok(format!("{re_str} - {im_abs_str}*i"))
        }
    } else {
        let im_str = format_decimal(im, digits)?;
        if im_str == "1" {
            Ok(format!("{re_str} + i"))
        } else {
            Ok(format!("{re_str} + {im_str}*i"))
        }
    }
}

/// Round a decimal digit string `0.d1d2d3…` to `digits` significant digits.
///
/// Rounds to nearest, ties to even (the digits beyond the requested
/// count are exact decimal digits of the binary value, so an exact tie is
/// a genuine `…5000…`).  Carries propagate leftwards; a carry out of the
/// leading digit (`9.9996 → 10.00`) is reported as an exponent increment
/// so the caller can re-place the decimal point.
///
/// Returns `(rounded_digits, exponent_increment)`; when `mantissa` has no
/// more than `digits` digits it is returned unchanged.
fn round_decimal_digits(mantissa: &[u8], digits: usize) -> (Vec<u8>, i64) {
    if digits == 0 || mantissa.len() <= digits {
        return (mantissa.to_vec(), 0);
    }
    let mut kept: Vec<u8> = mantissa[..digits].to_vec();
    let next = mantissa[digits];
    let rest_nonzero = mantissa[digits + 1..].iter().any(|&d| d != 0);
    let round_up = match next.cmp(&5) {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Less => false,
        // Exactly half: ties to even.
        std::cmp::Ordering::Equal => rest_nonzero || kept[digits - 1] % 2 == 1,
    };
    if !round_up {
        return (kept, 0);
    }
    // Propagate the carry from the last kept digit leftwards.
    for d in kept.iter_mut().rev() {
        if *d == 9 {
            *d = 0;
        } else {
            *d += 1;
            return (kept, 0);
        }
    }
    // Carried out of the leading digit: 0.999… → 1.000… × 10.
    kept[0] = 1;
    (kept, 1)
}

/// Format a `BigFloat` as a decimal string with `digits` significant digits.
///
/// Converts with [`crate::base::bigfloat::to_decimal_digits`] (exact,
/// from the binary mantissa), rounds to `digits` significant digits, then
/// formats the result with proper decimal point placement.
fn format_decimal(val: &BigFloat, digits: u32) -> Result<String, SymplexError> {
    if val.is_zero() {
        return Ok("0".to_string());
    }

    if val.is_inf_pos() {
        return Ok("oo".to_string());
    }
    if val.is_inf_neg() {
        return Ok("-oo".to_string());
    }

    let crate::base::bigfloat::DecimalDigits {
        sign,
        digits: mantissa,
        exponent,
    } = crate::base::bigfloat::to_decimal_digits(val)
        .map_err(|e| SymplexError::NotImplemented(format!("decimal conversion failed: {e:?}")))?;

    let sign_str = if sign == Sign::Neg { "-" } else { "" };

    // mantissa is Vec<u8> of decimal digits [d1, d2, d3, ...]
    // representing the value 0.d1d2d3... × 10^exponent.
    // So the actual value is d1.d2d3... × 10^(exponent-1).

    // Round (not truncate) to the requested digit count.
    let (rounded, exp_carry) = round_decimal_digits(&mantissa, digits as usize);
    let n = rounded.len();
    if n == 0 {
        return Ok("0".to_string());
    }

    let digit_chars: Vec<char> = rounded.iter().map(|&d| (b'0' + d) as char).collect();

    // The "adjusted exponent" is exponent-1 (shifting from 0.ddd to d.ddd notation).
    let adj_exp = exponent as i64 - 1 + exp_carry;

    // Decide between plain decimal and scientific notation.
    if adj_exp >= 0 && (adj_exp as usize) < n {
        // The decimal point falls within the digit string.
        // E.g., digits=[3,1,4,1,5], adj_exp=0 → "3.1415"
        let dot_pos = adj_exp as usize + 1;
        let mut s = String::with_capacity(n + 2);
        s.push_str(sign_str);
        for (i, &ch) in digit_chars.iter().enumerate() {
            if i == dot_pos && dot_pos < n {
                s.push('.');
            }
            s.push(ch);
        }
        // Trim trailing zeros after the decimal point.
        if s.contains('.') {
            let trimmed = s.trim_end_matches('0').trim_end_matches('.');
            Ok(trimmed.to_string())
        } else {
            Ok(s)
        }
    } else if adj_exp >= 0 && (adj_exp as usize) < n + 6 {
        // Small integer — pad with trailing zeros.
        let mut s = String::with_capacity(adj_exp as usize + 2);
        s.push_str(sign_str);
        for &ch in &digit_chars {
            s.push(ch);
        }
        let needed_zeros = (adj_exp as usize + 1).saturating_sub(digit_chars.len());
        for _ in 0..needed_zeros {
            s.push('0');
        }
        Ok(s)
    } else if (-4..0).contains(&adj_exp) {
        // Small fraction like 0.00123.
        let mut s = String::with_capacity(n + (-adj_exp) as usize + 3);
        s.push_str(sign_str);
        s.push_str("0.");
        let leading_zeros = (-adj_exp - 1) as usize;
        for _ in 0..leading_zeros {
            s.push('0');
        }
        for &ch in &digit_chars {
            s.push(ch);
        }
        // Trim trailing zeros.
        let trimmed = s.trim_end_matches('0');
        Ok(trimmed.to_string())
    } else {
        // Scientific notation: d1.d2d3...dn × 10^adj_exp.
        let mut s = String::with_capacity(n + 10);
        s.push_str(sign_str);
        s.push(digit_chars[0]);
        if n > 1 {
            s.push('.');
            for &ch in &digit_chars[1..] {
                s.push(ch);
            }
            // Trim trailing zeros in the fractional part.
            while s.ends_with('0') {
                s.pop();
            }
            if s.ends_with('.') {
                s.pop();
            }
        }
        s.push_str(&format!("e{adj_exp}"));
        Ok(s)
    }
}

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
}

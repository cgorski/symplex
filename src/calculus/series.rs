//! Taylor / Laurent / asymptotic series expansion.
//!
//! [`series`] computes the truncated expansion of an expression around a
//! point (Maclaurin when the point is `0`, asymptotic when the point is
//! `±∞` via [`series_at_infinity`]); [`series_dir`] from one side.
//!
//! # Contract
//!
//! `series(f, x, x0, n)` is the **asymptotic expansion of `f` as `x → x0`
//! through real values** — two-sided unless a direction is given
//! ([`series_dir`]; at `±∞` the side is implied) — in the same sense as
//! `limit` and the Gruntz algorithm: the result `S` contains every term of
//! exponent `< n` in `x − x0` (in `1/x` at `±∞`), each exact, and
//! `f − S = O((x − x0)ⁿ)` as `x` approaches `x0` along the reals from the
//! side(s) in question.  It is **not** a Laurent series over the complex
//! plane:
//!
//! * `exp(−1/x²)` at `0` is `0` to every order (all real Taylor
//!   coefficients vanish; SymPy: `series(exp(-1/x**2), x, 0, 4) = O(x**4)`),
//!   although `0` is an essential singularity;
//! * `exp(−1/x)` at `0` has no two-sided expansion (it is `O(xⁿ)` from the
//!   right and unbounded from the left) and is refused; from the right it
//!   is `0`, from the left it is refused (SymPy's default `dir='+'` gives
//!   `O(x**4)`, `dir='-'` returns `exp(-1/x)` unexpanded);
//! * a two-sided kink (`|x|`) is refused unless both sides agree
//!   (`cos|x| = cos x`);
//! * one-sided expansions and those at `±∞` are *log-extended*: a
//!   coefficient may contain `ln(x − x0)` from the right, `ln(x0 − x)`
//!   from the left (`ln x` at `±∞`), as in Stirling's series
//!   `ln Γ(x) = x ln x − x − ½ ln x + ½ ln 2π + 1/(12x) + …` or
//!   `Ei(x) = γ + ln x + x + …` for `x → 0⁺`; the remainder is then
//!   `O((x − x0)ⁿ·|ln(x − x0)|ᵏ)`.  A two-sided expansion through a
//!   logarithm exists only where the logarithms cancel: it is the common
//!   expansion of the two sides when neither has a logarithm up to order
//!   `n` (`ln(sin x) − ln x = −x²/6 + …`; `x^x` and `x³·ln x` at order 3
//!   are refused).
//!
//! When no such expansion exists, or it cannot be established (a
//! fractional power two-sided, an oscillation, an unknown function whose
//! derivatives are singular at the point, the Stieltjes constants that
//! `ζ(x)` needs beyond `1/(x − 1) + γ`), the result is `Err` and the public
//! API keeps a formal `Series` node.
//!
//! Consumers that need the complex-analytic (Laurent) expansion —
//! residues, Laurent coefficients, inverse transforms — must not take it
//! from [`series`] for a function that may have an essential singularity at
//! the point: [`laurent_series`] runs the engine without limits and refuses
//! an essential singularity, and residues have their own structural route
//! ([`residue`](crate::calculus::residue)).
//!
//! # Algorithm
//!
//! Expansion is performed by a **truncated Laurent-series arithmetic engine**
//! that walks the expression DAG bottom-up (explicit post-order, no
//! recursion) and combines the series of the children:
//!
//! * `Add` / `Mul` / integer `Pow` — exact truncated arithmetic (poles are
//!   represented by a negative leading exponent, so `sin x / x` needs no
//!   special handling).
//! * `exp`, `sin`, `cos`, `sinh`, `cosh`, `ln`, `atan`, `atanh`, `asin`,
//!   `asinh`, `tan`, `tanh`, `erf`, `LambertW`, `(1+u)^α` — composed from
//!   closed-form Maclaurin coefficients (Bernoulli numbers for `tan`/`tanh`,
//!   central binomials for `asin`, `(−n)^{n−1}/n!` for `W`) rather than
//!   repeated differentiation, so high orders stay fast.
//! * `Γ`, `ψ`, `ψ⁽ᵐ⁾` at their poles, `ζ` at `1`, `ln Γ` and `ψ` at `+∞`
//!   (Stirling), `Ei`, `Ci`, `Chi`, `li`, `Si`, `Shi`, `K₀`, `Y₀` at the
//!   zeros of their arguments — closed forms at the points where
//!   differentiation has nothing to evaluate (see *Special functions at
//!   their singular points* below).
//! * A function without such a rule of an argument `u = u₀ + w` with a
//!   known expansion: `Σ f⁽ᵏ⁾(u₀)/k!·wᵏ` when `f` is analytic at the
//!   constant `u₀` (SymPy's `Function._eval_nseries`; `w` may contain
//!   `ln x`).
//! * Any other `var`-dependent sub-expression falls back to Taylor
//!   coefficients by differentiation, evaluated at the expansion point.
//!
//! Fractional powers of a series with a zero constant term (Puiseux
//! expansions such as `√x·sin x`) and logarithmic singularities outside a
//! log-extended expansion are rejected — the caller then keeps an
//! unevaluated `Series` node instead of producing a wrong polynomial.
//! Such rejections are *definite*: the differentiation fallback is never
//! tried for them, because the low-order derivatives of `x^(5/2)` or `|x²|`
//! all vanish at `0` and would silently yield the wrong polynomial `0`.
//!
//! `|g|` is expanded as `±g` when the sign of `g` near the point is known:
//! always when the leading exponent of `g` is even (`|x²| = x²`,
//! `|x − 1| = 1 − x`), and one-sidedly otherwise.  For an odd leading
//! exponent the two one-sided expansions of the *whole* expression are
//! compared and accepted only when they agree (`cos|x| = cos x`); `e^|x|`
//! and `|sin x|` have no two-sided expansion and are rejected.
//!
//! # Design
//!
//! The result is a plain expression (polynomial in `(x − a)`, possibly with
//! negative powers).  No `O(·)` term is appended — the truncation order is
//! implicit in the `order` parameter: all terms with exponent `< order` are
//! present and exact.
//!
//! A coefficient that is a sum of constants is tested for a hidden zero
//! (`asinh 2 − ln(2 + √5)`) numerically as it forms
//! ([`settle_constant`]): the valuation of a series, and so every division
//! by it, depends on knowing which leading coefficients vanish.

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Signed, ToPrimitive, Zero};
use rustc_hash::FxHashMap;

use crate::base::arena::Arena;
use crate::base::assumptions::Props;
use crate::base::bernoulli::bernoulli;
use crate::base::combinatorics::{binomial, factorial};
use crate::base::errors::SymplexError;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::Q;
use crate::base::walk;
use crate::calculus::calculus_util::settle_constant;
use crate::transforms::{eval, subs};

/// Maximum number of sub-expressions expanded by differentiation before the
/// engine gives up on per-node fallbacks (the root is always tried).
const MAX_FALLBACK_NODES: usize = 4;

/// Largest derivative (tree size) the differentiation fallback evaluates.
const MAX_FALLBACK_SIZE: usize = 1_500;

/// Maximum integer exponent expanded by repeated multiplication; larger
/// exponents use the binomial series.
const MAX_INT_POWER: i64 = 64;

/// Minimum internal working precision of the engine.
const MIN_WORKING_ORDER: i64 = 4;

/// Number of precision-escalation rounds before giving up on a deep pole.
const MAX_PRECISION_ATTEMPTS: usize = 3;

// ═══════════════════════════════════════════════════════════════════════════
// Public entry points
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the series of `expr` in `var` around `point` with all terms of
/// exponent `< order` (in `var − point`): the two-sided real asymptotic
/// expansion of the [contract](self#contract).
///
/// For `point = 0` this is the Maclaurin series; for `point = ±∞` an
/// asymptotic expansion in `1/var` (see [`series_at_infinity`]).  Poles at
/// the expansion point produce negative powers.
///
/// Returns `Err` when no such expansion exists or cannot be established
/// (fractional-power or logarithmic singularity, the two sides differing,
/// an unsupported sub-expression whose derivatives are singular at the
/// point).
pub(crate) fn series(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    order: u32,
) -> Result<ExprId, SymplexError> {
    let mode = Mode {
        side: Side::Both,
        log_var: None,
        limits: true,
        partial: false,
    };
    series_with_pole_retry(arena, expr, var, point, order, mode)
}

/// [`series`] as `var → point` from one side, log-extended:
/// [`Direction::Right`] (`var > point`; the result may contain
/// `ln(var − point)`, `Ei(x) = γ + ln x + x + x²/4 + …`) or
/// [`Direction::Left`] (`ln(point − var)`); [`Direction::Both`] is
/// [`series`].  At `±∞` the direction is implied.
///
/// `exp(−1/x)` at `0` is `0` to every order from the right and has no
/// expansion from the left (nor a two-sided one).
///
/// [`Direction::Right`]: crate::calculus::limit::Direction::Right
/// [`Direction::Left`]: crate::calculus::limit::Direction::Left
/// [`Direction::Both`]: crate::calculus::limit::Direction::Both
pub(crate) fn series_dir(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    order: u32,
    dir: crate::calculus::limit::Direction,
) -> Result<ExprId, SymplexError> {
    use crate::calculus::limit::Direction;
    let mode = match dir {
        Direction::Both => return series(arena, expr, var, point, order),
        Direction::Right => Mode {
            side: Side::Above,
            log_var: Some(arena.symbol(LOG_PLACEHOLDER)),
            limits: true,
            partial: false,
        },
        Direction::Left => Mode {
            side: Side::Below,
            log_var: Some(arena.symbol(LOG_PLACEHOLDER)),
            limits: true,
            partial: false,
        },
    };
    let err = match series_with_pole_retry(arena, expr, var, point, order, mode) {
        Ok(s) => return Ok(s),
        Err(e) => e,
    };
    if order == 0
        || matches!(
            arena.node(point),
            ExprNode::Infinity | ExprNode::NegInfinity
        )
        || !matches!(arena.node(var), ExprNode::Symbol(_))
    {
        return Err(err);
    }
    for q in PUISEUX_RAMIFICATIONS {
        if let Ok(s) = puiseux_series(arena, expr, var, point, order, dir == Direction::Left, q) {
            return Ok(s);
        }
    }
    Err(err)
}

/// The ramification indices [`series_dir`] tries, in order, when there is
/// no expansion in integer powers: `√(x − a)` (branch points of `asin`,
/// `acos`, `acosh`, `√`), then `∛(x − a)`.
const PUISEUX_RAMIFICATIONS: [u32; 2] = [2, 3];

/// Stand-in for the ramified variable `τ` of [`puiseux_series`].
const PUISEUX_PLACEHOLDER: &str = "__series_tau";

/// The one-sided expansion of `expr` as `var → point` in powers of
/// `τ = |var − point|^(1/q)` (a Puiseux series), as SymPy gives for a
/// branch point: `series(asin(x**2), x, 1, 4)` is `π/2 − 2i·√(x − 1) −
/// i(x − 1)^(3/2)/6 + …` (SymPy's `asin._eval_nseries` puts `1 − u = t²`
/// for a positive `t`).
///
/// `var = point ± τ^q` with `τ → 0⁺`: the expansion in `τ` is an ordinary
/// one-sided (log-extended) expansion from above to order `q·order`, and
/// substituting `τ = (±(var − point))^(1/q)` back is exact on that side,
/// where the radicand is positive (so its principal root is the real `τ`,
/// and `ln τ = ln(±(var − point))/q`).
fn puiseux_series(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    order: u32,
    from_left: bool,
    q: u32,
) -> Result<ExprId, SymplexError> {
    let tau = arena.symbol(PUISEUX_PLACEHOLDER);
    let q_id = arena.int(i64::from(q));
    let tau_q = arena.pow(tau, q_id);
    let offset = if from_left { arena.neg(tau_q) } else { tau_q };
    let x_of_tau = arena.add(&[point, offset]);
    let in_tau = subs::subs(arena, expr, var, x_of_tau);
    let mode = Mode {
        side: Side::Above,
        log_var: Some(arena.symbol(LOG_PLACEHOLDER)),
        limits: true,
        partial: false,
    };
    let tau_order = order
        .checked_mul(q)
        .ok_or(SymplexError::ComputationFailed {
            operation: "series",
            reason: "order too large for a Puiseux expansion".into(),
        })?;
    let zero = arena.zero;
    let poly = series_with_pole_retry(arena, in_tau, tau, zero, tau_order, mode)?;
    // τ = (±(var − point))^(1/q); the expansion in τ already carries
    // `ln τ` as `ln(τ)`, which becomes `ln(radicand)/q`.
    let x_minus_a = arena.sub(var, point);
    let radicand = if from_left {
        arena.neg(x_minus_a)
    } else {
        x_minus_a
    };
    let radicand = eval::eval(arena, radicand);
    let ln_tau = arena.ln(tau);
    let ln_rad = arena.ln(radicand);
    let inv_q = arena.rational(1, i64::from(q));
    let ln_rad_q = arena.mul(&[ln_rad, inv_q]);
    let poly = arena.subs_structural(poly, ln_tau, ln_rad_q);
    let root = arena.pow(radicand, inv_q);
    let back = subs::subs(arena, poly, tau, root);
    let back = eval::eval(arena, back);
    if walk::contains(arena, back, tau)
        || contains_singular_atom(arena, back)
        || walk::has_unevaluated(arena, back)
    {
        return Err(SymplexError::ComputationFailed {
            operation: "series",
            reason: "no Puiseux expansion found".into(),
        });
    }
    Ok(back)
}

/// Stand-in for `ln t` during a log-extended expansion (replaced before the
/// result is returned).
const LOG_PLACEHOLDER: &str = "__series_ln_t";

/// [`series_in_mode`], retried as `(x − a)⁻ᵏ·series((x − a)ᵏ·f)` for
/// `k = 1..=5` when the direct expansion fails: a pole of a function the
/// engine only knows through the differentiation fallback (`sinh(ln x) =
/// (x − 1/x)/2`) is invisible to Taylor coefficients.  For the real
/// asymptotic expansion this is exact: `xᵏ·f = S + O(xⁿ⁺ᵏ)` gives
/// `f = S/xᵏ + O(xⁿ)` for `x ≠ 0`.  (Before, this retry lived only in
/// [`laurent_series`], reached as `series`'s fallback, which is now
/// restricted to meromorphic functions.)
fn series_with_pole_retry(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    order: u32,
    mode: Mode,
) -> Result<ExprId, SymplexError> {
    let err = match series_in_mode(arena, expr, var, point, order, mode) {
        Ok(s) => return Ok(s),
        Err(e) => e,
    };
    if matches!(
        arena.node(point),
        ExprNode::Infinity | ExprNode::NegInfinity
    ) || !matches!(arena.node(var), ExprNode::Symbol(_))
        || order == 0
        || retry_cannot_help(arena, expr, var, point, mode.side)
    {
        return Err(err);
    }
    let x_minus_a = if arena.is_zero_structural(point) {
        var
    } else {
        arena.sub(var, point)
    };
    for k in 1u32..=5 {
        let k_id = arena.int(k as i64);
        let multiplier = arena.pow(x_minus_a, k_id);
        let modified = arena.mul(&[expr, multiplier]);
        if let Ok(ts) = series_in_mode(arena, modified, var, point, order + k, mode) {
            let neg_k = arena.int(-(k as i64));
            let divisor = arena.pow(x_minus_a, neg_k);
            let result = arena.mul(&[ts, divisor]);
            let result = crate::transforms::expand::expand(arena, result);
            return Ok(eval::eval(arena, result));
        }
    }
    Err(err)
}

/// Can the retries of [`series_with_pole_retry`] not succeed where the
/// direct expansion of `expr` failed?  Decided from the limits of
/// `g = (var − point)·expr` from the side(s) of `side`:
///
/// * `g → 0` from each side: `expr` has no pole.  From
///   `xᵏ·f = S + O(xⁿ⁺ᵏ)`, a term `c·x⁻ʲ` (`j ≥ 1`) of `S/xᵏ` would make
///   `g` tend to `c` or diverge, so `S/xᵏ` would be a power series, which
///   the direct attempt would have found.
/// * two-sided, `g` tends to different finite limits from the two sides:
///   every `xᵏ·f = xᵏ⁻¹·g` has different coefficients of `xᵏ⁻¹` from the
///   two sides, so none has a two-sided expansion.
///
/// Each retry asks the limit engine for the limits of ever larger
/// derivatives: `series(atan(exp(−1/x)), x, 0, 3)` (limits 0 and π/2)
/// took 0.14 s to refuse, `acos(ln x)` from the left (unbounded, but only
/// logarithmically) 0.14 s, `ln(1 + exp(−1/x))` (`≈ −1/x` from the left,
/// `0` from the right) 60 ms.  A limit that cannot be determined answers
/// `false`.
fn retry_cannot_help(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    side: Side,
) -> bool {
    use crate::calculus::limit::Direction;
    let dirs: &[Direction] = match side {
        Side::Above => &[Direction::Right],
        Side::Below => &[Direction::Left],
        Side::Both => &[Direction::Right, Direction::Left],
    };
    let x_minus_a = if arena.is_zero_structural(point) {
        var
    } else {
        arena.sub(var, point)
    };
    let scaled = arena.mul(&[x_minus_a, expr]);
    let mut limits = Vec::with_capacity(dirs.len());
    for &dir in dirs {
        match crate::calculus::limit::limit_dir_generic(arena, scaled, var, point, dir) {
            Ok(v) if is_finite_constant(arena, v, var) => limits.push(v),
            _ => return false,
        }
    }
    if limits.iter().all(|&v| arena.is_zero_structural(v)) {
        return true;
    }
    match limits[..] {
        [right, left] => {
            let d = arena.sub(right, left);
            let d = eval::eval(arena, d);
            crate::calculus::limit::const_sign(arena, d).is_some_and(|s| s != 0)
        }
        _ => false,
    }
}

/// The expansion at a finite `point` (or `±∞`) in the given [`Mode`].
fn series_in_mode(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    order: u32,
    mode: Mode,
) -> Result<ExprId, SymplexError> {
    if order == 0 {
        return Ok(arena.zero);
    }
    match arena.node(point) {
        ExprNode::Infinity => return series_at_infinity(arena, expr, var, order, false),
        ExprNode::NegInfinity => return series_at_infinity(arena, expr, var, order, true),
        _ => {}
    }
    if !matches!(arena.node(var), ExprNode::Symbol(_)) {
        return Err(SymplexError::InvalidArgument {
            operation: "series",
            reason: "expansion variable must be a symbol".into(),
        });
    }
    let at_zero = arena.is_zero_structural(point);
    // Shift so that the expansion point becomes 0:  f(x) = f(a + t).
    let shifted = if at_zero {
        expr
    } else {
        let t_plus_a = arena.add(&[var, point]);
        subs::subs(arena, expr, var, t_plus_a)
    };
    let ts = expand_with_mode(arena, shifted, var, order as i64, mode)?;
    let mut poly = ts.to_expr(arena, var, order as i64);
    if let Some(l) = mode.log_var {
        // `l` is `ln|t|`: `ln t` from above, `ln(−t)` from below.
        let abs_t = if mode.side == Side::Below {
            let n = arena.neg(var);
            eval::eval(arena, n)
        } else {
            var
        };
        let ln_t = arena.ln(abs_t);
        poly = subs::subs(arena, poly, l, ln_t);
        poly = eval::eval(arena, poly);
    }
    let result = if at_zero {
        poly
    } else {
        let x_minus_a = arena.sub(var, point);
        let back = subs::subs(arena, poly, var, x_minus_a);
        eval::eval(arena, back)
    };
    if mode.log_var.is_some()
        && (contains_singular_atom(arena, result) || walk::has_unevaluated(arena, result))
    {
        return Err(SymplexError::ComputationFailed {
            operation: "series",
            reason: "a coefficient of the one-sided expansion is singular".into(),
        });
    }
    Ok(result)
}

/// Asymptotic expansion of `expr` as `var → +∞` (or `−∞` when `negative`),
/// with all terms of exponent `< order` in `1/var`.
///
/// Substitutes `var = ±1/t`, expands at `t = 0`, and substitutes back.
pub(crate) fn series_at_infinity(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    order: u32,
    negative: bool,
) -> Result<ExprId, SymplexError> {
    if order == 0 {
        return Ok(arena.zero);
    }
    let t = arena.symbol("_t");
    let one = arena.one;
    let inv_t = arena.div(one, t);
    let inv_t = if negative { arena.neg(inv_t) } else { inv_t };
    let in_t = subs::subs(arena, expr, var, inv_t);
    // t = ±1/x → 0⁺ only, so fractional powers of t^(even) are
    // single-valued, and `ln t` is real: the expansion is log-extended, with
    // `ln t = −ln(±x)` (Stirling's `x ln x − x − ½ ln x + …`).
    let log_t = arena.symbol(LOG_PLACEHOLDER);
    let ts = expand_log_extended(arena, in_t, t, order as i64, log_t, true)?;
    let poly = ts.to_expr(arena, t, order as i64);
    let ln_arg = if negative { arena.neg(var) } else { var };
    let ln_x = arena.ln(ln_arg);
    let minus_ln_x = arena.neg(ln_x);
    let poly = subs::subs(arena, poly, log_t, minus_ln_x);
    let inv_x = arena.div(one, var);
    let inv_x = if negative { arena.neg(inv_x) } else { inv_x };
    let back = subs::subs(arena, poly, t, inv_x);
    let result = eval::eval(arena, back);
    // Never return a bogus expansion: a coefficient that is infinite,
    // undefined or an unevaluated limit means the expansion failed.
    if contains_singular_atom(arena, result) || walk::has_unevaluated(arena, result) {
        return Err(SymplexError::ComputationFailed {
            operation: "series_at_infinity",
            reason: format!(
                "no asymptotic expansion in powers of 1/{}: a coefficient is singular",
                arena.display(var)
            ),
        });
    }
    Ok(result)
}

/// `true` if the expression contains `∞`, `−∞`, `zoo`, or `NaN` anywhere.
fn contains_singular_atom(arena: &Arena, id: ExprId) -> bool {
    walk::contains(arena, id, arena.infinity)
        || walk::contains(arena, id, arena.neg_infinity)
        || walk::contains(arena, id, arena.complex_infinity)
        || walk::contains(arena, id, arena.nan)
}

/// Compute a Laurent series expansion of `expr` in `var` around `point`.
///
/// Unlike [`series`] (a real asymptotic expansion) this is the
/// complex-analytic Laurent series, so it exists only where `expr` is
/// meromorphic at the point: the engine runs without taking limits in its
/// differentiation fallback, and an essential singularity is refused —
/// before, `exp(−1/x²)` had the "Laurent series" `0` (its real Taylor
/// series; the Laurent series has infinitely many negative powers).  As a
/// last resort this multiplies by `(x − a)^k`, `k = 1..=5`, and divides
/// back.
pub(crate) fn laurent_series(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    order: u32,
) -> Result<ExprId, SymplexError> {
    let mode = Mode {
        side: Side::Both,
        log_var: None,
        limits: false,
        partial: false,
    };
    let series = |arena: &mut Arena, e: ExprId, order: u32| {
        if matches!(
            arena.node(point),
            ExprNode::Infinity | ExprNode::NegInfinity
        ) {
            return Err(SymplexError::ComputationFailed {
                operation: "laurent_series",
                reason: "a Laurent series is taken at a finite point".into(),
            });
        }
        series_in_mode(arena, e, var, point, order, mode)
    };
    if let Ok(ts) = series(arena, expr, order) {
        return Ok(ts);
    }
    let x_minus_a = if arena.is_zero_structural(point) {
        var
    } else {
        arena.sub(var, point)
    };
    for k in 1u32..=5 {
        let k_id = arena.int(k as i64);
        let multiplier = arena.pow(x_minus_a, k_id);
        let modified = arena.mul(&[expr, multiplier]);
        if let Ok(ts) = series(arena, modified, order + k) {
            let neg_k = arena.int(-(k as i64));
            let divisor = arena.pow(x_minus_a, neg_k);
            let result = arena.mul(&[ts, divisor]);
            let result = crate::transforms::expand::expand(arena, result);
            return Ok(eval::eval(arena, result));
        }
    }
    Err(SymplexError::ComputationFailed {
        operation: "laurent_series",
        reason: "could not determine pole order (tried up to order 5)".into(),
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Truncated Laurent series
// ═══════════════════════════════════════════════════════════════════════════

/// `Σ_{i} coeffs[i] · x^(shift + i)`, exact for all exponents `< known`.
///
/// Invariant: `coeffs.len() == (known − shift) as usize`.
#[derive(Clone, Debug)]
pub(crate) struct TSeries {
    shift: i64,
    known: i64,
    coeffs: Vec<ExprId>,
}

impl TSeries {
    /// Exponent bound: coefficients are exact for all exponents `< known`.
    pub(crate) fn known(&self) -> i64 {
        self.known
    }

    /// Lowest stored exponent (negative for Laurent series).
    pub(crate) fn shift(&self) -> i64 {
        self.shift
    }

    /// Coefficient of `x^e`, zero outside the stored range (public alias of
    /// [`coeff_at`](Self::coeff_at) for other modules).
    pub(crate) fn coefficient(&self, arena: &Arena, e: i64) -> ExprId {
        self.coeff_at(arena, e)
    }

    fn zero(arena: &Arena, known: i64) -> Self {
        TSeries {
            shift: 0,
            known,
            coeffs: vec![arena.zero; known.max(0) as usize],
        }
    }

    fn constant(arena: &Arena, c: ExprId, known: i64) -> Self {
        let mut s = Self::zero(arena, known);
        if known > 0 {
            s.coeffs[0] = c;
        }
        s
    }

    fn var(arena: &Arena, known: i64) -> Self {
        let mut s = Self::zero(arena, known);
        if known > 1 {
            s.coeffs[1] = arena.one;
        }
        s
    }

    /// Coefficient of `x^e` (zero outside the stored range).
    fn coeff_at(&self, arena: &Arena, e: i64) -> ExprId {
        if e < self.shift || e >= self.known {
            arena.zero
        } else {
            self.coeffs[(e - self.shift) as usize]
        }
    }

    /// Exponent of the first structurally non-zero coefficient.
    fn leading_exponent(&self, arena: &Arena) -> Option<i64> {
        self.coeffs
            .iter()
            .position(|&c| !arena.is_zero_structural(c))
            .map(|i| self.shift + i as i64)
    }

    /// Drop leading structural zeros so that `shift` is the true valuation.
    fn normalized(mut self, arena: &Arena) -> Self {
        let lead = self
            .coeffs
            .iter()
            .position(|&c| !arena.is_zero_structural(c))
            .unwrap_or(self.coeffs.len());
        if lead > 0 {
            self.coeffs.drain(0..lead);
            self.shift += lead as i64;
        }
        self
    }

    /// Restrict to exponents `< n`.
    fn truncate_known(mut self, n: i64) -> Self {
        if n < self.known {
            let keep = (n - self.shift).max(0) as usize;
            self.coeffs.truncate(keep);
            self.known = n;
            if self.coeffs.is_empty() {
                self.shift = n;
            }
        }
        self
    }

    fn add(arena: &mut Arena, a: &TSeries, b: &TSeries) -> TSeries {
        let shift = a.shift.min(b.shift);
        let known = a.known.min(b.known);
        let mut coeffs = Vec::with_capacity((known - shift).max(0) as usize);
        for e in shift..known {
            let ca = a.coeff_at(arena, e);
            let cb = b.coeff_at(arena, e);
            let s = arena.add(&[ca, cb]);
            coeffs.push(
                if arena.is_zero_structural(ca) || arena.is_zero_structural(cb) {
                    eval::eval(arena, s)
                } else {
                    settle_constant(arena, s)
                },
            );
        }
        TSeries {
            shift,
            known,
            coeffs,
        }
    }

    fn scale(arena: &mut Arena, a: &TSeries, c: ExprId) -> TSeries {
        let coeffs = a
            .coeffs
            .iter()
            .map(|&x| {
                let p = arena.mul(&[c, x]);
                eval::eval(arena, p)
            })
            .collect();
        TSeries {
            shift: a.shift,
            known: a.known,
            coeffs,
        }
    }

    fn neg(arena: &mut Arena, a: &TSeries) -> TSeries {
        let m1 = arena.neg_one;
        Self::scale(arena, a, m1)
    }

    fn mul(arena: &mut Arena, a: &TSeries, b: &TSeries) -> TSeries {
        let a = a.clone().normalized(arena);
        let b = b.clone().normalized(arena);
        let shift = a.shift + b.shift;
        let known = (a.known + b.shift).min(b.known + a.shift);
        let len = (known - shift).max(0) as usize;
        let mut coeffs = Vec::with_capacity(len);
        for idx in 0..len {
            let mut terms = Vec::new();
            for (i, &ca) in a.coeffs.iter().enumerate() {
                if i > idx {
                    break;
                }
                let j = idx - i;
                if j >= b.coeffs.len() {
                    continue;
                }
                let cb = b.coeffs[j];
                if arena.is_zero_structural(ca) || arena.is_zero_structural(cb) {
                    continue;
                }
                terms.push(arena.mul(&[ca, cb]));
            }
            coeffs.push(match terms.len() {
                0 => arena.zero,
                1 => eval::eval(arena, terms[0]),
                _ => {
                    let s = arena.add(&terms);
                    settle_constant(arena, s)
                }
            });
        }
        TSeries {
            shift,
            known,
            coeffs,
        }
    }

    /// `1/a`.  Returns `None` if `a` is zero to the known precision.
    fn inverse(arena: &mut Arena, a: &TSeries) -> Option<TSeries> {
        let a = a.clone().normalized(arena);
        let v = a.leading_exponent(arena)?;
        let c0 = a.coeffs[0];
        let rel_known = a.known - v; // relative precision of 1 + w
        // w_i = a_{v+i}/c0 for i ≥ 1
        let inv_c0 = {
            let m1 = arena.neg_one;
            let p = arena.pow(c0, m1);
            eval::eval(arena, p)
        };
        let mut w: Vec<ExprId> = Vec::with_capacity(rel_known.max(0) as usize);
        w.push(arena.one);
        for i in 1..rel_known {
            let ai = a.coeff_at(arena, v + i);
            let p = arena.mul(&[ai, inv_c0]);
            w.push(eval::eval(arena, p));
        }
        // b_0 = 1, b_n = −Σ_{i=1}^{n} w_i b_{n−i}
        let mut b: Vec<ExprId> = Vec::with_capacity(w.len());
        b.push(arena.one);
        for n in 1..w.len() {
            let mut terms = Vec::new();
            for i in 1..=n {
                if arena.is_zero_structural(w[i]) || arena.is_zero_structural(b[n - i]) {
                    continue;
                }
                terms.push(arena.mul(&[w[i], b[n - i]]));
            }
            let s = match terms.len() {
                0 => arena.zero,
                1 => terms[0],
                _ => arena.add(&terms),
            };
            let ns = arena.neg(s);
            b.push(settle_constant(arena, ns));
        }
        let coeffs = b
            .iter()
            .map(|&x| {
                let p = arena.mul(&[inv_c0, x]);
                eval::eval(arena, p)
            })
            .collect();
        Some(TSeries {
            shift: -v,
            known: -v + rel_known,
            coeffs,
        })
    }

    /// `a^n` for integer `n`.
    fn pow_int(arena: &mut Arena, a: &TSeries, n: i64) -> Option<TSeries> {
        if n == 0 {
            return Some(Self::constant(arena, arena.one, a.known.max(1)));
        }
        if n.abs() > MAX_INT_POWER {
            return None;
        }
        let base = if n < 0 {
            Self::inverse(arena, a)?
        } else {
            a.clone()
        };
        let mut acc = base.clone();
        for _ in 1..n.abs() {
            acc = Self::mul(arena, &acc, &base);
        }
        Some(acc)
    }

    /// Remove the constant term, returning `(u0, w)` with `w = a − u0` (valuation ≥ 1).
    /// Replace a stored coefficient of `x⁰` by `c` (a value equal to it).
    fn set_constant(&mut self, c: ExprId) {
        if 0 >= self.shift && 0 < self.known {
            self.coeffs[(-self.shift) as usize] = c;
        }
    }

    fn split_constant(&self, arena: &Arena) -> (ExprId, TSeries) {
        let u0 = self.coeff_at(arena, 0);
        let mut w = self.clone();
        if 0 >= w.shift && 0 < w.known {
            w.coeffs[(-w.shift) as usize] = arena.zero;
        }
        (u0, w.normalized(arena))
    }

    /// `Σ_n f_n · w^n` for a series `w` with valuation ≥ 1.
    fn compose(arena: &mut Arena, f: &dyn Fn(&mut Arena, usize) -> ExprId, w: &TSeries) -> TSeries {
        let known = w.known;
        if known <= 0 {
            return TSeries {
                shift: known,
                known,
                coeffs: Vec::new(),
            };
        }
        let mut acc = Self::constant(arena, arena.zero, known);
        let f0 = f(arena, 0);
        acc.coeffs[0] = f0;
        if w.coeffs.iter().all(|&c| arena.is_zero_structural(c)) {
            return acc;
        }
        let mut p = w.clone().normalized(arena);
        let mut n = 1usize;
        loop {
            if p.shift >= known || p.coeffs.is_empty() {
                break;
            }
            let fn_ = f(arena, n);
            if !arena.is_zero_structural(fn_) {
                let term = Self::scale(arena, &p, fn_);
                acc = Self::add(arena, &acc, &term);
            }
            n += 1;
            if n > known as usize + 1 {
                break;
            }
            p = Self::mul(arena, &p, w).normalized(arena);
        }
        acc
    }

    /// Build the expression `Σ coeffs[i] x^(shift+i)` for exponents `< order`.
    fn to_expr(&self, arena: &mut Arena, var: ExprId, order: i64) -> ExprId {
        let mut terms = Vec::new();
        for (i, &c) in self.coeffs.iter().enumerate() {
            let e = self.shift + i as i64;
            if e >= order {
                break;
            }
            if arena.is_zero_structural(c) {
                continue;
            }
            let term = if e == 0 {
                c
            } else if e == 1 {
                arena.mul(&[c, var])
            } else {
                let ee = arena.int(e);
                let xp = arena.pow(var, ee);
                arena.mul(&[c, xp])
            };
            terms.push(term);
        }
        let s = match terms.len() {
            0 => arena.zero,
            1 => terms[0],
            _ => arena.add(&terms),
        };
        eval::eval(arena, s)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Closed-form Maclaurin coefficients
// ═══════════════════════════════════════════════════════════════════════════

fn rat_expr(arena: &mut Arena, r: Q) -> ExprId {
    let nid = arena.intern_num(r);
    arena.intern(ExprNode::Num(nid))
}

fn rat_i(n: i64) -> Q {
    Ratio::from_integer(BigInt::from(n))
}

/// The central binomial coefficient `C(2n, n)`.
fn central_binomial(n: u64) -> BigInt {
    binomial(2 * n, n)
}

fn pow_rat_i(r: &Q, n: i64) -> Q {
    let mut acc = Q::one();
    let base = if n < 0 { Q::one() / r } else { r.clone() };
    for _ in 0..n.unsigned_abs() {
        acc *= &base;
    }
    acc
}

/// Elementary functions with closed-form Maclaurin coefficients.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FnKind {
    Exp,
    Sin,
    Cos,
    Sinh,
    Cosh,
    /// `ln(1 + u)`
    Ln1p,
    Atan,
    Atanh,
    Asin,
    Asinh,
    Tan,
    Tanh,
    Erf,
    LambertW,
}

impl FnKind {
    /// The `n`-th Maclaurin coefficient of the function, as an exact rational
    /// where possible (`erf` carries a `2/√π` factor and is built as an
    /// expression).
    pub(crate) fn coefficient(self, arena: &mut Arena, n: usize) -> ExprId {
        let r = self.rational_coefficient(n);
        match self {
            FnKind::Erf => {
                if r.is_zero() {
                    return arena.zero;
                }
                // (2/√π) · (−1)^m / (m! (2m+1))
                let re = rat_expr(arena, r);
                let two = arena.int(2);
                let pi = arena.pi;
                let sp = arena.sqrt(pi);
                let f = arena.div(two, sp);
                let v = arena.mul(&[re, f]);
                eval::eval(arena, v)
            }
            _ => rat_expr(arena, r),
        }
    }

    /// Rational part of the `n`-th coefficient.
    pub(crate) fn rational_coefficient(self, n: usize) -> Q {
        let odd = n % 2 == 1;
        let m = n / 2;
        let sign_m = if m.is_multiple_of(2) {
            Q::one()
        } else {
            -Q::one()
        };
        match self {
            FnKind::Exp => Q::new(BigInt::one(), factorial(n as u64)),
            FnKind::Sin => {
                if odd {
                    sign_m / Q::from_integer(factorial(n as u64))
                } else {
                    Q::zero()
                }
            }
            FnKind::Cos => {
                if odd {
                    Q::zero()
                } else {
                    sign_m / Q::from_integer(factorial(n as u64))
                }
            }
            FnKind::Sinh => {
                if odd {
                    Q::new(BigInt::one(), factorial(n as u64))
                } else {
                    Q::zero()
                }
            }
            FnKind::Cosh => {
                if odd {
                    Q::zero()
                } else {
                    Q::new(BigInt::one(), factorial(n as u64))
                }
            }
            FnKind::Ln1p => {
                if n == 0 {
                    Q::zero()
                } else {
                    let s = if n % 2 == 1 { Q::one() } else { -Q::one() };
                    s / rat_i(n as i64)
                }
            }
            FnKind::Atan => {
                if odd {
                    sign_m / rat_i(n as i64)
                } else {
                    Q::zero()
                }
            }
            FnKind::Atanh => {
                if odd {
                    Q::one() / rat_i(n as i64)
                } else {
                    Q::zero()
                }
            }
            FnKind::Asin | FnKind::Asinh => {
                if !odd {
                    return Q::zero();
                }
                // C(2m,m) / (4^m (2m+1))
                let c = Q::from_integer(central_binomial(m as u64));
                let d = pow_rat_i(&rat_i(4), m as i64) * rat_i(n as i64);
                let v = c / d;
                if self == FnKind::Asinh { sign_m * v } else { v }
            }
            FnKind::Tan | FnKind::Tanh => {
                // tan x = Σ_{m≥1} (−1)^{m−1} 2^{2m}(2^{2m}−1) B_{2m} x^{2m−1}/(2m)!
                // tanh x = Σ_{m≥1} 2^{2m}(2^{2m}−1) B_{2m} x^{2m−1}/(2m)!
                if !odd {
                    return Q::zero();
                }
                let mm = m + 1; // n = 2mm − 1
                let two_pow = pow_rat_i(&rat_i(2), 2 * mm as i64);
                let b = bernoulli(2 * mm);
                let v = &two_pow * (&two_pow - Q::one()) * b
                    / Q::from_integer(factorial(2 * mm as u64));
                if self == FnKind::Tan {
                    if (mm - 1).is_multiple_of(2) { v } else { -v }
                } else {
                    v
                }
            }
            FnKind::Erf => {
                if !odd {
                    return Q::zero();
                }
                sign_m / (Q::from_integer(factorial(m as u64)) * rat_i(n as i64))
            }
            FnKind::LambertW => {
                if n == 0 {
                    return Q::zero();
                }
                // (−n)^{n−1} / n!
                let base = rat_i(-(n as i64));
                pow_rat_i(&base, n as i64 - 1) / Q::from_integer(factorial(n as u64))
            }
        }
    }
}

/// Generalised binomial coefficient `C(α, n)` for a symbolic or rational `α`.
fn gen_binomial_expr(arena: &mut Arena, alpha: ExprId, n: usize) -> ExprId {
    if n == 0 {
        return arena.one;
    }
    if let Some(a) = arena.as_num(alpha).cloned() {
        let mut acc = Q::one();
        for i in 0..n {
            acc = acc * (&a - rat_i(i as i64)) / rat_i(i as i64 + 1);
        }
        return rat_expr(arena, acc);
    }
    let mut factors = Vec::with_capacity(n + 1);
    for i in 0..n {
        let ie = arena.int(-(i as i64));
        factors.push(arena.add(&[alpha, ie]));
    }
    let inv_fact = rat_expr(arena, Q::new(BigInt::one(), factorial(n as u64)));
    factors.push(inv_fact);
    let p = arena.mul(&factors);
    eval::eval(arena, p)
}

// ═══════════════════════════════════════════════════════════════════════════
// The engine
// ═══════════════════════════════════════════════════════════════════════════

/// How the variable approaches the expansion point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Side {
    /// Two-sided (ordinary Maclaurin / Laurent expansion).
    Both,
    /// From above only (`x → 0⁺`; used for `x → ±∞` via `t = 1/x`).
    Above,
    /// From below only (`x → 0⁻`; `series_dir` from the left, and the
    /// cross-check of a two-sided expansion, see [`expand_with_mode`]).
    Below,
}

/// How one expansion is carried out.
#[derive(Clone, Copy, Debug)]
struct Mode {
    side: Side,
    /// With `Some(l)` (one-sided only) the expansion is *log-extended*: `l`
    /// is a `var`-free stand-in for `ln|var|` (`ln(var)` from above,
    /// `ln(−var)` from below), and `ln` of a series with a non-zero
    /// valuation `v` is `v·l + ln c + ln(1 + …)` instead of a refusal (from
    /// below `var^v = (−1)^v·|var|^v`, so `ln c` is `ln((−1)^v·c)`).
    /// Coefficients may then be polynomials in `l`; a term `c(l)·varᵏ` is
    /// still "of order `k`" (it is `o(var^(k−ε))`), as in Gruntz's
    /// algorithm, where `l = ln ω` belongs to a lower comparability class.
    /// `exp`, `sinh`, `cosh` of an argument whose constant term is `m·l`
    /// plus a constant, `m` an integer, shift the series by `|var|^m`
    /// (`exp(2 ln t) = t²`); any other dependence on `l` there is refused:
    /// that coefficient would not be of lower order.
    log_var: Option<ExprId>,
    /// May the differentiation fallback take limits of derivatives that
    /// cannot be substituted?  Off inside Gruntz (the limit engine calling
    /// itself through the series fallback has no overall budget).
    limits: bool,
    /// Accept a result exact to fewer terms than requested (its `known`
    /// says how many): Gruntz only needs the leading term, and `ζ(1 + w)`
    /// is known only to `O(w)`.
    partial: bool,
}

impl Mode {
    fn with_side(self, side: Side) -> Self {
        Mode { side, ..self }
    }
}

/// Why a structural expansion step produced no series.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Obstruction {
    /// No structural rule for this node; the caller may fall back to Taylor
    /// coefficients by differentiation.
    Unknown,
    /// No Laurent expansion exists here (Puiseux exponent, `|x|`-type
    /// singularity, non-real `|·|` argument).  The differentiation fallback
    /// must **not** be tried: the low-order derivatives of `x^(5/2)` or
    /// `|x²|` (`2x·sign(x²)`) all vanish at `0`, so it would silently return
    /// the wrong polynomial `0`.
    NoExpansion,
    /// `|g|` with `g` of odd valuation in a two-sided expansion: the two
    /// one-sided expansions of `|g|` differ, but the enclosing expression may
    /// still have a two-sided expansion (`cos|x|`).
    Kink,
    /// `ln` of a series with a non-zero valuation outside a log-extended
    /// expansion.  Two-sided, the enclosing expression may still have an
    /// expansion (`ln(sin x) − ln x = −x²/6 + …`): it is decided from the
    /// two log-extended one-sided expansions (see [`expand_with_mode`]).
    /// Without limits (a Laurent series) it is definite.
    Logarithmic,
}

/// [`expand_with_mode`] for a function that must be *meromorphic* at `0`
/// (a Laurent series in the complex sense, for [`laurent_series`] and
/// formal power series): two-sided, and the differentiation fallback takes
/// no limits, so an essential singularity (`exp(−1/x²)`, whose real
/// Taylor coefficients all vanish) is refused.
pub(crate) fn expand_meromorphic(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    order: i64,
) -> Result<TSeries, SymplexError> {
    expand_with_mode(
        arena,
        expr,
        var,
        order,
        Mode {
            side: Side::Both,
            log_var: None,
            limits: false,
            partial: false,
        },
    )
}

/// Log-extended expansion as `var → 0⁺` (see [`Mode::log_var`]): `log_var`
/// stands for `ln(var)` and may appear (polynomially) in the coefficients.
/// `limits` allows the differentiation fallback to take limits.
///
/// Used by [`series_at_infinity`] (`var = 1/x`, `log_var` a placeholder).
pub(crate) fn expand_log_extended(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    order: i64,
    log_var: ExprId,
    limits: bool,
) -> Result<TSeries, SymplexError> {
    expand_with_mode(
        arena,
        expr,
        var,
        order,
        Mode {
            side: Side::Above,
            log_var: Some(log_var),
            limits,
            partial: false,
        },
    )
}

/// [`expand_log_extended`] for a leading term: no limits in the
/// differentiation fallback, and a result exact to fewer than `order` terms
/// is returned rather than refused (its [`TSeries::known`] tells how far
/// it is exact).  Used by Gruntz's leading-term extraction (`var = ω`,
/// `log_var = ln ω`).
pub(crate) fn expand_leading(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    order: i64,
    log_var: ExprId,
) -> Result<TSeries, SymplexError> {
    expand_with_mode(
        arena,
        expr,
        var,
        order,
        Mode {
            side: Side::Above,
            log_var: Some(log_var),
            limits: false,
            partial: true,
        },
    )
}

/// Expand `expr` around `var = 0` with all exponents `< order` exact, in the
/// given [`Mode`].
///
/// From above ([`Side::Above`], used for `x → ±∞` via `t = 1/x`)
/// `(t^v)^α = t^{vα}` holds for every integer `vα`, which is not valid
/// two-sided (`√(x²) = |x|`).
///
/// A two-sided expansion that fails only because of `|g|` with odd
/// valuation (`|x|`, `|sin x|`) or a logarithmic singularity (`ln x`) is
/// decided from the two one-sided expansions, see [`expand_from_both_sides`].
fn expand_with_mode(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    order: i64,
    mode: Mode,
) -> Result<TSeries, SymplexError> {
    let mut split = false;
    match expand_from_side(arena, expr, var, order, mode, &mut split) {
        Err(_) if split => expand_from_both_sides(arena, expr, var, order, mode),
        r => r,
    }
}

/// The two-sided expansion of `expr` from its one-sided expansions, after
/// the direct one stopped at an odd-valuation `|g|` ([`Obstruction::Kink`])
/// or a logarithmic singularity ([`Obstruction::Logarithmic`]).
///
/// With limits (the real asymptotic expansion) both sides are
/// log-extended and taken to one order more: an expansion exists when no
/// coefficient of exponent `≤ order` contains `ln|x|` (so the remainder is
/// `O(x^order)` on each side) and the two agree below `order` —
/// `cos|x| = cos x`, `ln(sin x) − ln x = −x²/6 + …`.  Otherwise the refusal
/// is definite: `|sin x|`, `x^x = 1 + x·ln x + …`, `x³·ln x` at order 3.
/// (Before, a logarithmic singularity fell to the differentiation fallback,
/// which asked the limit engine for the limits of ever larger derivatives,
/// once more for each pole-retry multiplier: `series(sinh(x^x), x, 0, 3)`
/// took 0.4 s before refusing, 10 s in the nightly `fuzz_calculus` run.)
/// Without limits (a Laurent series) only a kink is retried, as before.
fn expand_from_both_sides(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    order: i64,
    mode: Mode,
) -> Result<TSeries, SymplexError> {
    let log_var = mode.limits.then(|| arena.symbol(LOG_PLACEHOLDER));
    let depth = order + i64::from(log_var.is_some());
    let mut expansions = Vec::with_capacity(2);
    for side in [Side::Above, Side::Below] {
        let side_mode = Mode {
            log_var,
            ..mode.with_side(side)
        };
        expansions.push(expand_from_side(
            arena, expr, var, depth, side_mode, &mut false,
        )?);
    }
    let no_expansion = |arena: &Arena, why: &str| SymplexError::ComputationFailed {
        operation: "series",
        reason: format!("no two-sided expansion of {}: {why}", arena.display(expr)),
    };
    if let Some(l) = log_var {
        for s in &mut expansions {
            for e in s.shift..s.known.min(order + 1) {
                let idx = (e - s.shift) as usize;
                let c = s.coeffs[idx];
                if !walk::contains(arena, c, l) {
                    continue;
                }
                // A polynomial in `l` whose terms cancel only once expanded.
                let x = crate::transforms::expand::expand(arena, c);
                let x = eval::eval(arena, x);
                if walk::contains(arena, x, l) {
                    return Err(no_expansion(
                        arena,
                        "a logarithmic singularity (a coefficient contains ln|x|)",
                    ));
                }
                s.coeffs[idx] = x;
            }
        }
    }
    let (above, below) = (&expansions[0], &expansions[1]);
    for e in above.shift.min(below.shift)..order {
        let (a, b) = (above.coeff_at(arena, e), below.coeff_at(arena, e));
        if a != b {
            let d = arena.sub(a, b);
            if !is_zero_const(arena, d) {
                return Err(no_expansion(
                    arena,
                    "the expansions from the left and from the right differ",
                ));
            }
        }
    }
    Ok(expansions.swap_remove(0).truncate_known(order))
}

/// [`expand_with_mode`] for one [`Side`], with the precision-escalation loop.
/// Sets `split` when a two-sided expansion failed in a way the one-sided
/// expansions may resolve (see [`expand_from_both_sides`]).
fn expand_from_side(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    order: i64,
    mode: Mode,
    split: &mut bool,
) -> Result<TSeries, SymplexError> {
    // Work with at least a few terms so that the valuation of every
    // sub-expression is visible (at precision 1 the variable itself would
    // truncate to nothing and `1/x` could not be expanded).
    let mut working = order.max(MIN_WORKING_ORDER);
    let mut last_err = None;
    let mut best: Option<TSeries> = None;
    for _attempt in 0..MAX_PRECISION_ATTEMPTS {
        let mut hidden_valuation = false;
        match expand_with_precision(
            arena,
            expr,
            var,
            working,
            mode,
            &mut hidden_valuation,
            split,
        ) {
            Ok(ts) if ts.known >= order => return Ok(ts.truncate_known(order)),
            Ok(ts) => {
                // Precision was lost through poles; increase and retry.
                working += order - ts.known + 1;
                if mode.partial && best.as_ref().is_none_or(|b| b.known < ts.known) {
                    best = Some(ts);
                }
            }
            Err(e) if hidden_valuation && !*split => {
                // A sub-expression's leading term lay beyond the working
                // window (e.g. `1/(x⁵ + x⁶)` at low order): widen and retry.
                last_err = Some(e);
                working = working * 2 + 4;
            }
            Err(e) => return Err(e),
        }
    }
    if let Some(ts) = best {
        return Ok(ts);
    }
    Err(last_err.unwrap_or(SymplexError::ComputationFailed {
        operation: "series",
        reason: "could not reach the requested order (deep pole)".into(),
    }))
}

/// One pass of the engine at working precision `n`.  Sets `hidden_valuation`
/// when some `var`-dependent sub-expression had *no* visible term at this
/// precision, so a failure may be curable by widening the window; sets
/// `split` when the failure was [`Obstruction::Kink`], or
/// [`Obstruction::Logarithmic`] in a two-sided real expansion.
fn expand_with_precision(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    n: i64,
    mode: Mode,
    hidden_valuation: &mut bool,
    split: &mut bool,
) -> Result<TSeries, SymplexError> {
    let order_ids = walk::post_order_ids(arena, expr);
    let mut cache: FxHashMap<ExprId, Option<TSeries>> = FxHashMap::default();
    let mut fallbacks_used = 0usize;
    for id in order_ids {
        if cache.contains_key(&id) {
            continue;
        }
        let ts = if !walk::contains(arena, id, var) {
            Some(TSeries::constant(arena, id, n))
        } else if id == var {
            Some(TSeries::var(arena, n))
        } else {
            match structural_series(arena, id, var, mode, &cache) {
                Ok(s) => Some(s),
                Err(Obstruction::Unknown) => {
                    match compose_at_constant(arena, id, var, mode, &cache) {
                        Some(composed) => composed,
                        None if id == expr || fallbacks_used < MAX_FALLBACK_NODES => {
                            fallbacks_used += 1;
                            taylor_by_differentiation(arena, id, var, n, mode)
                        }
                        None => None,
                    }
                }
                Err(Obstruction::Kink) => {
                    *split = true;
                    return Err(no_expansion_error(arena, id));
                }
                Err(Obstruction::Logarithmic) => {
                    *split |= mode.side == Side::Both && mode.limits;
                    return Err(no_expansion_error(arena, id));
                }
                Err(Obstruction::NoExpansion) => return Err(no_expansion_error(arena, id)),
            }
        };
        if let Some(s) = &ts
            && id != expr
            && s.clone().normalized(arena).coeffs.is_empty()
        {
            *hidden_valuation = true;
        }
        cache.insert(id, ts);
    }
    match cache.remove(&expr).flatten() {
        Some(ts) => Ok(ts),
        None => Err(SymplexError::ComputationFailed {
            operation: "series",
            reason: "no Laurent expansion at this point (singularity or unsupported function)"
                .into(),
        }),
    }
}

/// Stand-in for the argument of `f` in [`compose_at_constant`].
const COMPOSE_PLACEHOLDER: &str = "__series_u";

/// `f(u)` for a node with no structural rule and one `var`-dependent
/// argument whose expansion `u = u₀ + w` (`w → 0`) is known:
/// `Σ f⁽ᵏ⁾(u₀)/k!·wᵏ`, with the Taylor coefficients of `f` at the constant
/// `u₀` from the derivatives of `f(t)` — SymPy's `Function._eval_nseries`,
/// which also expands around an argument with a logarithm
/// (`f(1 + x + log x) → f(1 + logx) + x·f'(1 + logx)`).
///
/// Before, such a node fell to the differentiation fallback, which
/// differentiates the whole composite `f(u(x))` and asks the limit engine
/// for each derivative at the point: `series(acos(cosh(e^(−1/x²))/2), x,
/// 0, 3)` took seconds.  Where `w` contains `ln|x|` (a log-extended
/// expansion) the fallback cannot succeed at all — the derivatives of
/// `f(u(x))` contain `ln x` and diverge — and `series(atan(x^x), x, 0, 3)`
/// from the right took 1.9 s to refuse; it is `π/4 + x·ln(x)/2 + …`.
///
/// `None` when the rule does not apply (no such argument, `u₀` containing
/// `ln|x|`, a discontinuous `f`) or `f` is not visibly analytic at `u₀` (a
/// derivative singular there, `u₀` on a branch cut) and `w` has no
/// logarithm: the fallback is tried as before.  `Some(None)` when `f` is
/// not analytic at `u₀` and `w` contains `ln|x|`: no expansion of this form.
fn compose_at_constant(
    arena: &mut Arena,
    id: ExprId,
    var: ExprId,
    mode: Mode,
    cache: &FxHashMap<ExprId, Option<TSeries>>,
) -> Option<Option<TSeries>> {
    if matches!(
        arena.node(id),
        ExprNode::Add(_)
            | ExprNode::Mul(_)
            | ExprNode::Neg(_)
            | ExprNode::Abs(_)
            | ExprNode::Sign(_)
            | ExprNode::Heaviside(_)
            | ExprNode::DiracDelta(_)
            | ExprNode::Floor(_)
            | ExprNode::Ceiling(_)
            | ExprNode::Piecewise(_)
            | ExprNode::Min(_)
            | ExprNode::Max(_)
    ) || walk::has_unevaluated(arena, id)
    {
        return None;
    }
    let mut arg = None;
    for c in arena.children(id) {
        if walk::contains(arena, c, var) {
            if arg.is_some_and(|a| a != c) {
                return None;
            }
            arg = Some(c);
        }
    }
    let arg = arg?;
    let u = cache.get(&arg)?.as_ref()?.clone().normalized(arena);
    if u.shift < 0 {
        return None; // the argument tends to infinity
    }
    let (u0, w) = u.split_constant(arena);
    let has_log =
        |arena: &Arena, c: ExprId| mode.log_var.is_some_and(|l| walk::contains(arena, c, l));
    if has_log(arena, u0) {
        return None;
    }
    let definite = w.coeffs.iter().any(|&c| has_log(arena, c));
    let refuse = if definite { Some(None) } else { None };
    // On a branch cut of `f` the Taylor coefficients below are those along
    // the cut (the placeholder `t` counts as real): the expansion of the
    // side the principal value is continuous with.  An argument that leaves
    // the cut (`asin(2 + i·x)`, `i·x` crosses the cut `(1, ∞)`) has a
    // different expansion on the other side, the continuous one mapped by
    // the jump across the cut ([`cut_jump`]); the side is that of the first
    // coefficient of `w` off the line of the cut.  Up to 0.33 `limit(x/(asin(2
    // + i·x) − asin 2), x, 0)` was `√3` from both sides (it is `0` from the
    // right, `√3` from the left; SymPy agrees), from 0.34 to 0.36 such an
    // expansion was refused.  Coefficients count as real when they are for
    // real values of their symbols (`ln|x|`, parameters).
    // A real `w` crosses a cut on the imaginary axis (`atan(2i + x)`).
    let mut jump = None;
    if let Some((_, cut)) = crate::calculus::limit::branch_cut_of(arena, id)
        && crate::calculus::limit::on_branch_cut(arena, cut, u0)
        && (cut == crate::calculus::limit::BranchCut::ImaginaryAxis
            || !w
                .coeffs
                .iter()
                .all(|&c| crate::calculus::limit::inner_known_real(arena, c)))
    {
        let imaginary_axis = cut == crate::calculus::limit::BranchCut::ImaginaryAxis;
        let mut side = series_cut_side(arena, &w, imaginary_axis, mode);
        // Every visible coefficient along the cut: the argument must be
        // shown to stay on it (a term beyond the precision could leave it).
        if side == Some(0) && !stays_on_line(arena, arg, u0, var, imaginary_axis, mode) {
            side = None;
        }
        match side.and_then(|s| cut_jump(arena, id, u0, s)) {
            Some(j) => jump = Some(j),
            None => return Some(None),
        }
    }
    // `wᵏ` has valuation `≥ k·v_w`: terms up to `k·v_w < known` count.
    let terms = match w.leading_exponent(arena) {
        Some(v_w) => usize::try_from((w.known - 1) / v_w.max(1)).ok()?,
        None => 0,
    };
    let t = arena.symbol(COMPOSE_PLACEHOLDER);
    let mut f = subs::subs(arena, id, arg, t);
    if walk::contains(arena, f, var) {
        return None;
    }
    let mut coeffs = Vec::with_capacity(terms + 1);
    let mut factorial = Q::one();
    for k in 0..=terms {
        if k > 0 {
            f = crate::transforms::diff::diff(arena, f, t);
            factorial *= rat_i(k as i64);
        }
        if walk::has_unevaluated(arena, f)
            || crate::transforms::pattern::tree_size_capped(arena, f, MAX_FALLBACK_SIZE + 1)
                > MAX_FALLBACK_SIZE
        {
            return refuse;
        }
        let Some(value) = crate::calculus::limit::safe_substitute(arena, f, t, u0) else {
            return refuse;
        };
        if !is_finite_constant(arena, value, var) || walk::contains(arena, value, t) {
            return refuse;
        }
        let inv = rat_expr(arena, Q::one() / &factorial);
        let c = arena.mul(&[value, inv]);
        coeffs.push(eval::eval(arena, c));
    }
    let composed = TSeries::compose(
        arena,
        &|ar: &mut Arena, k: usize| coeffs.get(k).copied().unwrap_or(ar.zero),
        &w,
    );

    Some(Some(match jump {
        Some((a, b)) => {
            let scaled = TSeries::scale(arena, &composed, b);
            let c = constant_like(arena, a, &scaled);
            TSeries::add(arena, &c, &scaled)
        }
        None => composed,
    }))
}

/// Is `u − u₀` exactly real (`imaginary_axis`: exactly imaginary) for real
/// values of `var` on the side of the expansion?  (`2i + ix` stays on the
/// line `iℝ` of the cuts of `asinh`, `atan`.)
fn stays_on_line(
    arena: &mut Arena,
    u: ExprId,
    u0: ExprId,
    var: ExprId,
    imaginary_axis: bool,
    mode: Mode,
) -> bool {
    let t = arena.positive_symbol(SIDE_PLACEHOLDER);
    let signs: &[bool] = match mode.side {
        Side::Above => &[false],
        Side::Below => &[true],
        Side::Both => &[false, true],
    };
    let d = arena.sub(u, u0);
    signs.iter().all(|&negative| {
        let v = if negative { arena.neg(t) } else { t };
        let d = subs::subs(arena, d, var, v);
        let d = eval::eval(arena, d);
        let parts = crate::base::complex::decompose(arena, d);
        if !parts.exact {
            return false;
        }
        let p = if imaginary_axis { parts.re } else { parts.im };
        let p = eval::eval(arena, p);
        if arena.is_zero_structural(p) {
            return true;
        }
        // `ln(t²)/2 − ln|t|` (`|var| = t > 0`) is zero once `|t| = t` and
        // the logarithms are expanded.
        let mut assumptions = crate::base::assumptions::AssumptionCache::new();
        let p = crate::simplify::refine::refine_full(arena, &mut assumptions, p);
        let p = crate::simplify::log_expand::expand_log(arena, p);
        let p = crate::transforms::expand::expand(arena, p);
        let p = eval::eval(arena, p);
        arena.is_zero_structural(p)
    })
}

/// A positive stand-in for `|var|` in [`stays_on_line`].
const SIDE_PLACEHOLDER: &str = "__series_side";

/// The side of a branch cut from which `u₀ + w` leaves the point `u₀` of
/// the cut: the sign of the component perpendicular to the cut (`Im`, or
/// `Re` for the cuts on the imaginary axis) of the first coefficient of `w`
/// that has one, times the sign of `varᵏ` on the expansion's side.  `0`
/// when every visible coefficient lies along the cut.  `None` when that
/// sign is not known: a symbolic or logarithmic coefficient, or an odd
/// power in a two-sided expansion (the two sides differ).
fn series_cut_side(
    arena: &mut Arena,
    w: &TSeries,
    imaginary_axis: bool,
    mode: Mode,
) -> Option<i32> {
    for (i, &c) in w.coeffs.iter().enumerate() {
        let k = w.shift + i as i64;
        if k >= w.known {
            break;
        }
        if arena.is_zero_structural(c) {
            continue;
        }
        let parts = crate::base::complex::decompose(arena, c);
        let p = if imaginary_axis { parts.re } else { parts.im };
        let p = eval::eval(arena, p);
        if parts.exact && arena.is_zero_structural(p) {
            continue;
        }
        if !parts.exact || !walk::free_symbols(arena, p).is_empty() {
            return None;
        }
        let s = crate::calculus::limit::const_sign(arena, p)?;
        if s == 0 {
            continue;
        }
        let odd = k % 2 != 0;
        return match mode.side {
            Side::Above => Some(s),
            Side::Below => Some(if odd { -s } else { s }),
            Side::Both if !odd => Some(s),
            Side::Both => None,
        };
    }
    Some(0)
}

/// The jump of a branch function `id` across its cut at `u₀` for an
/// argument on the side `side` of the cut ([`series_cut_side`]): `(a, b)`
/// with `f = a + b·F` near `u₀`, `F` the continuation of `f` from the side
/// its principal value is continuous with — the expansion from the Taylor
/// coefficients at `u₀`.  These are the closed forms of
/// [`continuation_across_cut`](crate::calculus::limit::continuation_across_cut)
/// related to each other (SymPy's `asin._eval_nseries`: `π − asin` above
/// `(1, ∞)`, `−π − asin` below `(−∞, −1)`; `log._eval_nseries`: `−2πi`
/// below): `ln`: `F − 2πi` below; `u^α`: `e^{−2πiα}·F` below; `acos`: `−F`
/// above `(1, ∞)`, `2π − F` below `(−∞, −1)`; `atanh`: `F ± iπ`; `acosh`:
/// `−F` below `(−1, 1)`, `F − 2πi` below `(−∞, −1)`; `asinh`: `±iπ − F`;
/// `atan`: `F ∓ π`; `Ei`: `F ± iπ` on either side (its value on the axis is
/// the mean); `Ci`, `Chi`: `F − 2πi` below; `ln Γ`: `F + 2πi(n + 1)` below
/// `(−n − 1, −n)`.  `None` where no form is known (`W`, `polylog`, …) or
/// at a branch point.
fn cut_jump(arena: &mut Arena, id: ExprId, u0: ExprId, side: i32) -> Option<(ExprId, ExprId)> {
    use crate::base::libfn::LibFn;
    use crate::calculus::limit::const_sign;
    let zero = arena.zero;
    let one = arena.one;
    let m_one = arena.neg_one;
    let identity = Some((zero, one));
    if side == 0 {
        return identity;
    }
    let pi = arena.pi;
    let i = arena.i_unit;
    let two = arena.int(2);
    let i_pi = arena.mul(&[i, pi]);
    let two_pi_i = arena.mul(&[two, i_pi]);
    let two_pi = arena.mul(&[two, pi]);
    let neg = |arena: &mut Arena, v: ExprId| arena.neg(v);
    let re = arena.re(u0);
    let re = eval::eval(arena, re);
    let im = arena.im(u0);
    let im = eval::eval(arena, im);
    let above = side > 0;
    let r = match arena.node(id).clone() {
        ExprNode::Ln(_) => {
            if above {
                return identity;
            }
            (neg(arena, two_pi_i), one)
        }
        ExprNode::Pow(_, alpha) => {
            if above {
                return identity;
            }
            let p = arena.mul(&[two_pi_i, alpha]);
            let p = neg(arena, p);
            let e = arena.exp(p);
            (zero, eval::eval(arena, e))
        }
        ExprNode::Asin(_) | ExprNode::Acos(_) => {
            let positive = const_sign(arena, re)? > 0;
            // continuous: below `(1, ∞)`, above `(−∞, −1)`
            if above != positive {
                return identity;
            }
            match (matches!(arena.node(id), ExprNode::Asin(_)), positive) {
                (true, true) => (pi, m_one),
                (true, false) => (neg(arena, pi), m_one),
                (false, true) => (zero, m_one),
                (false, false) => (two_pi, m_one),
            }
        }
        ExprNode::Atanh(_) => {
            let positive = const_sign(arena, re)? > 0;
            if above != positive {
                return identity;
            }
            (if above { i_pi } else { neg(arena, i_pi) }, one)
        }
        ExprNode::Acosh(_) => {
            if above {
                return identity;
            }
            let rp1 = arena.add(&[re, one]);
            match const_sign(arena, rp1)? {
                s if s > 0 => (zero, m_one),
                s if s < 0 => (neg(arena, two_pi_i), one),
                _ => return None,
            }
        }
        ExprNode::Asinh(_) | ExprNode::Atan(_) => {
            let upper = const_sign(arena, im)? > 0;
            let abs_y = if upper { im } else { neg(arena, im) };
            let d = arena.sub(abs_y, one);
            if const_sign(arena, d)? <= 0 {
                return None;
            }
            // continuous: `Re > 0` for the upper cut, `Re < 0` for the lower
            if above == upper {
                return identity;
            }
            if matches!(arena.node(id), ExprNode::Atan(_)) {
                (if upper { neg(arena, pi) } else { pi }, one)
            } else {
                (if upper { i_pi } else { neg(arena, i_pi) }, m_one)
            }
        }
        ExprNode::Ei(_) => (if above { i_pi } else { neg(arena, i_pi) }, one),
        ExprNode::Ci(_) => {
            if above {
                return identity;
            }
            (neg(arena, two_pi_i), one)
        }
        ExprNode::Apply(f, _) if arena.lib_fn(f) == Some(LibFn::Chi) => {
            if above {
                return identity;
            }
            (neg(arena, two_pi_i), one)
        }
        ExprNode::LogGamma(_) => {
            if above {
                return identity;
            }
            let n = crate::calculus::limit::floor_of_negated(arena, u0)?;
            let k = arena.int(n + 1);
            (arena.mul(&[k, two_pi_i]), one)
        }
        _ => return None,
    };
    let a = eval::eval(arena, r.0);
    Some((a, r.1))
}

fn no_expansion_error(arena: &Arena, id: ExprId) -> SymplexError {
    SymplexError::ComputationFailed {
        operation: "series",
        reason: format!(
            "no Laurent expansion of {} at this point (fractional power, |·| or non-real argument)",
            arena.display(id)
        ),
    }
}

/// The cached series of a child node, or [`Obstruction::Unknown`] when the
/// child itself could not be expanded.
fn child(cache: &FxHashMap<ExprId, Option<TSeries>>, id: ExprId) -> Result<TSeries, Obstruction> {
    cache
        .get(&id)
        .cloned()
        .flatten()
        .ok_or(Obstruction::Unknown)
}

/// Combine children's series according to the node type.
fn structural_series(
    arena: &mut Arena,
    id: ExprId,
    var: ExprId,
    mode: Mode,
    cache: &FxHashMap<ExprId, Option<TSeries>>,
) -> Result<TSeries, Obstruction> {
    let side = mode.side;
    let node = arena.node(id).clone();
    let analytic = match node {
        ExprNode::Add(ref ch) => {
            let mut acc: Option<TSeries> = None;
            for &c in ch.iter() {
                let s = child(cache, c)?;
                acc = Some(match acc {
                    None => s,
                    Some(a) => TSeries::add(arena, &a, &s),
                });
            }
            acc
        }
        ExprNode::Mul(ref ch) => {
            let mut acc: Option<TSeries> = None;
            for &c in ch.iter() {
                let s = child(cache, c)?;
                acc = Some(match acc {
                    None => s,
                    Some(a) => TSeries::mul(arena, &a, &s),
                });
            }
            acc
        }
        ExprNode::Neg(inner) => {
            let s = child(cache, inner)?;
            Some(TSeries::neg(arena, &s))
        }
        ExprNode::Pow(base, exp) => {
            let b = child(cache, base)?;
            if !walk::contains(arena, exp, var) {
                let e = arena.as_num(exp).cloned().ok_or(Obstruction::Unknown)?;
                if e.is_integer() {
                    let ei = e.to_integer().to_i64().ok_or(Obstruction::Unknown)?;
                    if ei.abs() <= MAX_INT_POWER {
                        return TSeries::pow_int(arena, &b, ei).ok_or(Obstruction::Unknown);
                    }
                    // Large integer exponents: binomial series with
                    // closed-form coefficients instead of repeated products.
                }
                // Rational exponent: Puiseux expansions are refused.
                return pow_rational(arena, &b, exp, side);
            }
            // b^e with var-dependent exponent: exp(e · ln b).
            let e = child(cache, exp)?;
            let lnb = apply_ln(arena, &b, mode)?;
            let prod = TSeries::mul(arena, &e, &lnb);
            apply_exp_like(arena, FnKind::Exp, &prod, mode)
        }
        ExprNode::Abs(a) => return abs_series(arena, &child(cache, a)?, side),
        ExprNode::Sign(a) => return sign_series(arena, &child(cache, a)?, side, false),
        ExprNode::Heaviside(a) => return sign_series(arena, &child(cache, a)?, side, true),
        ExprNode::Apply(f, ref args) if args.len() == 2 && is_bessel(arena, f) => {
            let u = child(cache, args[1])?;
            if let Some(s) = log_bessel_series(arena, f, args[0], &u, var, mode)? {
                return Ok(s);
            }
            return bessel_series(arena, f, args[0], &u, var);
        }
        ExprNode::Gamma(_)
        | ExprNode::LogGamma(_)
        | ExprNode::Digamma(_)
        | ExprNode::Polygamma(..)
        | ExprNode::Zeta(_)
        | ExprNode::Ei(_)
        | ExprNode::Ci(_)
        | ExprNode::Si(_)
        | ExprNode::Li(_)
        | ExprNode::Erfc(_) => return special_series(arena, &node, mode, cache),
        ExprNode::Apply(f, ref args)
            if (args.len() == 1
                && (matches!(
                    arena.lib_fn(f),
                    Some(crate::base::libfn::LibFn::Shi | crate::base::libfn::LibFn::Chi)
                ) || matches!(arena.symbol_name(f), ERFCX_ASYMPTOTIC | EI_ASYMPTOTIC)))
                || PowerAsymptotic::from_name(arena.symbol_name(f)).is_some() =>
        {
            return special_series(arena, &node, mode, cache);
        }
        ExprNode::Exp(a) => apply_exp_like(arena, FnKind::Exp, &child(cache, a)?, mode),
        ExprNode::Sin(a) => apply_fn(arena, FnKind::Sin, &child(cache, a)?),
        ExprNode::Cos(a) => apply_fn(arena, FnKind::Cos, &child(cache, a)?),
        ExprNode::Sinh(a) => apply_exp_like(arena, FnKind::Sinh, &child(cache, a)?, mode),
        ExprNode::Cosh(a) => apply_exp_like(arena, FnKind::Cosh, &child(cache, a)?, mode),
        ExprNode::Tan(a) => apply_fn(arena, FnKind::Tan, &child(cache, a)?),
        ExprNode::Tanh(a) => apply_fn(arena, FnKind::Tanh, &child(cache, a)?),
        ExprNode::Atan(a) => {
            let u = child(cache, a)?;
            if let Some(s) = inverse_at_branch_point(arena, InverseFn::Atan, &u, mode)? {
                return Ok(s);
            }
            apply_fn(arena, FnKind::Atan, &u)
        }
        ExprNode::Atanh(a) => {
            let u = child(cache, a)?;
            if let Some(s) = inverse_at_branch_point(arena, InverseFn::Atanh, &u, mode)? {
                return Ok(s);
            }
            apply_fn(arena, FnKind::Atanh, &u)
        }
        ExprNode::Asin(a) => {
            let u = child(cache, a)?;
            if let Some(s) = inverse_at_branch_point(arena, InverseFn::Asin, &u, mode)? {
                return Ok(s);
            }
            apply_fn(arena, FnKind::Asin, &u)
        }
        ExprNode::Asinh(a) => {
            let u = child(cache, a)?;
            if let Some(s) = inverse_at_branch_point(arena, InverseFn::Asinh, &u, mode)? {
                return Ok(s);
            }
            apply_fn(arena, FnKind::Asinh, &u)
        }
        // Only their branch points have a structural rule; elsewhere the
        // differentiation fallback applies.
        ExprNode::Acos(a) => {
            let u = child(cache, a)?;
            return inverse_at_branch_point(arena, InverseFn::Acos, &u, mode)?
                .ok_or(Obstruction::Unknown);
        }
        ExprNode::Acosh(a) => {
            let u = child(cache, a)?;
            return inverse_at_branch_point(arena, InverseFn::Acosh, &u, mode)?
                .ok_or(Obstruction::Unknown);
        }
        ExprNode::Erf(a) => apply_fn(arena, FnKind::Erf, &child(cache, a)?),
        ExprNode::LambertW(a) => apply_fn(arena, FnKind::LambertW, &child(cache, a)?),
        ExprNode::Ln(a) => return apply_ln(arena, &child(cache, a)?, mode),
        _ => None,
    };
    analytic.ok_or(Obstruction::Unknown)
}

/// `exp`, `sinh`, `cosh` of `a`.  In a log-extended expansion (see
/// [`Mode::log_var`]) a constant term `m·l + r` with an integer `m` gives
/// `e^(m·l) = |var|^m` (`x^(x + 2) = x²·e^(x·ln x)`, `sinh(ln x) =
/// (x − 1/x)/2`); any other dependence on `l` (`e^(l/2)`, `e^(l²)`) is
/// refused: that coefficient would not be of lower order.
fn apply_exp_like(arena: &mut Arena, kind: FnKind, a: &TSeries, mode: Mode) -> Option<TSeries> {
    let Some(l) = mode.log_var else {
        return apply_fn(arena, kind, a);
    };
    let u0 = a.coeff_at(arena, 0);
    if !walk::contains(arena, u0, l) {
        return apply_fn(arena, kind, a);
    }
    let m = crate::transforms::diff::diff(arena, u0, l);
    let m = eval::eval(arena, m);
    let m = arena
        .as_num(m)
        .filter(|q| q.is_integer())?
        .to_integer()
        .to_i64()?;
    let zero = arena.zero;
    let r = subs::subs(arena, u0, l, zero);
    let r = eval::eval(arena, r);
    if walk::contains(arena, r, l) || m.abs() > MAX_INT_POWER {
        return None;
    }
    let mut b = a.clone();
    b.set_constant(r);
    // e^(±(m·l + b)) = (±|var|)^(±m)·e^(±b), with |var|^m = (−1)^m·var^m below.
    let sign_flip = mode.side == Side::Below && m % 2 != 0;
    let shifted_exp = |arena: &mut Arena, b: &TSeries, m: i64| -> Option<TSeries> {
        let mut s = apply_fn(arena, FnKind::Exp, b)?;
        if sign_flip {
            s = TSeries::neg(arena, &s);
        }
        s.shift += m;
        s.known += m;
        Some(s)
    };
    let plus = shifted_exp(arena, &b, m)?;
    if kind == FnKind::Exp {
        return Some(plus);
    }
    let neg_b = TSeries::neg(arena, &b);
    let minus = shifted_exp(arena, &neg_b, -m)?;
    let minus = if kind == FnKind::Sinh {
        TSeries::neg(arena, &minus)
    } else {
        minus
    };
    let sum = TSeries::add(arena, &plus, &minus);
    let half = arena.rational(1, 2);
    Some(TSeries::scale(arena, &sum, half))
}

fn is_bessel(arena: &Arena, f: crate::base::node::SymbolId) -> bool {
    use crate::base::libfn::LibFn;
    matches!(
        arena.lib_fn(f),
        Some(LibFn::BesselJ | LibFn::BesselY | LibFn::BesselI | LibFn::BesselK)
    )
}

/// A Bessel function of the series `u` at a zero of `u`: `J_n(u)` and
/// `I_n(u)` of integer order from `J_n(z) = Σ_k (−1)^k (z/2)^{2k+n}/(k!(k+n)!)`
/// (`I_n` without the signs; `J_{−n} = (−1)ⁿ J_n`, `I_{−n} = I_n`); `Y_ν`,
/// `K_ν` (logarithmic singularity) and `J_ν`, `I_ν` of any other order (a
/// fractional power, or unknown) have no Laurent expansion there.  Before,
/// the differentiation fallback returned the coefficients of `Y₀` in terms
/// of `Y₀(0) = −∞` as if they were finite.  An argument with a pole is an
/// essential singularity; elsewhere the fallback (finite values) applies.
fn bessel_series(
    arena: &mut Arena,
    f: crate::base::node::SymbolId,
    order: ExprId,
    u: &TSeries,
    var: ExprId,
) -> Result<TSeries, Obstruction> {
    use crate::base::libfn::LibFn;
    let u = u.clone().normalized(arena);
    let Some(k) = u.leading_exponent(arena) else {
        return Err(Obstruction::NoExpansion);
    };
    if k < 0 {
        return Err(Obstruction::NoExpansion);
    }
    if k == 0 {
        return Err(Obstruction::Unknown);
    }
    let lib = arena.lib_fn(f).ok_or(Obstruction::Unknown)?;
    if walk::contains(arena, order, var) || !matches!(lib, LibFn::BesselJ | LibFn::BesselI) {
        return Err(Obstruction::NoExpansion);
    }
    let n = arena
        .as_num(order)
        .filter(|r| r.is_integer())
        .and_then(|r| r.to_integer().to_i64())
        .ok_or(Obstruction::NoExpansion)?;
    let alternate = lib == LibFn::BesselJ;
    let reflect_sign = alternate && n < 0 && n % 2 != 0;
    let n = n.unsigned_abs();
    let coefficient = move |ar: &mut Arena, m: usize| -> ExprId {
        let m = m as u64;
        if m < n || (m - n) % 2 == 1 {
            return ar.zero;
        }
        let kk = (m - n) / 2;
        let denom = factorial(kk) * factorial(kk + n) * (BigInt::one() << (2 * kk + n));
        let mut c = Q::new(BigInt::one(), denom);
        if alternate && kk % 2 == 1 {
            c = -c;
        }
        if reflect_sign {
            c = -c;
        }
        rat_expr(ar, c)
    };
    Ok(TSeries::compose(arena, &coefficient, &u))
}

/// `|g|` for a series `g` with leading term `c·x^k`, `c` a real constant of
/// known sign and all visible coefficients real.
///
/// Near `0` the sign of `g` is that of `c·x^k`, so `|g| = ±g`:
///
/// * `k` even (including `k = 0`): `sign(c)·g` on both sides;
/// * `k` odd, one-sided: `sign(c)·g` from above, `−sign(c)·g` from below;
/// * `k` odd, two-sided: [`Obstruction::Kink`] — no expansion of `|g|`
///   itself, but the caller re-expands the whole expression from each side.
///
/// A constant term of unknown sign (`|a + x|`) is left to the
/// differentiation fallback (`|a| + sign(a)·x`, correct for real `a ≠ 0`);
/// an unknown sign with `k ≠ 0`, or a non-real coefficient, is a definite
/// [`Obstruction::NoExpansion`] — differentiating `|g|` at a zero of `g`
/// gives `sign(0) = 0` and hence a wrong all-zero polynomial.
fn abs_series(arena: &mut Arena, g: &TSeries, side: Side) -> Result<TSeries, Obstruction> {
    let g = g.clone().normalized(arena);
    // No visible term: the valuation is hidden at this precision.  Fail
    // definitively so that the caller widens the window instead of
    // differentiating.
    let k = g.leading_exponent(arena).ok_or(Obstruction::NoExpansion)?;
    let c = g.coeff_at(arena, k);
    let Some(c_positive) = constant_sign(arena, c) else {
        return Err(if k == 0 {
            Obstruction::Unknown
        } else {
            Obstruction::NoExpansion
        });
    };
    if !g.coeffs.iter().all(|&co| is_known_real(arena, co)) {
        return Err(Obstruction::NoExpansion);
    }
    let flip_below = k.rem_euclid(2) == 1;
    let positive = match side {
        Side::Both if flip_below => return Err(Obstruction::Kink),
        Side::Below if flip_below => !c_positive,
        Side::Both | Side::Above | Side::Below => c_positive,
    };
    Ok(if positive { g } else { TSeries::neg(arena, &g) })
}

/// `sign(g)` (or `H(g)` when `heaviside`) for a series `g` whose leading
/// term `c·x^k` has a real constant `c` of known sign and whose visible
/// coefficients are real: near `0` the sign of `g` is that of `c·x^k` (see
/// [`abs_series`] for the sides), a constant `±1` (`1` or `0`) exact to any
/// order.
///
/// Anything else is a definite [`Obstruction::NoExpansion`]: the
/// differentiation fallback evaluates `sign(g(0)) = sign(0) = 0` and
/// `d sign(g) = 0`, the wrong polynomial `0` — `atanh(x²)·sign(x²)` expanded
/// to `0` instead of `x²`.
fn sign_series(
    arena: &mut Arena,
    g: &TSeries,
    side: Side,
    heaviside: bool,
) -> Result<TSeries, Obstruction> {
    let g = g.clone().normalized(arena);
    let k = g.leading_exponent(arena).ok_or(Obstruction::NoExpansion)?;
    let c = g.coeff_at(arena, k);
    // A non-zero constant term of unknown sign: `sign(a + x)` is
    // `sign(a)` near 0, which the differentiation fallback finds.
    let Some(c_positive) = constant_sign(arena, c) else {
        return Err(if k == 0 {
            Obstruction::Unknown
        } else {
            Obstruction::NoExpansion
        });
    };
    if !g.coeffs.iter().all(|&co| is_known_real(arena, co)) {
        return Err(Obstruction::NoExpansion);
    }
    let flip_below = k.rem_euclid(2) == 1;
    let positive = match side {
        Side::Both if flip_below => return Err(Obstruction::Kink),
        Side::Below if flip_below => !c_positive,
        Side::Both | Side::Above | Side::Below => c_positive,
    };
    let value = match (positive, heaviside) {
        (true, _) => arena.one,
        (false, false) => arena.neg_one,
        (false, true) => arena.zero,
    };
    Ok(TSeries::constant(arena, value, g.known.max(1)))
}

/// `Some(true)` if the var-free constant `c` is known positive, `Some(false)`
/// if known negative, `None` if zero or of unknown sign.
fn constant_sign(arena: &mut Arena, c: ExprId) -> Option<bool> {
    if let Some(r) = arena.as_num(c) {
        return if r.is_zero() {
            None
        } else {
            Some(r.is_positive())
        };
    }
    let mut cache = crate::base::assumptions::AssumptionCache::new();
    if cache.query(arena, c, Props::POSITIVE) == Some(true) {
        return Some(true);
    }
    if cache.query(arena, c, Props::NEGATIVE) == Some(true) {
        return Some(false);
    }
    // A real constant the assumption system leaves open (`ln 3`,
    // `cos(3/2)`): its certified value decides.
    match crate::calculus::limit::const_sign(arena, c) {
        Some(1) => Some(true),
        Some(-1) => Some(false),
        _ => None,
    }
}

/// Is the var-free constant `c` known to be real?
fn is_known_real(arena: &Arena, c: ExprId) -> bool {
    arena.as_num(c).is_some() || {
        let mut cache = crate::base::assumptions::AssumptionCache::new();
        cache.query(arena, c, Props::REAL) == Some(true)
    }
}

fn is_zero_const(arena: &mut Arena, c: ExprId) -> bool {
    let v = settle_constant(arena, c);
    arena.is_zero_structural(v) || arena.as_num(v).is_some_and(|r| r.is_zero())
}

/// `f(a)` for an elementary `f` with known Maclaurin series.
fn apply_fn(arena: &mut Arena, kind: FnKind, a: &TSeries) -> Option<TSeries> {
    let a = a.clone().normalized(arena);
    if a.shift < 0 && a.leading_exponent(arena).is_some_and(|v| v < 0) {
        return None; // essential singularity
    }
    let (u0, w) = a.split_constant(arena);
    let u0_zero = is_zero_const(arena, u0);
    let compose = |arena: &mut Arena, k: FnKind, w: &TSeries| -> TSeries {
        TSeries::compose(
            arena,
            &move |ar: &mut Arena, i: usize| k.coefficient(ar, i),
            w,
        )
    };
    match kind {
        FnKind::Exp => {
            let s = compose(arena, FnKind::Exp, &w);
            if u0_zero {
                Some(s)
            } else {
                let e = arena.exp(u0);
                let e = eval::eval(arena, e);
                Some(TSeries::scale(arena, &s, e))
            }
        }
        FnKind::Sin | FnKind::Cos => {
            let sw = compose(arena, FnKind::Sin, &w);
            let cw = compose(arena, FnKind::Cos, &w);
            if u0_zero {
                return Some(if kind == FnKind::Sin { sw } else { cw });
            }
            let su = arena.sin(u0);
            let su = eval::eval(arena, su);
            let cu = arena.cos(u0);
            let cu = eval::eval(arena, cu);
            let (t1, t2) = if kind == FnKind::Sin {
                // sin(u0 + w) = sin u0 cos w + cos u0 sin w
                (
                    TSeries::scale(arena, &cw, su),
                    TSeries::scale(arena, &sw, cu),
                )
            } else {
                // cos(u0 + w) = cos u0 cos w − sin u0 sin w
                let nsu = arena.neg(su);
                (
                    TSeries::scale(arena, &cw, cu),
                    TSeries::scale(arena, &sw, nsu),
                )
            };
            Some(TSeries::add(arena, &t1, &t2))
        }
        FnKind::Sinh | FnKind::Cosh => {
            let sw = compose(arena, FnKind::Sinh, &w);
            let cw = compose(arena, FnKind::Cosh, &w);
            if u0_zero {
                return Some(if kind == FnKind::Sinh { sw } else { cw });
            }
            let su = arena.sinh(u0);
            let su = eval::eval(arena, su);
            let cu = arena.cosh(u0);
            let cu = eval::eval(arena, cu);
            let (t1, t2) = if kind == FnKind::Sinh {
                (
                    TSeries::scale(arena, &cw, su),
                    TSeries::scale(arena, &sw, cu),
                )
            } else {
                (
                    TSeries::scale(arena, &cw, cu),
                    TSeries::scale(arena, &sw, su),
                )
            };
            Some(TSeries::add(arena, &t1, &t2))
        }
        // `ln` is dispatched to `apply_ln` (it needs the constant term split
        // differently); a direct `Ln1p` request has no meaning here.
        FnKind::Ln1p => None,
        _ => {
            if !u0_zero {
                return None; // fallback: differentiate
            }
            Some(compose(arena, kind, &w))
        }
    }
}

/// The inverse trigonometric and hyperbolic functions, for
/// [`inverse_at_branch_point`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InverseFn {
    Asin,
    Acos,
    Atan,
    Asinh,
    Acosh,
    Atanh,
}

/// The expansion of an inverse trigonometric or hyperbolic function whose
/// argument `u` tends to one of its branch points: `±1` for `asin`,
/// `acos`, `acosh`, `atanh`, `±i` for `asinh`, `atan`.  `Ok(None)` when
/// `u` tends elsewhere (the other rules apply).
///
/// Follows SymPy's `asin._eval_nseries` / `acos._eval_nseries`, which put
/// `1 ∓ u = t²` at the branch points, and the `rewrite(log)` of
/// `atanh._eval_nseries` / `acosh._eval_nseries`; here in closed form, with
/// `F(z) = ₂F₁(½, ½; 3/2; z) = Σ C(2k, k)·zᵏ/(4ᵏ(2k + 1))`
/// (`asin √z = √z·F(z)`) and `s` the distance of `u` from the point:
///
/// * `asin u = π/2 − √(2s)·F(s/2)`, `s = 1 − u`;
///   `asin u = −π/2 + √(2s)·F(s/2)`, `s = 1 + u`; `acos = π/2 − asin`;
/// * `acosh u = √(2s)·F(−s/2)`, `s = u − 1`;
///   `acosh u = iπ − i√(2s)·F(s/2)`, `s = u + 1` (for real `s` only: the
///   cut of `acosh` is `(−∞, 1)`);
/// * `atanh u = (ln(1 + u) − ln(1 − u))/2`;
/// * `asinh u = −i·asin(iu)`, `atan u = −i·atanh(iu)`.
///
/// Each identity holds for the principal branches on a neighbourhood of
/// the point minus the cut (checked against mpmath at real and complex
/// arguments on both sides).  So the square root (a Puiseux exponent, or
/// `|x|` in a two-sided expansion) or the logarithm (outside a
/// log-extended expansion) makes the answer a definite
/// [`Obstruction::NoExpansion`] — and where `s` vanishes to an order that
/// makes `√(2s)` a power series (`asin(1 − x⁴) = π/2 − √2·x² − …`) the
/// expansion is exact.  Before 0.32 these points fell to the
/// differentiation fallback, whose derivatives all diverge: it asked the
/// limit engine for the limit of ever larger derivatives, once more for
/// each pole-retry multiplier, and `series(asin(x³), x, 1, 4)` took 42 s
/// before refusing (`fuzz_calculus` timed out on `asin(x²)`).
fn inverse_at_branch_point(
    arena: &mut Arena,
    kind: InverseFn,
    u: &TSeries,
    mode: Mode,
) -> Result<Option<TSeries>, Obstruction> {
    let u = u.clone().normalized(arena);
    if u.leading_exponent(arena).is_some_and(|v| v < 0) {
        return Ok(None); // the argument tends to infinity
    }
    let u0 = u.coeff_at(arena, 0);
    let one = arena.one;
    let neg_one = arena.neg_one;
    let i = arena.i_unit;
    let neg_i = {
        let n = arena.neg(i);
        eval::eval(arena, n)
    };
    // The branch point as written (`asin(sin 1)` for 1 is not), so that the
    // structural tests below see it.
    let targets: &[ExprId] = match kind {
        InverseFn::Asinh | InverseFn::Atan => &[i, neg_i],
        _ => &[one, neg_one],
    };
    let u0 = match targets.iter().find(|&&t| equals_constant(arena, u0, t)) {
        Some(&t) => t,
        None => return Ok(None),
    };
    let mut u = u;
    u.set_constant(u0);
    match kind {
        InverseFn::Asin | InverseFn::Acos => {
            let at_plus = if u0 == one {
                true
            } else if u0 == neg_one {
                false
            } else {
                return Ok(None);
            };
            // s = 1 ∓ u, of valuation ≥ 1.
            let signed = if at_plus {
                TSeries::neg(arena, &u)
            } else {
                u.clone()
            };
            let c1 = TSeries::constant(arena, one, u.known);
            let s = TSeries::add(arena, &c1, &signed).normalized(arena);
            let g = sqrt2s_times_f(arena, &s, false, mode.side)?;
            let half_pi = {
                let h = rat_expr(arena, Q::new(BigInt::one(), BigInt::from(2)));
                let p = arena.mul(&[arena.pi, h]);
                eval::eval(arena, p)
            };
            // asin at +1: π/2 − g; at −1: −π/2 + g.  acos = π/2 − asin.
            let (constant, g_sign_negative) = match (kind, at_plus) {
                (InverseFn::Asin, true) => (half_pi, true),
                (InverseFn::Asin, false) => (arena.neg(half_pi), false),
                (_, true) => (arena.zero, false),
                (_, false) => (arena.pi, true),
            };
            let constant = eval::eval(arena, constant);
            let g = if g_sign_negative {
                TSeries::neg(arena, &g)
            } else {
                g
            };
            let c = TSeries::constant(arena, constant, g.known.max(u.known));
            Ok(Some(TSeries::add(arena, &c, &g)))
        }
        InverseFn::Acosh => {
            let at_plus = if u0 == one {
                true
            } else if u0 == neg_one {
                false
            } else {
                return Ok(None);
            };
            // s = u ∓ 1.
            let c = TSeries::constant(arena, if at_plus { neg_one } else { one }, u.known);
            let s = TSeries::add(arena, &u, &c).normalized(arena);
            if at_plus {
                return sqrt2s_times_f(arena, &s, true, mode.side).map(Some);
            }
            if !s.coeffs.iter().all(|&co| is_known_real(arena, co)) {
                return Err(Obstruction::NoExpansion); // may cross the cut
            }
            let g = sqrt2s_times_f(arena, &s, false, mode.side)?;
            let minus_i_g = TSeries::scale(arena, &g, neg_i);
            let i_pi = {
                let p = arena.mul(&[i, arena.pi]);
                eval::eval(arena, p)
            };
            let c = TSeries::constant(arena, i_pi, minus_i_g.known.max(u.known));
            Ok(Some(TSeries::add(arena, &c, &minus_i_g)))
        }
        InverseFn::Atanh => {
            if u0 != one && u0 != neg_one {
                return Ok(None);
            }
            let c1 = TSeries::constant(arena, one, u.known);
            let one_plus = TSeries::add(arena, &c1, &u);
            let neg_u = TSeries::neg(arena, &u);
            let one_minus = TSeries::add(arena, &c1, &neg_u);
            // One of the two logarithms is singular: outside a log-extended
            // expansion there is no expansion (not a case for the
            // differentiation fallback).
            let definite = |e: Obstruction| match e {
                Obstruction::Unknown => Obstruction::NoExpansion,
                other => other,
            };
            let lp = apply_ln(arena, &one_plus, mode).map_err(definite)?;
            let lm = apply_ln(arena, &one_minus, mode).map_err(definite)?;
            let neg_lm = TSeries::neg(arena, &lm);
            let diff = TSeries::add(arena, &lp, &neg_lm);
            let half = rat_expr(arena, Q::new(BigInt::one(), BigInt::from(2)));
            Ok(Some(TSeries::scale(arena, &diff, half)))
        }
        InverseFn::Asinh | InverseFn::Atan => {
            if u0 != i && u0 != neg_i {
                return Ok(None);
            }
            // asinh u = −i·asin(iu), atan u = −i·atanh(iu).
            let iu = TSeries::scale(arena, &u, i);
            let inner = if kind == InverseFn::Asinh {
                InverseFn::Asin
            } else {
                InverseFn::Atanh
            };
            let Some(r) = inverse_at_branch_point(arena, inner, &iu, mode)? else {
                return Ok(None);
            };
            Ok(Some(TSeries::scale(arena, &r, neg_i)))
        }
    }
}

/// Is the constant `c` equal to `target` (structurally, or as a hidden
/// zero of `c − target`, see [`settle_constant`])?  A rational other than
/// `target` is not.
fn equals_constant(arena: &mut Arena, c: ExprId, target: ExprId) -> bool {
    if c == target {
        return true;
    }
    if arena.as_num(c).is_some() || !walk::free_symbols(arena, c).is_empty() {
        return false;
    }
    let d = arena.sub(c, target);
    is_zero_const(arena, d)
}

/// `√(2s)·F(±s/2)` for a series `s` of valuation ≥ 1, with
/// `F(z) = Σ C(2k, k)·zᵏ/(4ᵏ(2k + 1))` (see [`inverse_at_branch_point`]);
/// `negate` takes `F(−s/2)`.  The square root is [`pow_rational`]'s, so a
/// Puiseux exponent or an `|x|` is a definite [`Obstruction::NoExpansion`].
fn sqrt2s_times_f(
    arena: &mut Arena,
    s: &TSeries,
    negate: bool,
    side: Side,
) -> Result<TSeries, Obstruction> {
    let two = arena.int(2);
    let two_s = TSeries::scale(arena, s, two);
    let half_exp = arena.rational(1, 2);
    let root = pow_rational(arena, &two_s, half_exp, side)?;
    let factor = rat_expr(
        arena,
        Q::new(
            if negate {
                -BigInt::one()
            } else {
                BigInt::one()
            },
            BigInt::from(2),
        ),
    );
    let z = TSeries::scale(arena, s, factor);
    let f = TSeries::compose(
        arena,
        &|ar: &mut Arena, k: usize| {
            let k64 = k as u64;
            let num = binomial(2 * k64, k64);
            let den = (BigInt::one() << (2 * k)) * BigInt::from(2 * k64 + 1);
            rat_expr(ar, Q::new(num, den))
        },
        &z,
    );
    Ok(TSeries::mul(arena, &root, &f))
}

/// `ln(a)` — requires a non-zero constant term ([`Obstruction::Logarithmic`]
/// otherwise), except in a log-extended expansion ([`Mode::log_var`]),
/// where `a = c·var^v·(1 + …)` with `c` real of known sign and real visible
/// coefficients gives `ln c + v·ln(var) + ln(1 + …)` (for `var > 0` the
/// argument of `var^v` is 0, so no multiple of `2πi` is lost; for
/// `var < 0` it is `ln((−1)^v·c) + v·ln(−var) + …`, the sign moved into
/// the real constant).
///
/// A constant term on the cut (negative real) with a non-real correction is
/// a definite [`Obstruction::NoExpansion`]: `ln(−1 + i·x)` tends to `iπ`
/// from `x > 0` and to `−iπ` from `x < 0` (before, the series was
/// `iπ − i·x + …` on both sides).
fn apply_ln(arena: &mut Arena, a: &TSeries, mode: Mode) -> Result<TSeries, Obstruction> {
    let mut a = a.clone().normalized(arena);
    let v = a.leading_exponent(arena).ok_or(Obstruction::Unknown)?;
    let mut log_shift = None;
    if v != 0 {
        // logarithmic singularity
        let l = match mode.log_var {
            Some(l) if mode.side != Side::Both => l,
            _ => return Err(Obstruction::Logarithmic),
        };
        a.shift -= v;
        a.known -= v;
        // From below `var^v = (−1)^v·|var|^v` and `l = ln|var|`.
        if mode.side == Side::Below && v % 2 != 0 {
            a = TSeries::neg(arena, &a);
        }
        let c = a.coeff_at(arena, 0);
        match constant_sign(arena, c) {
            Some(true) => {}
            Some(false) if a.coeffs.iter().all(|&co| is_known_real(arena, co)) => {}
            _ => return Err(Obstruction::Unknown),
        }
        let vl = {
            let ve = arena.int(v);
            let p = arena.mul(&[ve, l]);
            eval::eval(arena, p)
        };
        log_shift = Some(vl);
    }
    let (u0, w) = a.split_constant(arena);
    // On the cut from below `ln` is `2πi` less than its principal value
    // there, which is the limit from above (SymPy's `log._eval_nseries`).
    // (A non-real `w` whose visible terms are all real may leave the cut
    // beyond the precision: undecided.)
    let below = if approaches_cut(arena, u0, &w) {
        match series_cut_side(arena, &w, false, mode) {
            Some(s) if s != 0 => s < 0,
            _ => return Err(Obstruction::NoExpansion),
        }
    } else {
        false
    };
    // ln(u0 + w) = ln u0 + ln(1 + w/u0)
    let inv_u0 = {
        let m1 = arena.neg_one;
        let p = arena.pow(u0, m1);
        eval::eval(arena, p)
    };
    let w_over = TSeries::scale(arena, &w, inv_u0);
    let mut s = TSeries::compose(
        arena,
        &|ar: &mut Arena, i: usize| FnKind::Ln1p.coefficient(ar, i),
        &w_over,
    );
    let mut ln_u0 = ln_of_constant(arena, u0);
    if let Some(vl) = log_shift {
        let sum = arena.add(&[ln_u0, vl]);
        ln_u0 = eval::eval(arena, sum);
    }
    if below {
        let two = arena.int(2);
        let m = arena.mul(&[two, arena.pi, arena.i_unit]);
        let sum = arena.sub(ln_u0, m);
        ln_u0 = eval::eval(arena, sum);
    }
    if !arena.is_zero_structural(ln_u0) && !s.coeffs.is_empty() && s.shift <= 0 {
        let idx = (-s.shift) as usize;
        let c = arena.add(&[s.coeffs[idx], ln_u0]);
        s.coeffs[idx] = eval::eval(arena, c);
    }
    Ok(s)
}

/// `ln c` of a constant, with `ln(p/q) = ln p − ln q` for a positive
/// rational, so that the constant cancels structurally against one spelled
/// with integers (the `ln 2` of K₀'s expansion against
/// `ln(x/2) = ln x + ln(1/2)`).
pub(crate) fn ln_of_constant(arena: &mut Arena, c: ExprId) -> ExprId {
    let l = match arena.as_num(c).cloned() {
        Some(r) if r.is_positive() && !r.is_integer() => {
            let p = rat_expr(arena, Q::from_integer(r.numer().clone()));
            let q = rat_expr(arena, Q::from_integer(r.denom().clone()));
            let lp = arena.ln(p);
            let lq = arena.ln(q);
            arena.sub(lp, lq)
        }
        _ => arena.ln(c),
    };
    eval::eval(arena, l)
}

/// Does `u0 + w` approach the negative real axis (the cut of `ln` and of
/// fractional powers) with a correction `w` that is not known to be real?
fn approaches_cut(arena: &mut Arena, u0: ExprId, w: &TSeries) -> bool {
    is_known_real(arena, u0)
        && constant_sign(arena, u0) == Some(false)
        && !w.coeffs.iter().all(|&c| is_known_real(arena, c))
}

/// `a^α` for a var-free non-integer exponent: `u0^α · Σ C(α,n) (w/u0)^n`.
///
/// A non-zero valuation `v` is accepted only when `vα` is an integer and the
/// result is single-valued: always for expansions from above, otherwise only
/// when `v / denom(α)` is even (so that no `|x|` appears).  A genuine
/// Puiseux series (`x^(5/2)`) or an `|x|` is a definite
/// [`Obstruction::NoExpansion`]: the differentiation fallback would see only
/// vanishing derivatives at low orders and return `0`.
fn pow_rational(
    arena: &mut Arena,
    a: &TSeries,
    alpha: ExprId,
    side: Side,
) -> Result<TSeries, Obstruction> {
    let mut a = a.clone().normalized(arena);
    // No visible term: the valuation is hidden at this precision; fail
    // definitively so that the caller widens the window.
    let v = a.leading_exponent(arena).ok_or(Obstruction::NoExpansion)?;
    let mut outer_shift = 0i64;
    if v != 0 {
        let ar = arena.as_num(alpha).cloned().ok_or(Obstruction::Unknown)?;
        let va = &ar * rat_i(v);
        if !va.is_integer() {
            return Err(Obstruction::NoExpansion); // genuine Puiseux series
        }
        if side != Side::Above && !ar.is_integer() {
            let q = ar.denom().to_i64().ok_or(Obstruction::Unknown)?;
            if (v / q) % 2 != 0 {
                return Err(Obstruction::NoExpansion); // would introduce |x|
            }
        }
        outer_shift = va.to_integer().to_i64().ok_or(Obstruction::Unknown)?;
        a.shift -= v;
        a.known -= v;
    }
    let (u0, w) = a.split_constant(arena);
    // From below the cut `u^α = e^{−2πiα}` times its principal value there.
    let below =
        if arena.as_num(alpha).is_none_or(|r| !r.is_integer()) && approaches_cut(arena, u0, &w) {
            let mode = Mode {
                side,
                log_var: None,
                limits: false,
                partial: false,
            };
            match series_cut_side(arena, &w, false, mode) {
                Some(s) if s != 0 => s < 0,
                _ => return Err(Obstruction::NoExpansion),
            }
        } else {
            false
        };
    let inv_u0 = {
        let m1 = arena.neg_one;
        let p = arena.pow(u0, m1);
        eval::eval(arena, p)
    };
    let w_over = TSeries::scale(arena, &w, inv_u0);
    let s = TSeries::compose(
        arena,
        &move |ar: &mut Arena, i: usize| gen_binomial_expr(ar, alpha, i),
        &w_over,
    );
    let u0a = arena.pow(u0, alpha);
    let u0a = if below {
        let two = arena.int(2);
        let p = arena.mul(&[two, arena.pi, arena.i_unit, alpha]);
        let p = arena.neg(p);
        let rot = arena.exp(p);
        arena.mul(&[u0a, rot])
    } else {
        u0a
    };
    let u0a = eval::eval(arena, u0a);
    let mut r = TSeries::scale(arena, &s, u0a);
    r.shift += outer_shift;
    r.known += outer_shift;
    Ok(r)
}

// ═══════════════════════════════════════════════════════════════════════════
// Special functions at their singular points
// ═══════════════════════════════════════════════════════════════════════════
//
// Where the argument of `Γ`, `ψ`, `ψ⁽ᵐ⁾`, `ζ`, `ln Γ`, `Ei`, `Ci`, `Chi`,
// `li`, `K₀`, `Y₀` tends to a pole, a logarithmic singularity or `+∞`, the
// differentiation fallback has nothing to evaluate; these rules give the
// expansion there in closed form (DLMF 5.7.1, 5.7.4, 5.11.1, 5.11.2, 5.15.1,
// 6.6.2, 6.6.5, 6.6.6, 6.2.8, 10.31.2, 10.8.2, 25.2.4; the recurrences
// Γ(z+1) = zΓ(z), ψ(z+1) = ψ(z) + 1/z).  Elsewhere they answer
// [`Obstruction::Unknown`] and the fallback applies as before.  Logarithmic
// singularities need a log-extended expansion ([`Mode::log_var`]); without
// one they are a definite [`Obstruction::NoExpansion`].

/// Most poles `Γ(−n)`, `ψ(−n)` unwound by the recurrence.
const MAX_POLE_SHIFT: i64 = 64;

/// Head of Gruntz's internal `erfcx(z) = e^{z²} erfc(z)` for `z → +∞`
/// (SymPy's `_erfs`), with the asymptotic series
/// `(1/√π) Σ (−1)ᵏ (2k−1)!!/(2ᵏ z^{2k+1})` (DLMF 7.12.1).  It exists only
/// inside a limit computation.
pub(crate) const ERFCX_ASYMPTOTIC: &str = "__erfcx_asym";

/// Head of Gruntz's internal `e^{−z} Ei(z)` for `z → ±∞` (SymPy's
/// `_eis`), with the asymptotic series `Σ k!/z^{k+1}` (DLMF 6.12.2).
pub(crate) const EI_ASYMPTOTIC: &str = "__ei_asym";

/// Most terms of an asymptotic (Stirling-type) series.
const MAX_ASYMPTOTIC_TERMS: i64 = 40;

/// The argument series `a` as `u0 + w` with `w` of valuation `≥ 1`, if `a`
/// has no pole.  `Err(NoExpansion)` when `a` has no visible term (a hidden
/// valuation: the caller widens the window).
fn split_regular(arena: &mut Arena, a: &TSeries) -> Result<Option<(ExprId, TSeries)>, Obstruction> {
    let a = a.clone().normalized(arena);
    if a.coeffs.is_empty() {
        return Err(Obstruction::NoExpansion);
    }
    if a.shift < 0 {
        return Ok(None);
    }
    let (u0, w) = a.split_constant(arena);
    Ok(Some((settle_constant(arena, u0), w)))
}

/// `n` if the constant `u0` is the integer `−n ≤ 0`.
fn nonpositive_integer(arena: &Arena, u0: ExprId) -> Option<i64> {
    if arena.is_zero_structural(u0) {
        return Some(0);
    }
    let r = arena.as_num(u0)?;
    if r.is_integer() && !r.is_positive() {
        r.to_integer().to_i64().map(|n| -n)
    } else {
        None
    }
}

fn is_one_const(arena: &Arena, u0: ExprId) -> bool {
    arena.as_num(u0).is_some_and(|r| r.is_one())
}

/// `ζ(k)` for an integer `k ≥ 2` (`π²/6`, `ζ(3)`, …).
fn zeta_int(arena: &mut Arena, k: usize) -> ExprId {
    let ke = arena.int(k as i64);
    let z = arena.zeta(ke);
    eval::eval(arena, z)
}

/// `c·s` for a rational `c`.
fn scale_rat(arena: &mut Arena, s: &TSeries, c: Q) -> TSeries {
    let ce = rat_expr(arena, c);
    TSeries::scale(arena, s, ce)
}

/// The constant series `c` with the precision of `like`.
fn constant_like(arena: &Arena, c: ExprId, like: &TSeries) -> TSeries {
    TSeries::constant(arena, c, like.known)
}

/// `(w + s)^(−power)` for an integer shift `s`.
fn recip_shifted(arena: &mut Arena, w: &TSeries, s: i64, power: i64) -> Option<TSeries> {
    let se = arena.int(s);
    let c = constant_like(arena, se, w);
    let base = TSeries::add(arena, w, &c);
    TSeries::pow_int(arena, &base, -power)
}

/// `ln Γ(1 + w) = −γw + Σ_{k≥2} (−1)ᵏ ζ(k) wᵏ/k` (DLMF 5.7.3).
fn ln_gamma_one_plus(arena: &mut Arena, w: &TSeries) -> TSeries {
    TSeries::compose(
        arena,
        &|ar: &mut Arena, k: usize| match k {
            0 => ar.zero,
            1 => {
                let g = ar.euler_gamma;
                ar.neg(g)
            }
            _ => {
                let z = zeta_int(ar, k);
                let sign = if k.is_multiple_of(2) { 1 } else { -1 };
                let c = rat_expr(ar, Q::new(BigInt::from(sign), BigInt::from(k)));
                let p = ar.mul(&[c, z]);
                eval::eval(ar, p)
            }
        },
        w,
    )
}

/// `Γ(1 + w)` from the Taylor coefficients `gₙ` of `exp(ln Γ(1 + w))`,
/// `g₀ = 1`, `n·gₙ = Σ_{k=1}^{n} k·c_k·g_{n−k}` with `c_k` those of
/// [`ln_gamma_one_plus`].  The coefficients are built once as closed forms
/// (polynomials in `γ`, `ζ(k)`; none vanishes, `gₙ ≈ (−1)ⁿ`), so the
/// series arithmetic of `exp` — with a numerical zero test of every sum of
/// `ζ` values — is avoided.
fn gamma_one_plus(arena: &mut Arena, w: &TSeries) -> TSeries {
    // `TSeries::compose` asks for coefficients up to `w.known + 1`.
    let n_max = w.known.max(1) as usize + 2;
    let mut c: Vec<ExprId> = Vec::with_capacity(n_max + 1);
    c.push(arena.zero);
    for k in 1..=n_max {
        c.push(if k == 1 {
            let g = arena.euler_gamma;
            arena.neg(g)
        } else {
            let z = zeta_int(arena, k);
            let sign = if k.is_multiple_of(2) { 1 } else { -1 };
            let q = rat_expr(arena, Q::new(BigInt::from(sign), BigInt::from(k)));
            let p = arena.mul(&[q, z]);
            eval::eval(arena, p)
        });
    }
    let mut g: Vec<ExprId> = Vec::with_capacity(n_max + 1);
    g.push(arena.one);
    for n in 1..=n_max {
        let mut terms = Vec::with_capacity(n);
        for k in 1..=n {
            let kq = rat_expr(arena, Q::new(BigInt::from(k), BigInt::from(n)));
            terms.push(arena.mul(&[kq, c[k], g[n - k]]));
        }
        let s = arena.add(&terms);
        let s = crate::transforms::expand::expand(arena, s);
        g.push(eval::eval(arena, s));
    }
    TSeries::compose(
        arena,
        &move |ar: &mut Arena, n: usize| g.get(n).copied().unwrap_or(ar.zero),
        w,
    )
}

/// `ψ⁽ᵐ⁾(1 + w) = Σ_j (−1)^{m+j+1} (m+j)!/j! ζ(m+j+1) wʲ` (`ψ(1) = −γ`),
/// the derivatives of [`ln_gamma_one_plus`].
fn polygamma_one_plus(arena: &mut Arena, m: usize, w: &TSeries) -> TSeries {
    TSeries::compose(
        arena,
        &move |ar: &mut Arena, j: usize| {
            if m == 0 && j == 0 {
                let g = ar.euler_gamma;
                return ar.neg(g);
            }
            let z = zeta_int(ar, m + j + 1);
            let mut c =
                Q::from_integer(factorial((m + j) as u64)) / Q::from_integer(factorial(j as u64));
            if (m + j + 1) % 2 == 1 {
                c = -c;
            }
            let ce = rat_expr(ar, c);
            let p = ar.mul(&[ce, z]);
            eval::eval(ar, p)
        },
        w,
    )
}

/// For an argument `a → +∞` in a log-extended expansion from above (a pole
/// with a positive leading coefficient): `(ln a, 1/a, p)` with `p` the
/// valuation of `1/a`.
fn at_plus_infinity(arena: &mut Arena, a: &TSeries, mode: Mode) -> Option<(TSeries, TSeries, i64)> {
    mode.log_var?;
    if mode.side != Side::Above {
        return None;
    }
    let a = a.clone().normalized(arena);
    let v = a.leading_exponent(arena)?;
    if v >= 0 {
        return None;
    }
    let c = a.coeff_at(arena, v);
    if constant_sign(arena, c) != Some(true) {
        return None;
    }
    let ln_a = apply_ln(arena, &a, mode).ok()?;
    let inv = TSeries::inverse(arena, &a)?;
    Some((ln_a, inv, -v))
}

/// For an argument `a → +∞` in an expansion from above (a pole with a
/// positive leading coefficient): `(1/a, p)` with `p` the valuation of
/// `1/a` (no logarithm needed).
fn pole_to_plus_infinity(arena: &mut Arena, a: &TSeries, mode: Mode) -> Option<(TSeries, i64)> {
    if mode.side != Side::Above {
        return None;
    }
    let a = a.clone().normalized(arena);
    let v = a.leading_exponent(arena)?;
    if v >= 0 || constant_sign(arena, a.coeff_at(arena, v)) != Some(true) {
        return None;
    }
    let inv = TSeries::inverse(arena, &a)?;
    Some((inv, -v))
}

/// Number of terms `K` of an asymptotic series in `1/a` (valuation `p`) so
/// that its remainder lies beyond the working precision `target`.
fn asymptotic_terms(target: i64, p: i64) -> i64 {
    (target.max(1) / (2 * p) + 2).min(MAX_ASYMPTOTIC_TERMS)
}

/// `Σ_{k=1}^{K} c_k·inv^(2k+offset)` with `c_k = coeff(k)`.
fn odd_even_sum(
    arena: &mut Arena,
    inv: &TSeries,
    k_max: i64,
    offset: i64,
    coeff: &dyn Fn(usize) -> Q,
) -> Option<TSeries> {
    let inv2 = TSeries::mul(arena, inv, inv);
    let mut power = TSeries::pow_int(arena, inv, 2 + offset)?;
    let mut acc: Option<TSeries> = None;
    for k in 1..=k_max {
        let c = coeff(k as usize);
        if !c.is_zero() {
            let term = scale_rat(arena, &power, c);
            acc = Some(match acc {
                None => term,
                Some(s) => TSeries::add(arena, &s, &term),
            });
        }
        power = TSeries::mul(arena, &power, &inv2);
    }
    acc
}

/// Expansion of a special function whose argument tends to one of its
/// singular points (see the section comment).
fn special_series(
    arena: &mut Arena,
    node: &ExprNode,
    mode: Mode,
    cache: &FxHashMap<ExprId, Option<TSeries>>,
) -> Result<TSeries, Obstruction> {
    use crate::base::libfn::LibFn;
    let unknown = Err(Obstruction::Unknown);
    match *node {
        ExprNode::Erfc(a) => {
            let e = apply_fn(arena, FnKind::Erf, &child(cache, a)?).ok_or(Obstruction::Unknown)?;
            let one = constant_like(arena, arena.one, &e);
            let ne = TSeries::neg(arena, &e);
            Ok(TSeries::add(arena, &one, &ne))
        }
        ExprNode::Gamma(a) | ExprNode::Digamma(a) => {
            let a = child(cache, a)?;
            if matches!(node, ExprNode::Digamma(_))
                && let Some((ln_a, inv, p)) = at_plus_infinity(arena, &a, mode)
            {
                // ψ(a) = ln a − 1/(2a) − Σ B₂ₖ/(2k a^{2k}) (DLMF 5.11.2)
                let k_max = asymptotic_terms(a.known - a.shift, p);
                let half = scale_rat(arena, &inv, Q::new(BigInt::from(-1), BigInt::from(2)));
                let mut acc = TSeries::add(arena, &ln_a, &half);
                let tail = odd_even_sum(arena, &inv, k_max, 0, &|k| {
                    -bernoulli(2 * k) / rat_i(2 * k as i64)
                })
                .ok_or(Obstruction::Unknown)?;
                acc = TSeries::add(arena, &acc, &tail);
                return Ok(acc.truncate_known(p * (2 * k_max + 2)));
            }
            let Some((u0, w)) = split_regular(arena, &a)? else {
                return unknown;
            };
            let Some(n) = nonpositive_integer(arena, u0) else {
                return unknown;
            };
            if n > MAX_POLE_SHIFT || w.coeffs.is_empty() {
                return Err(Obstruction::NoExpansion);
            }
            if matches!(node, ExprNode::Gamma(_)) {
                // Γ(−n + w) = Γ(1 + w) / Π_{k=0}^{n} (w − n + k)
                let mut acc = gamma_one_plus(arena, &w);
                for k in 0..=n {
                    let r = recip_shifted(arena, &w, k - n, 1).ok_or(Obstruction::NoExpansion)?;
                    acc = TSeries::mul(arena, &acc, &r);
                }
                Ok(acc)
            } else {
                // ψ(−n + w) = ψ(1 + w) − Σ_{k=0}^{n} 1/(w − n + k)
                let mut acc = polygamma_one_plus(arena, 0, &w);
                for k in 0..=n {
                    let r = recip_shifted(arena, &w, k - n, 1).ok_or(Obstruction::NoExpansion)?;
                    let nr = TSeries::neg(arena, &r);
                    acc = TSeries::add(arena, &acc, &nr);
                }
                Ok(acc)
            }
        }
        ExprNode::Polygamma(m, a) => {
            let Some(m) = arena
                .as_num(m)
                .filter(|r| r.is_integer() && r.is_positive())
                .and_then(|r| r.to_integer().to_usize())
                .filter(|&m| m <= MAX_INT_POWER as usize)
            else {
                return unknown;
            };
            let a = child(cache, a)?;
            if let Some((inv, p)) = pole_to_plus_infinity(arena, &a, mode) {
                // ψ⁽ᵐ⁾(a) = (−1)^{m+1} [(m−1)!/a^m + m!/(2a^{m+1})
                //   + Σ B₂ₖ (2k+m−1)!/((2k)! a^{2k+m})]  (DLMF 5.15.8)
                let k_max = asymptotic_terms(a.known - a.shift, p);
                let mi = m as i64;
                let sign = if m % 2 == 1 { Q::one() } else { -Q::one() };
                let lead = TSeries::pow_int(arena, &inv, mi).ok_or(Obstruction::Unknown)?;
                let lead = scale_rat(
                    arena,
                    &lead,
                    &sign * Q::from_integer(factorial(m as u64 - 1)),
                );
                let second = TSeries::pow_int(arena, &inv, mi + 1).ok_or(Obstruction::Unknown)?;
                let second = scale_rat(
                    arena,
                    &second,
                    &sign * Q::from_integer(factorial(m as u64)) / rat_i(2),
                );
                let mut acc = TSeries::add(arena, &lead, &second);
                let tail = odd_even_sum(arena, &inv, k_max, mi, &|k| {
                    &sign * bernoulli(2 * k) * Q::from_integer(factorial((2 * k + m - 1) as u64))
                        / Q::from_integer(factorial(2 * k as u64))
                })
                .ok_or(Obstruction::Unknown)?;
                acc = TSeries::add(arena, &acc, &tail);
                return Ok(acc.truncate_known(p * (2 * k_max + 2 + mi)));
            }
            let Some((u0, w)) = split_regular(arena, &a)? else {
                return unknown;
            };
            let Some(n) = nonpositive_integer(arena, u0) else {
                return unknown;
            };
            if n > MAX_POLE_SHIFT || w.coeffs.is_empty() {
                return Err(Obstruction::NoExpansion);
            }
            // ψ⁽ᵐ⁾(−n + w) = ψ⁽ᵐ⁾(1 + w) − (−1)ᵐ m! Σ_{k=0}^{n} (w − n + k)^{−m−1}
            let mut acc = polygamma_one_plus(arena, m, &w);
            let mut c = Q::from_integer(factorial(m as u64));
            if m.is_multiple_of(2) {
                c = -c;
            }
            for k in 0..=n {
                let r = recip_shifted(arena, &w, k - n, m as i64 + 1)
                    .ok_or(Obstruction::NoExpansion)?;
                let t = scale_rat(arena, &r, c.clone());
                acc = TSeries::add(arena, &acc, &t);
            }
            Ok(acc)
        }
        ExprNode::Zeta(a) => {
            let a = child(cache, a)?;
            let Some((u0, w)) = split_regular(arena, &a)? else {
                return unknown;
            };
            if !is_one_const(arena, u0) {
                return unknown;
            }
            // ζ(1 + w) = 1/w + γ + O(w): the higher coefficients are
            // Stieltjes constants, which have no representation here, so
            // the expansion is exact only below the valuation of w.
            let v = w.leading_exponent(arena).ok_or(Obstruction::NoExpansion)?;
            let inv = TSeries::inverse(arena, &w).ok_or(Obstruction::NoExpansion)?;
            let g = TSeries::constant(arena, arena.euler_gamma, v);
            Ok(TSeries::add(arena, &inv, &g).truncate_known(v))
        }
        ExprNode::LogGamma(a) => {
            let a = child(cache, a)?;
            if let Some((ln_a, inv, p)) = at_plus_infinity(arena, &a, mode) {
                // Stirling: (a − ½) ln a − a + ½ ln 2π + Σ B₂ₖ/(2k(2k−1) a^{2k−1})
                let k_max = asymptotic_terms(a.known - a.shift, p);
                let half = rat_expr(arena, Q::new(BigInt::from(-1), BigInt::from(2)));
                let hc = constant_like(arena, half, &a);
                let a_half = TSeries::add(arena, &a, &hc);
                let main = TSeries::mul(arena, &a_half, &ln_a);
                let na = TSeries::neg(arena, &a);
                // ½ ln 2π, spelled ½ ln 2 + ½ ln π (SymPy's form in limits),
                // so that its exponential simplifies to √2·√π.
                let two_pi = {
                    let two = arena.int(2);
                    let pi = arena.pi;
                    let l2 = arena.ln(two);
                    let lp = arena.ln(pi);
                    let s = arena.add(&[l2, lp]);
                    let h = rat_expr(arena, Q::new(BigInt::one(), BigInt::from(2)));
                    let p = arena.mul(&[h, s]);
                    let p = crate::transforms::expand::expand(arena, p);
                    eval::eval(arena, p)
                };
                let c = constant_like(arena, two_pi, &main);
                let mut acc = TSeries::add(arena, &main, &na);
                acc = TSeries::add(arena, &acc, &c);
                let tail = odd_even_sum(arena, &inv, k_max, -1, &|k| {
                    bernoulli(2 * k) / rat_i((2 * k * (2 * k - 1)) as i64)
                })
                .ok_or(Obstruction::Unknown)?;
                acc = TSeries::add(arena, &acc, &tail);
                return Ok(acc.truncate_known(p * (2 * k_max + 1)));
            }
            let Some((u0, _)) = split_regular(arena, &a)? else {
                return unknown;
            };
            if !arena.is_zero_structural(u0) {
                return unknown;
            }
            // ln Γ(a) = ln Γ(1 + a) − ln a for a → 0⁺.
            let a_n = a.clone().normalized(arena);
            let lead = a_n
                .leading_exponent(arena)
                .ok_or(Obstruction::NoExpansion)?;
            let c = a_n.coeff_at(arena, lead);
            if mode.log_var.is_none() || constant_sign(arena, c) != Some(true) {
                return Err(Obstruction::NoExpansion);
            }
            let ln_a = apply_ln(arena, &a_n, mode).map_err(|_| Obstruction::NoExpansion)?;
            let lg = ln_gamma_one_plus(arena, &a_n);
            let nl = TSeries::neg(arena, &ln_a);
            Ok(TSeries::add(arena, &lg, &nl))
        }
        ExprNode::Si(a) | ExprNode::Ei(a) | ExprNode::Ci(a) => {
            let a = child(cache, a)?;
            let Some((u0, _)) = split_regular(arena, &a)? else {
                return unknown;
            };
            if !arena.is_zero_structural(u0) {
                return unknown;
            }
            let kind = match node {
                ExprNode::Si(_) => IntegralKind::Si,
                ExprNode::Ei(_) => IntegralKind::Ei,
                _ => IntegralKind::Ci,
            };
            integral_at_zero(arena, kind, &a, mode)
        }
        ExprNode::Apply(f, ref args)
            if PowerAsymptotic::from_name(arena.symbol_name(f)).is_some() =>
        {
            let Some(kind) = PowerAsymptotic::from_name(arena.symbol_name(f)) else {
                return unknown;
            };
            power_asymptotic_series(arena, f, kind, args, mode, cache)
        }
        ExprNode::Apply(f, ref args)
            if args.len() == 1
                && matches!(arena.symbol_name(f), ERFCX_ASYMPTOTIC | EI_ASYMPTOTIC) =>
        {
            let erfcx = arena.symbol_name(f) == ERFCX_ASYMPTOTIC;
            let a = child(cache, args[0])?.normalized(arena);
            let v = a.leading_exponent(arena).ok_or(Obstruction::NoExpansion)?;
            if v == 0
                && a.coeffs
                    .iter()
                    .skip(1)
                    .all(|&c| arena.is_zero_structural(c))
            {
                // A constant argument `c + O(ωᴷ)` (in a log-extended
                // expansion `ln(1/ω)` is the constant `−ln ω`): the function
                // is `f(c) + O(ωᴷ)`, its derivative being bounded away from
                // the logarithmic singularity of `e^{−z} Ei z` at `0`.
                let c0 = a.coeffs[0];
                if erfcx || !is_zero_const(arena, c0) {
                    let c = arena.intern(ExprNode::Apply(f, smallvec::smallvec![c0]));
                    return Ok(TSeries::constant(arena, c, a.known));
                }
                return unknown;
            }
            if v >= 0 || mode.side != Side::Above {
                return unknown;
            }
            let positive = constant_sign(arena, a.coeff_at(arena, v));
            if positive != Some(true) && (erfcx || positive.is_none()) {
                return unknown;
            }
            let inv = TSeries::inverse(arena, &a).ok_or(Obstruction::NoExpansion)?;
            let p = -v;
            let k_max = if erfcx {
                asymptotic_terms(a.known - a.shift, p)
            } else {
                ((a.known - a.shift).max(1) / p + 2).min(MAX_ASYMPTOTIC_TERMS)
            };
            let mut acc: Option<TSeries> = None;
            let mut power = inv.clone();
            let step = if erfcx {
                TSeries::mul(arena, &inv, &inv)
            } else {
                inv.clone()
            };
            let mut c = Q::one();
            for k in 0..=k_max {
                if k > 0 {
                    c = if erfcx {
                        -c * rat_i(2 * k - 1) / rat_i(2)
                    } else {
                        c * rat_i(k)
                    };
                }
                let term = scale_rat(arena, &power, c.clone());
                acc = Some(match acc {
                    None => term,
                    Some(s) => TSeries::add(arena, &s, &term),
                });
                power = TSeries::mul(arena, &power, &step);
            }
            let mut s = acc.ok_or(Obstruction::Unknown)?;
            // Remainder: the first omitted term.
            let omitted = if erfcx { 2 * k_max + 3 } else { k_max + 2 };
            s = s.truncate_known(p * omitted);
            if erfcx {
                let rsp = {
                    let pi = arena.pi;
                    let sp = arena.sqrt(pi);
                    let one = arena.one;
                    let q = arena.div(one, sp);
                    eval::eval(arena, q)
                };
                s = TSeries::scale(arena, &s, rsp);
            }
            Ok(s)
        }
        ExprNode::Apply(f, ref args) => {
            let kind = match arena.lib_fn(f) {
                Some(LibFn::Shi) => IntegralKind::Shi,
                Some(LibFn::Chi) => IntegralKind::Chi,
                _ => return unknown,
            };
            let a = child(cache, args[0])?;
            let Some((u0, _)) = split_regular(arena, &a)? else {
                return unknown;
            };
            if !arena.is_zero_structural(u0) {
                return unknown;
            }
            integral_at_zero(arena, kind, &a, mode)
        }
        ExprNode::Li(a) => {
            let a = child(cache, a)?;
            let Some((u0, _)) = split_regular(arena, &a)? else {
                return unknown;
            };
            if !is_one_const(arena, u0) {
                return unknown;
            }
            // li(a) = Ei(ln a), ln a → 0.
            let l = apply_ln(arena, &a, mode)?;
            integral_at_zero(arena, IntegralKind::Ei, &l, mode)
        }
        _ => unknown,
    }
}

/// The internal functions of Gruntz's asymptotic rewrite whose expansion at
/// `z → +∞` is a power series in `1/z` with a factor of `e^{±z}`, `e^{z²}`
/// or `2^{−z}` taken out (see `gruntz::asymptotic_rewrite`).  Each exists
/// only inside a limit computation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PowerAsymptotic {
    /// `√v·e^{−v} erfi(√v) ~ (1/√π) Σ (2k−1)!!/(2v)ᵏ` (DLMF 7.12.1 at `iz`;
    /// SymPy's `erfi._eval_aseries`), a function of `v = z²` so that
    /// `erfi(√x)` has a series in `1/x`.
    Erfi,
    /// `√v·e^{v} erfc(√v) ~ (1/√π) Σ (−1)ᵏ (2k−1)!!/(2v)ᵏ` (DLMF 7.12.1):
    /// [`ERFCX_ASYMPTOTIC`] of `z = √v` times `z`, whose series has no
    /// half-integer powers when `z²` has none (`erfc(√x)`, and `Γ(1/2, x) =
    /// √π·erfc(√x)`).
    Erfc,
    /// `√(2πz) e^{−z} Iν(z) ~ Σ (−1)ᵏ aₖ(ν)/zᵏ`,
    /// `aₖ(ν) = ∏_{j≤k} (4ν² − (2j−1)²)/(8j)` (DLMF 10.40.1).
    BesselI,
    /// `√(2z/π) e^{z} Kν(z) ~ Σ aₖ(ν)/zᵏ` (DLMF 10.40.2).
    BesselK,
    /// `z^{1−s} e^{z} Γ(s, z) ~ Σ (s−1)(s−2)⋯(s−k)/zᵏ` (DLMF 8.11.2; with
    /// `Eₙ(z) = z^{n−1}Γ(1 − n, z)`, 8.19.1).
    UpperGamma,
    /// `2^z (ζ(z) − 1) = 1 + Σ_{n≥3} (2/n)^z`, which is `1` to every order in
    /// `1/z` (DLMF 25.2.1).
    Zeta,
}

/// Head of [`PowerAsymptotic::Erfi`].
pub(crate) const ERFIX_ASYMPTOTIC: &str = "__erfix2_asym";
/// Head of [`PowerAsymptotic::Erfc`].
pub(crate) const ERFCX2_ASYMPTOTIC: &str = "__erfcx2_asym";
/// Head of [`PowerAsymptotic::BesselI`] (arguments `ν, z`).
pub(crate) const BESSELI_ASYMPTOTIC: &str = "__besseli_asym";
/// Head of [`PowerAsymptotic::BesselK`] (arguments `ν, z`).
pub(crate) const BESSELK_ASYMPTOTIC: &str = "__besselk_asym";
/// Head of [`PowerAsymptotic::UpperGamma`] (arguments `s, z`).
pub(crate) const UPPERGAMMA_ASYMPTOTIC: &str = "__uppergamma_asym";
/// Head of [`PowerAsymptotic::Zeta`].
pub(crate) const ZETA_ASYMPTOTIC: &str = "__zeta_asym";

impl PowerAsymptotic {
    pub(crate) fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            ERFIX_ASYMPTOTIC => Self::Erfi,
            ERFCX2_ASYMPTOTIC => Self::Erfc,
            BESSELI_ASYMPTOTIC => Self::BesselI,
            BESSELK_ASYMPTOTIC => Self::BesselK,
            UPPERGAMMA_ASYMPTOTIC => Self::UpperGamma,
            ZETA_ASYMPTOTIC => Self::Zeta,
            _ => return None,
        })
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Erfi => ERFIX_ASYMPTOTIC,
            Self::Erfc => ERFCX2_ASYMPTOTIC,
            Self::BesselI => BESSELI_ASYMPTOTIC,
            Self::BesselK => BESSELK_ASYMPTOTIC,
            Self::UpperGamma => UPPERGAMMA_ASYMPTOTIC,
            Self::Zeta => ZETA_ASYMPTOTIC,
        }
    }

    /// Does the head take a parameter (`ν`, `s`) before the argument?
    pub(crate) fn has_parameter(self) -> bool {
        matches!(self, Self::BesselI | Self::BesselK | Self::UpperGamma)
    }
}

/// The expansion of a [`PowerAsymptotic`] function of `a → +∞` (a pole with
/// a positive leading coefficient, from above): `Σ_{k≤K} cₖ·a^{−(step·k +
/// offset)}`, exact below its first omitted term; of a constant argument
/// `c + O(ωᴷ)`, the constant `f(c)`.  The parameter must be constant.
fn power_asymptotic_series(
    arena: &mut Arena,
    f: crate::base::node::SymbolId,
    kind: PowerAsymptotic,
    args: &[ExprId],
    mode: Mode,
    cache: &FxHashMap<ExprId, Option<TSeries>>,
) -> Result<TSeries, Obstruction> {
    let unknown = Err(Obstruction::Unknown);
    let (param, z) = match (kind.has_parameter(), args) {
        (false, &[z]) => (None, z),
        (true, &[p, z]) => (Some(p), z),
        _ => return unknown,
    };
    let is_constant = |arena: &Arena, s: &TSeries| {
        s.coeffs
            .iter()
            .skip(1)
            .all(|&c| arena.is_zero_structural(c))
    };
    // Only a constant parameter (a series `c + 0·ω + …`, or a literal 0).
    let param = match param {
        Some(p) if arena.is_zero_structural(p) => Some(arena.zero),
        Some(p) => {
            let ps = child(cache, p)?.normalized(arena);
            if ps.coeffs.is_empty() || ps.shift != 0 || !is_constant(arena, &ps) {
                return unknown;
            }
            Some(ps.coeffs[0])
        }
        None => None,
    };
    let a = child(cache, z)?.normalized(arena);
    let v = a.leading_exponent(arena).ok_or(Obstruction::NoExpansion)?;
    if v == 0 && is_constant(arena, &a) {
        let mut new_args: smallvec::SmallVec<[ExprId; 2]> = smallvec::SmallVec::new();
        new_args.extend(param);
        new_args.push(a.coeffs[0]);
        let c = arena.intern(ExprNode::Apply(f, new_args));
        return Ok(TSeries::constant(arena, c, a.known));
    }
    if v >= 0
        || mode.side != Side::Above
        || constant_sign(arena, a.coeff_at(arena, v)) != Some(true)
    {
        return unknown;
    }
    let inv = TSeries::inverse(arena, &a).ok_or(Obstruction::NoExpansion)?;
    let p = -v;
    let target = (a.known - a.shift).max(1);
    let k_max = (target / p + 2).min(MAX_ASYMPTOTIC_TERMS);
    let one = arena.one;
    if kind == PowerAsymptotic::Zeta {
        // 1 to every order: the corrections `(2/n)^z` are exponentially small.
        return Ok(TSeries::constant(arena, one, p * (k_max + 1)));
    }
    let nu2 = param.map(|nu| {
        let four = arena.int(4);
        let two = arena.int(2);
        let sq = arena.pow(nu, two);
        let t = arena.mul(&[four, sq]);
        eval::eval(arena, t)
    });
    // `Σ cₖ·a⁻ᵏ`, `c₀ = 1`, `cₖ = cₖ₋₁·(ratio of consecutive terms)`.
    let mut coeff = one;
    let mut power = TSeries::constant(arena, one, inv.known);
    let mut acc: Option<TSeries> = None;
    for k in 0..=k_max {
        if k > 0 {
            let factor = match kind {
                PowerAsymptotic::Erfi => {
                    rat_expr(arena, Q::new(BigInt::from(2 * k - 1), BigInt::from(2)))
                }
                PowerAsymptotic::Erfc => {
                    rat_expr(arena, Q::new(BigInt::from(1 - 2 * k), BigInt::from(2)))
                }
                PowerAsymptotic::BesselI | PowerAsymptotic::BesselK => {
                    let odd = (2 * k - 1) * (2 * k - 1);
                    let odd = arena.int(odd);
                    let nu2 = nu2.unwrap_or(arena.zero);
                    let d = arena.sub(nu2, odd);
                    let s = if kind == PowerAsymptotic::BesselI {
                        -1
                    } else {
                        1
                    };
                    let q = rat_expr(arena, Q::new(BigInt::from(s), BigInt::from(8 * k)));
                    arena.mul(&[q, d])
                }
                PowerAsymptotic::UpperGamma => {
                    let s = param.unwrap_or(arena.zero);
                    let ke = arena.int(k);
                    arena.sub(s, ke)
                }
                PowerAsymptotic::Zeta => arena.zero,
            };
            let c = arena.mul(&[coeff, factor]);
            let c = crate::transforms::expand::expand(arena, c);
            coeff = eval::eval(arena, c);
            power = TSeries::mul(arena, &power, &inv);
        }
        if arena.is_zero_structural(coeff) {
            // A terminating series (`K_{1/2}`, `Γ(n, z)`): exact.
            break;
        }
        let term = TSeries::scale(arena, &power, coeff);
        acc = Some(match acc {
            None => term,
            Some(s) => TSeries::add(arena, &s, &term),
        });
    }
    let mut s = acc.unwrap_or_else(|| TSeries::zero(arena, a.known));
    if !arena.is_zero_structural(coeff) {
        // The first omitted term is of order `a^{−(K+1)}`.
        s = s.truncate_known(p * (k_max + 1));
    }
    if matches!(kind, PowerAsymptotic::Erfi | PowerAsymptotic::Erfc) {
        let rsp = {
            let pi = arena.pi;
            let sp = arena.sqrt(pi);
            let q = arena.div(one, sp);
            eval::eval(arena, q)
        };
        s = TSeries::scale(arena, &s, rsp);
    }
    Ok(s)
}

/// The integral functions expanded at a zero of their argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IntegralKind {
    Si,
    Shi,
    Ei,
    Ci,
    Chi,
}

/// `Si`, `Shi` (entire), `Ei(a) = γ + ln|a| + Σ aᵏ/(k·k!)`,
/// `Ci(a) = γ + ln a + Σ (−1)ᵐ a²ᵐ/(2m·(2m)!)`, `Chi` likewise without the
/// signs, for `a → 0` (DLMF 6.6.2, 6.6.5, 6.6.6; SymPy's `Ci(−x) = Ci(x) + iπ`
/// is the principal `ln` of a negative `a`, and `Ei` of a real argument is
/// real).  The logarithmic ones need a log-extended expansion.
fn integral_at_zero(
    arena: &mut Arena,
    kind: IntegralKind,
    a: &TSeries,
    mode: Mode,
) -> Result<TSeries, Obstruction> {
    let a = a.clone().normalized(arena);
    if a.coeffs.is_empty() {
        return Err(Obstruction::NoExpansion);
    }
    let coefficient = move |ar: &mut Arena, k: usize| -> ExprId {
        let kq = rat_i(k as i64);
        let fact = Q::from_integer(factorial(k as u64));
        let odd = k % 2 == 1;
        let m = k / 2;
        let alt = if m.is_multiple_of(2) {
            Q::one()
        } else {
            -Q::one()
        };
        let c = match kind {
            IntegralKind::Si if odd => alt / (kq * fact),
            IntegralKind::Shi if odd => Q::one() / (kq * fact),
            IntegralKind::Ei if k > 0 => Q::one() / (kq * fact),
            IntegralKind::Ci if k > 0 && !odd => alt / (kq * fact),
            IntegralKind::Chi if k > 0 && !odd => Q::one() / (kq * fact),
            IntegralKind::Ei | IntegralKind::Ci | IntegralKind::Chi if k == 0 => {
                return ar.euler_gamma;
            }
            _ => Q::zero(),
        };
        rat_expr(ar, c)
    };
    let series = TSeries::compose(arena, &coefficient, &a);
    if matches!(kind, IntegralKind::Si | IntegralKind::Shi) {
        return Ok(series);
    }
    if mode.log_var.is_none() {
        return Err(Obstruction::NoExpansion);
    }
    let log_arg = if kind == IntegralKind::Ei {
        abs_series(arena, &a, mode.side)?
    } else {
        a
    };
    let ln = apply_ln(arena, &log_arg, mode).map_err(|_| Obstruction::NoExpansion)?;
    Ok(TSeries::add(arena, &series, &ln))
}

/// `K₀(u)` and `Y₀(u)` at a zero of `u` in a log-extended expansion (DLMF
/// 10.31.2, 10.8.2):
/// `K₀(z) = −(ln(z/2) + γ) I₀(z) + Σ_{k≥1} H_k (z/2)^{2k}/(k!)²`,
/// `Y₀(z) = (2/π)[(ln(z/2) + γ) J₀(z) + Σ_{k≥1} (−1)^{k+1} H_k (z/2)^{2k}/(k!)²]`,
/// for `u → 0⁺` (positive leading coefficient).  `Ok(None)` when the rule
/// does not apply.
fn log_bessel_series(
    arena: &mut Arena,
    f: crate::base::node::SymbolId,
    order: ExprId,
    u: &TSeries,
    var: ExprId,
    mode: Mode,
) -> Result<Option<TSeries>, Obstruction> {
    use crate::base::libfn::LibFn;
    let lib = arena.lib_fn(f);
    let is_k = match lib {
        Some(LibFn::BesselK) => true,
        Some(LibFn::BesselY) => false,
        _ => return Ok(None),
    };
    if mode.log_var.is_none()
        || walk::contains(arena, order, var)
        || !arena.is_zero_structural(order)
    {
        return Ok(None);
    }
    let u = u.clone().normalized(arena);
    let Some(k) = u.leading_exponent(arena) else {
        return Err(Obstruction::NoExpansion);
    };
    if k <= 0 {
        return Ok(None);
    }
    if constant_sign(arena, u.coeff_at(arena, k)) != Some(true) {
        return Err(Obstruction::NoExpansion);
    }
    let alternate = !is_k;
    let bessel0 = move |ar: &mut Arena, m: usize| -> ExprId {
        if m % 2 == 1 {
            return ar.zero;
        }
        let kk = (m / 2) as u64;
        let f = factorial(kk);
        let mut c = Q::new(BigInt::one(), &f * &f * (BigInt::one() << (2 * kk)));
        if alternate && kk % 2 == 1 {
            c = -c;
        }
        rat_expr(ar, c)
    };
    let tail = move |ar: &mut Arena, m: usize| -> ExprId {
        if m % 2 == 1 || m == 0 {
            return ar.zero;
        }
        let kk = (m / 2) as u64;
        let h: Q = (1..=kk)
            .map(|j| Q::new(BigInt::one(), BigInt::from(j)))
            .sum();
        let f = factorial(kk);
        let mut c = h / Q::from_integer(&f * &f * (BigInt::one() << (2 * kk)));
        if alternate && kk.is_multiple_of(2) {
            c = -c;
        }
        rat_expr(ar, c)
    };
    let i0 = TSeries::compose(arena, &bessel0, &u);
    let s = TSeries::compose(arena, &tail, &u);
    let ln_u = apply_ln(arena, &u, mode).map_err(|_| Obstruction::NoExpansion)?;
    // ln(u/2) + γ
    let shift = {
        let two = arena.int(2);
        let l2 = arena.ln(two);
        let g = arena.euler_gamma;
        let nl2 = arena.neg(l2);
        let s = arena.add(&[g, nl2]);
        eval::eval(arena, s)
    };
    let sc = constant_like(arena, shift, &ln_u);
    let lg = TSeries::add(arena, &ln_u, &sc);
    let prod = TSeries::mul(arena, &lg, &i0);
    if is_k {
        let np = TSeries::neg(arena, &prod);
        Ok(Some(TSeries::add(arena, &np, &s)))
    } else {
        let sum = TSeries::add(arena, &prod, &s);
        let two_over_pi = {
            let two = arena.int(2);
            let pi = arena.pi;
            let q = arena.div(two, pi);
            eval::eval(arena, q)
        };
        Ok(Some(TSeries::scale(arena, &sum, two_over_pi)))
    }
}

/// Taylor coefficients by repeated differentiation at `0`.
///
/// When direct substitution of `0` into a derivative is singular or
/// indeterminate (`atan(1/t)` at `t = 0` gives `atan(zoo)`), the
/// coefficient is the *limit* of that derivative at `0` — one-sided
/// (`0⁺`) for expansions at `±∞`, two-sided otherwise.  A limit that does
/// not exist or is not finite means there is no Taylor expansion.
fn taylor_by_differentiation(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    n: i64,
    mode: Mode,
) -> Option<TSeries> {
    let side = mode.side;
    let zero = arena.zero;

    let mut coeffs = Vec::with_capacity(n.max(0) as usize);
    let mut current = expr;
    let mut factorial = Q::one();
    // At a discontinuity of `sign`, `H`, `⌊·⌋`, … the value at the point is
    // not the one-sided limit: `tan(sign x)` substituted `sign 0 = 0` and
    // expanded to `0` from both sides.  With such nodes every coefficient
    // is the limit from the side of the expansion.
    let directional = crate::calculus::limit::has_directional_nodes(arena, expr, var);
    // Coefficients `0..n` are claimed exact up to `O(xⁿ)`, which needs the
    // `n`-th derivative bounded near the point (Taylor's remainder): its
    // limit is taken too and must be finite.  Without it `1/ln x` "expanded"
    // to `0` (`f(0⁺) = 0`, but `1/ln x` is not `O(x)`), and
    // `−2 − (x − 3/4)/ln x` to `−2`.
    for k in 0..=n {
        // A derivative that stays formal (`d re(u)/dx` for a variable not
        // declared real) has no value at the point; asking `limit` for one
        // sent Gruntz after `Subs(Derivative(…))` for seconds.
        if walk::has_unevaluated(arena, current) {
            return None;
        }
        // Derivatives of nested functions grow exponentially; the limit
        // taken of each (below) expands them, and `atanh(acosh x)³` at
        // `x = −1` never returned.
        if crate::transforms::pattern::tree_size_capped(arena, current, MAX_FALLBACK_SIZE + 1)
            > MAX_FALLBACK_SIZE
        {
            return None;
        }
        // `safe_substitute` refuses a singular sub-expression instead of
        // letting canonicalisation fold `0·sinh(ln 0)` to `0` (the series
        // of `x·sinh(ln x)` lost its constant term −1/2 that way).
        let value =
            crate::calculus::limit::safe_substitute(arena, current, var, zero).unwrap_or(arena.nan);
        if k == n {
            // Only a definitely unbounded `n`-th derivative refuses; one whose
            // limit cannot be determined (`d³(x·Γ(x))`) is given the benefit
            // of the doubt, as before.
            if !directional && is_finite_constant(arena, value, var) {
                break;
            }
            if !mode.limits {
                return None;
            }
            use crate::calculus::limit::Direction;
            let dirs: &[Direction] = match side {
                Side::Above => &[Direction::Right],
                Side::Below => &[Direction::Left],
                Side::Both => &[Direction::Right, Direction::Left],
            };
            for &dir in dirs {
                if let Ok(lim) =
                    crate::calculus::limit::limit_dir_generic(arena, current, var, zero, dir)
                    && !is_finite_constant(arena, lim, var)
                {
                    return None;
                }
            }
            break;
        }
        let value = if !directional && is_finite_constant(arena, value, var) {
            value
        } else {
            if !mode.limits {
                return None;
            }
            let dir = match side {
                Side::Above => crate::calculus::limit::Direction::Right,
                Side::Below => crate::calculus::limit::Direction::Left,
                Side::Both => crate::calculus::limit::Direction::Both,
            };
            let lim =
                crate::calculus::limit::limit_dir_generic(arena, current, var, zero, dir).ok()?;
            if !is_finite_constant(arena, lim, var) {
                return None;
            }
            lim
        };
        let coeff = if k == 0 {
            value
        } else {
            let inv = rat_expr(arena, Q::one() / &factorial);
            let p = arena.mul(&[value, inv]);
            eval::eval(arena, p)
        };
        coeffs.push(coeff);
        current = crate::transforms::diff::diff(arena, current, var);
        factorial *= rat_i(k + 1);
    }
    Some(TSeries {
        shift: 0,
        known: n,
        coeffs,
    })
}

/// A finite, fully evaluated constant (no `±∞`, `zoo`, `NaN`, hidden
/// singularity such as `ln 0`, unevaluated node, or occurrence of `var`).
fn is_finite_constant(arena: &Arena, value: ExprId, var: ExprId) -> bool {
    crate::calculus::limit::is_finite_limit_value(arena, value, var)
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

    /// Compare a series numerically against the target function at a small point.
    fn check_close(a: &mut Arena, series: ExprId, target: ExprId, x: ExprId, at: f64, tol: f64) {
        let nid = a.intern_num(Ratio::from_float(at).unwrap());
        let pt = a.intern(ExprNode::Num(nid));
        let s = subs::subs(a, series, x, pt);
        let t = subs::subs(a, target, x, pt);
        let sv = crate::transforms::evalf::eval_const_f64(a, s).unwrap();
        let tv = crate::transforms::evalf::eval_const_f64(a, t).unwrap();
        assert!(
            (sv - tv).abs() < tol,
            "series {} vs target {} at {at}: {sv} vs {tv}",
            display(a, series),
            display(a, target)
        );
    }

    #[test]
    fn series_constant() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let five = a.int(5);
        let zero = a.zero;
        let result = series(&mut a, five, x, zero, 3).unwrap();
        assert_eq!(display(&a, result), "5");
    }

    #[test]
    fn series_x_around_zero() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let zero = a.zero;
        let result = series(&mut a, x, x, zero, 3).unwrap();
        assert_eq!(display(&a, result), "x");
    }

    #[test]
    fn series_polynomial_is_exact() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let three = a.int(3);
        let x3 = a.pow(x, three);
        let zero = a.zero;
        let result = series(&mut a, x3, x, zero, 5).unwrap();
        assert_eq!(display(&a, result), "x^3");
    }

    #[test]
    fn series_order_zero_is_zero() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let zero = a.zero;
        let result = series(&mut a, x, x, zero, 0).unwrap();
        assert_eq!(result, a.zero);
    }

    #[test]
    fn series_exp_sin_cos_fast_paths() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let zero = a.zero;
        let e = a.exp(x);
        let s = series(&mut a, e, x, zero, 5).unwrap();
        assert_eq!(display(&a, s), "1/24*x^4 + 1/6*x^3 + 1/2*x^2 + x + 1");
        let sn = a.sin(x);
        let s = series(&mut a, sn, x, zero, 6).unwrap();
        assert_eq!(display(&a, s), "1/120*x^5 - 1/6*x^3 + x");
        let c = a.cos(x);
        let s = series(&mut a, c, x, zero, 5).unwrap();
        assert_eq!(display(&a, s), "1/24*x^4 - 1/2*x^2 + 1");
    }

    #[test]
    fn series_composition_sin_x_squared() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let zero = a.zero;
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let f = a.sin(x2);
        let s = series(&mut a, f, x, zero, 8).unwrap();
        assert_eq!(display(&a, s), "-1/6*x^6 + x^2");
    }

    #[test]
    fn series_sin_over_x_is_laurent_free() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let zero = a.zero;
        let sn = a.sin(x);
        let f = a.div(sn, x);
        let s = series(&mut a, f, x, zero, 5).unwrap();
        assert_eq!(display(&a, s), "1/120*x^4 - 1/6*x^2 + 1");
    }

    #[test]
    fn series_pole_gives_laurent() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let zero = a.zero;
        let one = a.one;
        // 1/(x(1−x)) = 1/x + 1 + x + x² + …
        let omx = a.sub(one, x);
        let den = a.mul(&[x, omx]);
        let f = a.div(one, den);
        let s = series(&mut a, f, x, zero, 3).unwrap();
        assert_eq!(display(&a, s), "x^2 + x + 1/x + 1");
    }

    #[test]
    fn series_tan_asin_erf_lambertw() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let zero = a.zero;
        let t = a.tan(x);
        let s = series(&mut a, t, x, zero, 8).unwrap();
        assert_eq!(display(&a, s), "17/315*x^7 + 2/15*x^5 + 1/3*x^3 + x");
        let th = a.tanh(x);
        let s = series(&mut a, th, x, zero, 6).unwrap();
        assert_eq!(display(&a, s), "2/15*x^5 - 1/3*x^3 + x");
        let asn = a.asin(x);
        let s = series(&mut a, asn, x, zero, 6).unwrap();
        assert_eq!(display(&a, s), "3/40*x^5 + 1/6*x^3 + x");
        let w = a.lambertw(x);
        let s = series(&mut a, w, x, zero, 5).unwrap();
        assert_eq!(display(&a, s), "-8/3*x^4 + 3/2*x^3 - x^2 + x");
        let e = a.erf(x);
        let s = series(&mut a, e, x, zero, 4).unwrap();
        check_close(&mut a, s, e, x, 0.1, 1e-5);
    }

    #[test]
    fn series_ln_and_binomial() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let zero = a.zero;
        let one = a.one;
        let opx = a.add(&[one, x]);
        let l = a.ln(opx);
        let s = series(&mut a, l, x, zero, 4).unwrap();
        assert_eq!(display(&a, s), "1/3*x^3 - 1/2*x^2 + x");
        let half = a.rational(1, 2);
        let sq = a.pow(opx, half);
        let s = series(&mut a, sq, x, zero, 4).unwrap();
        assert_eq!(display(&a, s), "1/16*x^3 - 1/8*x^2 + 1/2*x + 1");
        // ln(2 + x) = ln 2 + x/2 − x²/8 + …
        let two = a.int(2);
        let tpx = a.add(&[two, x]);
        let l2 = a.ln(tpx);
        let s = series(&mut a, l2, x, zero, 3).unwrap();
        check_close(&mut a, s, l2, x, 0.01, 1e-6);
    }

    #[test]
    fn series_around_nonzero_point() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.one;
        let e = a.exp(x);
        let s = series(&mut a, e, x, one, 4).unwrap();
        check_close(&mut a, s, e, x, 1.01, 1e-8);
    }

    #[test]
    fn puiseux_is_rejected() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let zero = a.zero;
        let sq = a.sqrt(x);
        let sn = a.sin(x);
        let f = a.mul(&[sq, sn]);
        assert!(series(&mut a, f, x, zero, 5).is_err());
        let l = a.ln(x);
        assert!(series(&mut a, l, x, zero, 5).is_err());
    }

    #[test]
    fn puiseux_is_rejected_below_the_singular_derivative() {
        // x^(5/2): the first three derivatives vanish at 0, so a
        // differentiation fallback at order 3 would return the wrong
        // polynomial 0.  The refusal must be definite at every order.
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let zero = a.zero;
        let e = a.rational(5, 2);
        let f = a.pow(x, e);
        assert!(series(&mut a, f, x, zero, 3).is_err());
        assert!(series(&mut a, f, x, zero, 6).is_err());
    }

    #[test]
    fn abs_of_even_valuation_is_signed_argument() {
        // |x²| = x²; |x − 1| = 1 − x near 0; |−x² + x³| = x² − x³.
        // sympy: series(Abs(x**2), x, 0, 6) == x**2;
        //        series(Abs(x - 1), x, 0, 6) == 1 - x
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let zero = a.zero;
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let f = a.abs(x2);
        let s = series(&mut a, f, x, zero, 6).unwrap();
        assert_eq!(s, x2);
        let one = a.one;
        let xm1 = a.sub(x, one);
        let g = a.abs(xm1);
        let s = series(&mut a, g, x, zero, 6).unwrap();
        let expected = a.sub(one, x);
        assert_eq!(s, expected);
        let three = a.int(3);
        let x3 = a.pow(x, three);
        let h_arg = a.sub(x3, x2);
        let h = a.abs(h_arg);
        let s = series(&mut a, h, x, zero, 6).unwrap();
        let expected = a.sub(x2, x3);
        assert_eq!(s, expected);
    }

    #[test]
    fn abs_of_odd_valuation_needs_both_sides_to_agree() {
        // |x| and |sin x| have a kink; cos|x| = cos x and |x|² = x² do not.
        // sympy: series(cos(Abs(x)), x, 0, 6) == x**4/24 - x**2/2 + 1
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let zero = a.zero;
        let ax = a.abs(x);
        assert!(series(&mut a, ax, x, zero, 6).is_err());
        let sn = a.sin(x);
        let asn = a.abs(sn);
        assert!(series(&mut a, asn, x, zero, 6).is_err());
        let ex = a.exp(ax);
        assert!(series(&mut a, ex, x, zero, 6).is_err());
        let cs = a.cos(ax);
        let s = series(&mut a, cs, x, zero, 6).unwrap();
        assert_eq!(display(&a, s), "1/24*x^4 - 1/2*x^2 + 1");
        let two = a.int(2);
        let ax2 = a.pow(ax, two);
        let s = series(&mut a, ax2, x, zero, 6).unwrap();
        let x2 = a.pow(x, two);
        assert_eq!(s, x2);
    }

    #[test]
    fn abs_at_infinity_is_one_sided() {
        // sympy: series(Abs(x), x, oo, 3) == x; series(Abs(x), x, -oo, 3) == -x
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let ax = a.abs(x);
        let s = series_at_infinity(&mut a, ax, x, 3, false).unwrap();
        assert_eq!(s, x);
        let s = series_at_infinity(&mut a, ax, x, 3, true).unwrap();
        let neg_x = a.neg(x);
        assert_eq!(s, neg_x);
    }

    #[test]
    fn series_at_infinity_rational() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.one;
        let xp1 = a.add(&[x, one]);
        let f = a.div(x, xp1);
        let s = series_at_infinity(&mut a, f, x, 3, false).unwrap();
        assert_eq!(display(&a, s), "x^(-2) - 1/x + 1");
        // sqrt(x²+1) − x  ~  1/(2x) − 1/(8x³)
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let x2p1 = a.add(&[x2, one]);
        let root = a.sqrt(x2p1);
        let g = a.sub(root, x);
        let s = series_at_infinity(&mut a, g, x, 4, false).unwrap();
        assert_eq!(display(&a, s), "-1/8*x^(-3) + 1/(2*x)");
    }

    #[test]
    fn exp_pow_with_variable_exponent() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let zero = a.zero;
        let one = a.one;
        // (1+x)^x = 1 + x² − x³/2 + …
        let opx = a.add(&[one, x]);
        let f = a.pow(opx, x);
        let s = series(&mut a, f, x, zero, 4).unwrap();
        assert_eq!(display(&a, s), "-1/2*x^3 + x^2 + 1");
    }

    #[test]
    fn large_integer_powers_use_binomial_series() {
        // Beyond MAX_INT_POWER repeated multiplication is replaced by the
        // binomial series; the result must be exact and fast.
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let zero = a.zero;
        let one = a.one;
        let opx = a.add(&[one, x]);
        let big = a.int(MAX_INT_POWER + 36); // 100
        let f = a.pow(opx, big);
        let start = std::time::Instant::now();
        let s = series(&mut a, f, x, zero, 3).unwrap();
        assert!(start.elapsed().as_secs_f64() < 1.0);
        assert_eq!(display(&a, s), "4950*x^2 + 100*x + 1");
        // negative: (1+x)^(-70) = 1 − 70x + 2485x²
        let neg = a.int(-(MAX_INT_POWER + 6));
        let f = a.pow(opx, neg);
        let s = series(&mut a, f, x, zero, 3).unwrap();
        assert_eq!(display(&a, s), "2485*x^2 - 70*x + 1");
        // with a zero at the origin: (x + x²)^70 = x^70 + 70 x^71 + …
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let base = a.add(&[x, x2]);
        let e70 = a.int(70);
        let f = a.pow(base, e70);
        let s = series(&mut a, f, x, zero, 72).unwrap();
        assert_eq!(display(&a, s), "70*x^71 + x^70");
        // odd valuation with a negative integer exponent is fine two-sided:
        // (x + x²)^(-65) = x^(-65) (1 + x)^(-65) = x^(-65) − 65 x^(-64) + …
        // (the pole of order 65 forces the precision-escalation retry)
        let em65 = a.int(-65);
        let f = a.pow(base, em65);
        let ts = expand_meromorphic(&mut a, f, x, 1).unwrap();
        assert_eq!(ts.shift(), -65);
        assert!(ts.known() >= 1);
        let c = ts.coefficient(&a, -65);
        assert_eq!(display(&a, c), "1");
        let c = ts.coefficient(&a, -64);
        assert_eq!(display(&a, c), "-65");
        let c = ts.coefficient(&a, -63);
        assert_eq!(display(&a, c), "2145");
    }

    #[test]
    fn low_order_requests_still_see_the_pole() {
        // At order 1 the variable itself would be truncated away at the
        // requested precision; the engine must widen its working window.
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let zero = a.zero;
        let one = a.one;
        let inv = a.div(one, x);
        let s = series(&mut a, inv, x, zero, 1).unwrap();
        assert_eq!(display(&a, s), "1/x");
        // 1/(x⁵ + x⁶) = x⁻⁵ − x⁻⁴ + x⁻³ − …  (valuation 5 hidden at precision 4)
        let five = a.int(5);
        let six = a.int(6);
        let x5 = a.pow(x, five);
        let x6 = a.pow(x, six);
        let d = a.add(&[x5, x6]);
        let f = a.div(one, d);
        let s = series(&mut a, f, x, zero, 1).unwrap();
        assert_eq!(
            display(&a, s),
            "1/x + x^(-3) + x^(-5) - x^(-2) - x^(-4) - 1"
        );
        // A genuine failure is still reported after the bounded retries.
        let l = a.ln(x);
        let start = std::time::Instant::now();
        assert!(series(&mut a, l, x, zero, 1).is_err());
        assert!(start.elapsed().as_secs_f64() < 1.0);
    }

    #[test]
    fn tan_coefficients_match_bernoulli_formula() {
        // 1, 1/3, 2/15, 17/315, 62/2835
        let expected = [(1, 1), (3, 1), (5, 2), (7, 17), (9, 62)];
        let denoms = [1i64, 3, 15, 315, 2835];
        for (i, (n, num)) in expected.iter().enumerate() {
            let c = FnKind::Tan.rational_coefficient(*n as usize);
            assert_eq!(
                c,
                Q::new(BigInt::from(*num), BigInt::from(denoms[i])),
                "n={n}"
            );
        }
    }
}

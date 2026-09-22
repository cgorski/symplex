//! Taylor / Laurent series expansion.
//!
//! [`series`] computes the truncated expansion of an expression around a
//! point (Maclaurin when the point is `0`, asymptotic when the point is
//! `±∞` via [`series_at_infinity`]).
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
//! * Any other `var`-dependent sub-expression falls back to Taylor
//!   coefficients by differentiation, evaluated at the expansion point.
//!
//! Fractional powers of a series with a zero constant term (Puiseux
//! expansions such as `√x·sin x`) and logarithmic singularities are
//! rejected — the caller then keeps an unevaluated `Series` node instead of
//! producing a wrong polynomial.  Such rejections are *definite*: the
//! differentiation fallback is never tried for them, because the low-order
//! derivatives of `x^(5/2)` or `|x²|` all vanish at `0` and would silently
//! yield the wrong polynomial `0`.
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
use crate::transforms::{eval, subs};

/// Maximum number of sub-expressions expanded by differentiation before the
/// engine gives up on per-node fallbacks (the root is always tried).
const MAX_FALLBACK_NODES: usize = 4;

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
/// exponent `< order` (in `var − point`).
///
/// For `point = 0` this is the Maclaurin series; for `point = ±∞` an
/// asymptotic expansion in `1/var` (see [`series_at_infinity`]).  Poles at
/// the expansion point produce negative powers (Laurent series).
///
/// Returns `Err` when no Laurent expansion exists (fractional-power or
/// logarithmic singularity, essential singularity, or an unsupported
/// sub-expression whose derivatives are singular at the point).
pub(crate) fn series(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    order: u32,
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
    let ts = expand_maclaurin(arena, shifted, var, order as i64, false)?;
    let poly = ts.to_expr(arena, var, order as i64);
    if at_zero {
        Ok(poly)
    } else {
        let x_minus_a = arena.sub(var, point);
        let back = subs::subs(arena, poly, var, x_minus_a);
        Ok(eval::eval(arena, back))
    }
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
    // t = 1/x → 0⁺ only, so fractional powers of t^(even) are single-valued.
    let ts = expand_maclaurin(arena, in_t, t, order as i64, true)?;
    let poly = ts.to_expr(arena, t, order as i64);
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
/// Kept for API compatibility: the main [`series`] engine already produces
/// Laurent expansions.  As a last resort this multiplies by `(x − a)^k`,
/// `k = 1..=5`, and divides back.
pub(crate) fn laurent_series(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    order: u32,
) -> Result<ExprId, SymplexError> {
    if let Ok(ts) = series(arena, expr, var, point, order) {
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
        if let Ok(ts) = series(arena, modified, var, point, order + k) {
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

    /// Do `a` and `b` agree on every coefficient of exponent `< order`?
    /// (Structural comparison of the evaluated coefficients.)
    fn same(arena: &Arena, a: &TSeries, b: &TSeries, order: i64) -> bool {
        let lo = a.shift.min(b.shift);
        let hi = a.known.min(b.known).min(order);
        (lo..hi).all(|e| a.coeff_at(arena, e) == b.coeff_at(arena, e))
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
            coeffs.push(eval::eval(arena, s));
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
            let s = match terms.len() {
                0 => arena.zero,
                1 => terms[0],
                _ => arena.add(&terms),
            };
            coeffs.push(eval::eval(arena, s));
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
            b.push(eval::eval(arena, ns));
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
    /// From below only (`x → 0⁻`; used to cross-check `|g|` with odd
    /// valuation, see [`expand_maclaurin`]).
    Below,
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
}

/// Expand `expr` around `var = 0` with all exponents `< order` exact.
///
/// `one_sided` marks expansions where the variable only approaches `0`
/// from above (used for `x → ±∞` via `t = 1/x`); this permits
/// `(t^v)^α = t^{vα}` for every integer `vα`, which is not valid two-sided
/// (`√(x²) = |x|`).
///
/// A two-sided expansion that fails only because of `|g|` with odd
/// valuation (`|x|`, `|sin x|`) is retried from each side separately and
/// accepted when both sides agree (`cos|x| = cos x`); otherwise there is
/// no two-sided expansion and the error is returned.
pub(crate) fn expand_maclaurin(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    order: i64,
    one_sided: bool,
) -> Result<TSeries, SymplexError> {
    let side = if one_sided { Side::Above } else { Side::Both };
    let mut kink = false;
    match expand_from_side(arena, expr, var, order, side, &mut kink) {
        Err(_) if kink => {
            let above = expand_from_side(arena, expr, var, order, Side::Above, &mut false)?;
            let below = expand_from_side(arena, expr, var, order, Side::Below, &mut false)?;
            if TSeries::same(arena, &above, &below, order) {
                Ok(above)
            } else {
                Err(SymplexError::ComputationFailed {
                    operation: "series",
                    reason: format!(
                        "no two-sided expansion: the expansions of {} from the left and from the right differ (|·| with odd valuation)",
                        arena.display(expr)
                    ),
                })
            }
        }
        r => r,
    }
}

/// [`expand_maclaurin`] for one [`Side`], with the precision-escalation loop.
/// Sets `kink` when the failure was an odd-valuation `|g|` in a two-sided
/// expansion.
fn expand_from_side(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    order: i64,
    side: Side,
    kink: &mut bool,
) -> Result<TSeries, SymplexError> {
    // Work with at least a few terms so that the valuation of every
    // sub-expression is visible (at precision 1 the variable itself would
    // truncate to nothing and `1/x` could not be expanded).
    let mut working = order.max(MIN_WORKING_ORDER);
    let mut last_err = None;
    for _attempt in 0..MAX_PRECISION_ATTEMPTS {
        let mut hidden_valuation = false;
        match expand_with_precision(arena, expr, var, working, side, &mut hidden_valuation, kink) {
            Ok(ts) if ts.known >= order => return Ok(ts.truncate_known(order)),
            Ok(ts) => {
                // Precision was lost through poles; increase and retry.
                working += order - ts.known + 1;
            }
            Err(e) if hidden_valuation && !*kink => {
                // A sub-expression's leading term lay beyond the working
                // window (e.g. `1/(x⁵ + x⁶)` at low order): widen and retry.
                last_err = Some(e);
                working = working * 2 + 4;
            }
            Err(e) => return Err(e),
        }
    }
    Err(last_err.unwrap_or(SymplexError::ComputationFailed {
        operation: "series",
        reason: "could not reach the requested order (deep pole)".into(),
    }))
}

/// One pass of the engine at working precision `n`.  Sets `hidden_valuation`
/// when some `var`-dependent sub-expression had *no* visible term at this
/// precision, so a failure may be curable by widening the window; sets
/// `kink` when the failure was [`Obstruction::Kink`].
fn expand_with_precision(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    n: i64,
    side: Side,
    hidden_valuation: &mut bool,
    kink: &mut bool,
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
            match structural_series(arena, id, var, side, &cache) {
                Ok(s) => Some(s),
                Err(Obstruction::Unknown) => {
                    if id == expr || fallbacks_used < MAX_FALLBACK_NODES {
                        fallbacks_used += 1;
                        taylor_by_differentiation(arena, id, var, n, side)
                    } else {
                        None
                    }
                }
                Err(Obstruction::Kink) => {
                    *kink = true;
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
    side: Side,
    cache: &FxHashMap<ExprId, Option<TSeries>>,
) -> Result<TSeries, Obstruction> {
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
            let lnb = apply_ln(arena, &b).ok_or(Obstruction::Unknown)?;
            let prod = TSeries::mul(arena, &e, &lnb);
            apply_fn(arena, FnKind::Exp, &prod)
        }
        ExprNode::Abs(a) => return abs_series(arena, &child(cache, a)?, side),
        ExprNode::Exp(a) => apply_fn(arena, FnKind::Exp, &child(cache, a)?),
        ExprNode::Sin(a) => apply_fn(arena, FnKind::Sin, &child(cache, a)?),
        ExprNode::Cos(a) => apply_fn(arena, FnKind::Cos, &child(cache, a)?),
        ExprNode::Sinh(a) => apply_fn(arena, FnKind::Sinh, &child(cache, a)?),
        ExprNode::Cosh(a) => apply_fn(arena, FnKind::Cosh, &child(cache, a)?),
        ExprNode::Tan(a) => apply_fn(arena, FnKind::Tan, &child(cache, a)?),
        ExprNode::Tanh(a) => apply_fn(arena, FnKind::Tanh, &child(cache, a)?),
        ExprNode::Atan(a) => apply_fn(arena, FnKind::Atan, &child(cache, a)?),
        ExprNode::Atanh(a) => apply_fn(arena, FnKind::Atanh, &child(cache, a)?),
        ExprNode::Asin(a) => apply_fn(arena, FnKind::Asin, &child(cache, a)?),
        ExprNode::Asinh(a) => apply_fn(arena, FnKind::Asinh, &child(cache, a)?),
        ExprNode::Erf(a) => apply_fn(arena, FnKind::Erf, &child(cache, a)?),
        ExprNode::LambertW(a) => apply_fn(arena, FnKind::LambertW, &child(cache, a)?),
        ExprNode::Ln(a) => apply_ln(arena, &child(cache, a)?),
        _ => None,
    };
    analytic.ok_or(Obstruction::Unknown)
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

/// `Some(true)` if the var-free constant `c` is known positive, `Some(false)`
/// if known negative, `None` if zero or of unknown sign.
fn constant_sign(arena: &Arena, c: ExprId) -> Option<bool> {
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
    None
}

/// Is the var-free constant `c` known to be real?
fn is_known_real(arena: &Arena, c: ExprId) -> bool {
    arena.as_num(c).is_some() || {
        let mut cache = crate::base::assumptions::AssumptionCache::new();
        cache.query(arena, c, Props::REAL) == Some(true)
    }
}

fn is_zero_const(arena: &mut Arena, c: ExprId) -> bool {
    let v = eval::eval(arena, c);
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

/// `ln(a)` — requires a non-zero constant term.
fn apply_ln(arena: &mut Arena, a: &TSeries) -> Option<TSeries> {
    let a = a.clone().normalized(arena);
    if a.leading_exponent(arena)? != 0 {
        return None; // logarithmic singularity
    }
    let (u0, w) = a.split_constant(arena);
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
    let ln_u0 = arena.ln(u0);
    let ln_u0 = eval::eval(arena, ln_u0);
    if !arena.is_zero_structural(ln_u0) && !s.coeffs.is_empty() && s.shift <= 0 {
        let idx = (-s.shift) as usize;
        let c = arena.add(&[s.coeffs[idx], ln_u0]);
        s.coeffs[idx] = eval::eval(arena, c);
    }
    Some(s)
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
    let u0a = eval::eval(arena, u0a);
    let mut r = TSeries::scale(arena, &s, u0a);
    r.shift += outer_shift;
    r.known += outer_shift;
    Ok(r)
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
    side: Side,
) -> Option<TSeries> {
    let zero = arena.zero;
    let mut coeffs = Vec::with_capacity(n.max(0) as usize);
    let mut current = expr;
    let mut factorial = Q::one();
    for k in 0..n {
        let at0 = subs::subs(arena, current, var, zero);
        let value = eval::eval(arena, at0);
        let value = if is_finite_constant(arena, value, var) {
            value
        } else {
            let dir = match side {
                Side::Above => crate::calculus::limit::Direction::Right,
                Side::Below => crate::calculus::limit::Direction::Left,
                Side::Both => crate::calculus::limit::Direction::Both,
            };
            let lim = crate::calculus::limit::limit_dir(arena, current, var, zero, dir).ok()?;
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
        if k + 1 < n {
            current = crate::transforms::diff::diff(arena, current, var);
            factorial *= rat_i(k + 1);
        }
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
    value != arena.infinity
        && value != arena.neg_infinity
        && crate::calculus::limit::is_valid_limit_value(arena, value, var)
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
        let ts = expand_maclaurin(&mut a, f, x, 1, false).unwrap();
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

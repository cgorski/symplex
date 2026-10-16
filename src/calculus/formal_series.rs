//! Formal power series with exact, lazily computed coefficients.
//!
//! A [`FormalPowerSeries`] represents `f(x) = Σ_{k≥0} a_k (x − a)^k` (or a
//! Laurent series when `f` has a pole at `a`).  Every coefficient is an
//! exact symbolic expression ([`Ex`]) — rational numbers for elementary
//! functions with rational Maclaurin coefficients, expressions such as
//! `2/√π` or `ln 2` otherwise.
//!
//! # Design
//!
//! Coefficients are computed **lazily and memoised**; there is no fixed
//! truncation order.  A series is one of:
//!
//! * a **closed-form** series (`exp`, `sin`, `cos`, `sinh`, `cosh`, `tan`,
//!   `tanh`, `atan`, `atanh`, `asin`, `asinh`, `erf`, `W`, `ln(a + c·xᵐ)`,
//!   `(a + c·xᵐ)^α`), whose `k`-th coefficient — and general term
//!   [`general_term`](FormalPowerSeries::general_term) — are known in
//!   closed form;
//! * an **engine** series for any other expression, expanded on demand by
//!   the truncated-series engine behind [`Ex::series`](crate::api::expr::Ex::series) (poles are
//!   allowed: negative exponents are available through
//!   [`truncate`](FormalPowerSeries::truncate));
//! * an explicit **polynomial** (finite coefficient list);
//! * a **derived** series: `add`, `sub`, `mul`, `scale`, `compose`,
//!   `derivative`, `integral`, `inverse` (`1/f`, needs `f(a) ≠ 0` or a
//!   pole) or `reversion` (compositional inverse by Lagrange inversion,
//!   needs `f(a) = 0`, `f'(a) ≠ 0`) of other series.  Derived coefficients
//!   are computed from the operands' coefficients exactly (principal parts
//!   included), to any index.
//!
//! # Puiseux series and symbolic powers
//!
//! In general a series is `(x − a)^ρ · Σ_k a_k (x − a)^(k/d)`: the
//! [`offset`](FormalPowerSeries::offset) `ρ` is a non-integer rational or a
//! symbolic exponent (`√x·eˣ = x^(1/2)·Σ xᵏ/k!`, `x^a·eˣ = x^a·Σ xᵏ/k!`)
//! and the [`ramification`](FormalPowerSeries::ramification) `d` allows
//! fractional steps (`exp(√x) = Σ x^(k/2)/k!`).  `coefficient(k)` and
//! `general_term` refer to the term of exponent `ρ + k/d`
//! ([`exponent`](FormalPowerSeries::exponent)); for a Taylor or Laurent
//! series `ρ = 0`, `d = 1` and nothing changes.
//!
//! A function with no Laurent or Puiseux expansion at the point (`ln x`,
//! `exp(1/x)`, `|x|` at 0) gives a [refused](FormalPowerSeries::is_refused)
//! series: its coefficients are unevaluated stand-ins (never `nan`), its
//! truncation a formal `Series` node, and
//! [`try_coefficient`](FormalPowerSeries::try_coefficient) /
//! [`try_truncate`](FormalPowerSeries::try_truncate) return `Err`.
//!
//! Series are created with [`Ex::fps`](crate::api::expr::Ex::fps) /
//! [`Ex::fps_maclaurin`](crate::api::expr::Ex::fps_maclaurin) or
//! [`FormalPowerSeries::from_coefficients`].
//!
//! ```
//! use symplex::prelude::*;
//!
//! let ctx = Context::new();
//! let x = ctx.symbol("x");
//! let e = x.exp().fps_maclaurin(&x);
//! assert_eq!(e.coefficient(5).to_string(), "1/120");
//! assert_eq!(e.truncate(4).to_string(), "1/6*x^3 + 1/2*x^2 + x + 1");
//!
//! // (e^x)² = e^{2x}: a_3 = 8/6 = 4/3
//! let sq = e.mul(&e).unwrap();
//! assert_eq!(sq.coefficient(3).to_string(), "4/3");
//!
//! // A Puiseux series: x^(1/3)·sin x = x^(4/3) − x^(10/3)/6 + …
//! let p = (&x.cbrt() * &x.sin()).fps_maclaurin(&x);
//! assert_eq!(p.truncate(4).to_string(), "x^(4/3) - 1/6*x^(10/3)");
//! ```

use std::sync::{Arc, Mutex};

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Zero};

use crate::api::context::Context;
use crate::api::expr::Ex;
use crate::base::arena::Arena;
use crate::base::cast;
use crate::base::combinatorics::factorial;
use crate::base::errors::SymplexError;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::Q;
use crate::base::walk;
use crate::calculus::series::{FnKind, TSeries, expand_meromorphic};
use crate::transforms::eval;

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

/// A formal power series `Σ a_k (x − point)^k` with exact, lazily computed
/// coefficients — more generally a Laurent or Puiseux series
/// `(x − point)^ρ · Σ_k a_k (x − point)^(k/d)` (see
/// [`offset`](Self::offset) and [`ramification`](Self::ramification)).
/// See the [module docs](self) for the design.
pub struct FormalPowerSeries {
    ctx: Context,
    var: ExprId,
    point: ExprId,
    /// Exponent offset `ρ` (zero for a Taylor or Laurent series): a
    /// non-integer rational or an expression free of `var`.
    offset: ExprId,
    /// Ramification `d ≥ 1`: index `k` is the exponent `ρ + k/d`.
    ram: u32,
    /// No coefficient below this index is non-zero (negative for a
    /// principal part).
    valuation: i64,
    /// `Σ r_p (x − point)^p ln(x − point)` (`p ≥ 0`), from `integral` of a
    /// series with an `(x − point)^(−1)` term; empty otherwise.
    log_terms: Vec<(i64, ExprId)>,
    source: Source,
    /// Memoised coefficients `a_valuation, a_valuation+1, …`.
    memo: Mutex<Vec<Coeff>>,
    zero: ExprId,
}

/// A memoised coefficient; `exact` is false for an unevaluated stand-in
/// (the expansion does not exist, or failed at this order).
#[derive(Clone, Copy, Debug)]
struct Coeff {
    id: ExprId,
    exact: bool,
}

/// Closed-form elementary series `f(c · xᵐ)` (all with `f(0)` known).
#[derive(Clone, Debug)]
enum Known {
    /// `f(c·xᵐ)` for an elementary `f`.
    Elementary { kind: FnKind, c: ExprId, m: usize },
    /// `ln(a + c·xᵐ) = ln a + Σ_{n≥1} (−1)^{n+1} (c/a)^n x^{mn} / n`
    Ln { a: ExprId, c: ExprId, m: usize },
    /// `(a + c·xᵐ)^α = a^α Σ C(α, n) (c/a)^n x^{mn}`
    Binomial {
        alpha: ExprId,
        a: ExprId,
        c: ExprId,
        m: usize,
    },
}

enum Source {
    /// A closed-form series in the series variable (`var`, or
    /// `(x − point)^(1/d)` for a ramified series).
    Known(Known),
    /// Expanded on demand by the truncated-series engine, `expr` in `svar`
    /// (the shifted function, or its ramified form in a placeholder).
    Engine {
        expr: ExprId,
        svar: ExprId,
        cache: Mutex<Option<TSeries>>,
    },
    Polynomial(Vec<ExprId>),
    /// No Laurent or Puiseux expansion of `expr` about the point exists
    /// (`ln x`, `exp(1/x)`, `|x|` at 0): coefficients are unevaluated
    /// stand-ins and `truncate` is a formal `Series` node.
    Refused {
        expr: ExprId,
        reason: String,
    },
    /// Operands share `offset` and `ram` (aligned by `add`/`sub`).
    Add(Box<FormalPowerSeries>, Box<FormalPowerSeries>),
    Sub(Box<FormalPowerSeries>, Box<FormalPowerSeries>),
    /// Operands share `ram`; the offsets add.
    Mul(Box<FormalPowerSeries>, Box<FormalPowerSeries>),
    Scale(Box<FormalPowerSeries>, ExprId),
    Compose(Box<FormalPowerSeries>, Box<FormalPowerSeries>),
    Derivative(Box<FormalPowerSeries>),
    Integral(Box<FormalPowerSeries>),
    /// `1/f`, with the index of `f`'s leading coefficient.
    Inverse(Box<FormalPowerSeries>, i64),
    Reversion(Box<FormalPowerSeries>),
    /// The same function under another `(offset, ram)`: coefficient `k` is
    /// the inner coefficient `(k·mul − shift)/div` (zero unless integral).
    Reindex {
        inner: Box<FormalPowerSeries>,
        mul: u32,
        div: u32,
        shift: i64,
    },
}

impl Clone for FormalPowerSeries {
    fn clone(&self) -> Self {
        // Lock order memo → cache, as in `coeff` (memo → context → cache).
        let memo = self.memo.lock().map(|m| m.clone()).unwrap_or_default();
        let source = match &self.source {
            Source::Known(k) => Source::Known(k.clone()),
            Source::Engine { expr, svar, cache } => Source::Engine {
                expr: *expr,
                svar: *svar,
                cache: Mutex::new(cache.lock().ok().and_then(|c| c.clone())),
            },
            Source::Polynomial(v) => Source::Polynomial(v.clone()),
            Source::Refused { expr, reason } => Source::Refused {
                expr: *expr,
                reason: reason.clone(),
            },
            Source::Add(a, b) => Source::Add(a.clone(), b.clone()),
            Source::Sub(a, b) => Source::Sub(a.clone(), b.clone()),
            Source::Mul(a, b) => Source::Mul(a.clone(), b.clone()),
            Source::Scale(a, c) => Source::Scale(a.clone(), *c),
            Source::Compose(a, b) => Source::Compose(a.clone(), b.clone()),
            Source::Derivative(a) => Source::Derivative(a.clone()),
            Source::Integral(a) => Source::Integral(a.clone()),
            Source::Inverse(a, l) => Source::Inverse(a.clone(), *l),
            Source::Reversion(a) => Source::Reversion(a.clone()),
            Source::Reindex {
                inner,
                mul,
                div,
                shift,
            } => Source::Reindex {
                inner: inner.clone(),
                mul: *mul,
                div: *div,
                shift: *shift,
            },
        };
        FormalPowerSeries {
            ctx: self.ctx.clone(),
            var: self.var,
            point: self.point,
            offset: self.offset,
            ram: self.ram,
            valuation: self.valuation,
            log_terms: self.log_terms.clone(),
            source,
            memo: Mutex::new(memo),
            zero: self.zero,
        }
    }
}

impl std::fmt::Debug for FormalPowerSeries {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FormalPowerSeries({})", self.truncate(6))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

fn rat_expr(arena: &mut Arena, r: Q) -> ExprId {
    let nid = arena.intern_num(r);
    arena.intern(ExprNode::Num(nid))
}

fn rat_i(n: i64) -> Q {
    Ratio::from_integer(BigInt::from(n))
}

fn add_all(arena: &mut Arena, terms: &[ExprId]) -> ExprId {
    let s = match terms.len() {
        0 => arena.zero,
        1 => terms[0],
        _ => arena.add(terms),
    };
    eval::eval(arena, s)
}

fn mul_all(arena: &mut Arena, factors: &[ExprId]) -> ExprId {
    let p = match factors.len() {
        0 => arena.one,
        1 => factors[0],
        _ => arena.mul(factors),
    };
    eval::eval(arena, p)
}

fn pow_i(arena: &mut Arena, base: ExprId, n: i64) -> ExprId {
    if n == 0 {
        return arena.one;
    }
    if n == 1 {
        return base;
    }
    let e = arena.int(n);
    let p = arena.pow(base, e);
    eval::eval(arena, p)
}

/// Generalised binomial `C(α, n)`.
fn gen_binomial(arena: &mut Arena, alpha: ExprId, n: usize) -> ExprId {
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
    factors.push(rat_expr(arena, Q::new(BigInt::one(), factorial(n as u64))));
    mul_all(arena, &factors)
}

/// Truncated polynomial product `(a·b)[0..n]`.
fn poly_mul(arena: &mut Arena, a: &[ExprId], b: &[ExprId], n: usize) -> Vec<ExprId> {
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let mut terms = Vec::new();
        for (i, &ai) in a.iter().enumerate().take(k + 1) {
            let j = k - i;
            if j >= b.len() || arena.is_zero_structural(ai) || arena.is_zero_structural(b[j]) {
                continue;
            }
            terms.push(arena.mul(&[ai, b[j]]));
        }
        out.push(add_all(arena, &terms));
    }
    out
}

/// Coefficients of `1/a` to order `n` (requires `a[0] ≠ 0`).
fn poly_inverse(arena: &mut Arena, a: &[ExprId], n: usize) -> Vec<ExprId> {
    let a0 = a.first().copied().unwrap_or(arena.zero);
    let inv_a0 = {
        let m1 = arena.neg_one;
        let p = arena.pow(a0, m1);
        eval::eval(arena, p)
    };
    let mut b: Vec<ExprId> = Vec::with_capacity(n);
    b.push(inv_a0);
    for k in 1..n {
        let mut terms = Vec::new();
        for i in 1..=k {
            let ai = a.get(i).copied().unwrap_or(arena.zero);
            if arena.is_zero_structural(ai) || arena.is_zero_structural(b[k - i]) {
                continue;
            }
            terms.push(arena.mul(&[ai, b[k - i]]));
        }
        let s = add_all(arena, &terms);
        let v = arena.mul(&[inv_a0, s]);
        let v = arena.neg(v);
        b.push(eval::eval(arena, v));
    }
    b
}

fn is_zero_expr(arena: &mut Arena, e: ExprId) -> bool {
    let v = eval::eval(arena, e);
    arena.is_zero_structural(v) || arena.as_num(v).is_some_and(|r| r.is_zero())
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match m.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    }
}

/// `lcm(a, b)`, `None` above [`MAX_RAMIFICATION`].
fn lcm_u32(a: u32, b: u32) -> Option<u32> {
    let (a, b) = (u64::from(a), u64::from(b));
    let l = a / crate::base::numeric::gcd_u64(a, b).max(1) * b;
    u32::try_from(l).ok().filter(|&l| l <= MAX_RAMIFICATION)
}

/// The largest ramification index tried or produced (`x^(1/d)`).
const MAX_RAMIFICATION: u32 = 360;

/// Stand-in for `(x − point)^(1/d)` while a ramified series is built.
const RAMIFIED_PLACEHOLDER: &str = "__fps_t";

/// The value of a number expression as an `i64` when it is an integer.
fn as_integer(arena: &Arena, e: ExprId) -> Option<i64> {
    use num_traits::ToPrimitive;
    let q = arena.as_num(e)?;
    if q.is_integer() {
        q.to_integer().to_i64()
    } else {
        None
    }
}

/// `q` as an `i64` when it is an integer.
fn q_to_i64(q: &Q) -> Option<i64> {
    use num_traits::ToPrimitive;
    if q.is_integer() {
        q.to_integer().to_i64()
    } else {
        None
    }
}

/// The rational part of an exponent: `3/2` for `a + 3/2`, `0` for `a`.
fn rational_part(arena: &Arena, e: ExprId) -> Q {
    if let Some(q) = arena.as_num(e) {
        return q.clone();
    }
    let mut acc = Q::zero();
    if let ExprNode::Add(args) = arena.node(e) {
        for &t in args {
            if let Some(q) = arena.as_num(t) {
                acc += q;
            }
        }
    }
    acc
}

fn contains_singular(arena: &Arena, id: ExprId) -> bool {
    walk::contains(arena, id, arena.nan)
        || walk::contains(arena, id, arena.complex_infinity)
        || walk::contains(arena, id, arena.infinity)
        || walk::contains(arena, id, arena.neg_infinity)
}

/// The most coefficients past the valuation a series computes (its prefix
/// is memoised, and composition and reversion lay it out densely):
/// `cast::MAX_SERIES_TERMS`.  Up to 0.58 a user's `coefficient(k)` built
/// the whole prefix `a_v … a_k`, and `coefficient(10¹²)` asked the allocator
/// for terabytes.
const MAX_COEFFICIENTS: usize = cast::MAX_SERIES_TERMS;

/// The stand-in `Res_{var=point} f·(var − point)^−(k+1)` for the `k`-th
/// coefficient beyond [`MAX_COEFFICIENTS`] (the derivative form of
/// [`taylor_standin`] has `k` nested nodes).
fn residue_standin(arena: &mut Arena, f: ExprId, var: ExprId, point: ExprId, k: i64) -> ExprId {
    let base = if arena.is_zero_structural(point) {
        var
    } else {
        arena.sub(var, point)
    };
    let e = arena.int(k.saturating_add(1).saturating_neg());
    let p = arena.pow(base, e);
    let body = arena.mul(&[f, p]);
    arena.intern(ExprNode::Residue(body, var, point))
}

/// The unevaluated `k`-th Taylor coefficient `Subs(dᵏ f/dxᵏ, x, point)/k!`,
/// returned where the coefficient cannot be computed (never `nan`).
fn taylor_standin(arena: &mut Arena, f: ExprId, var: ExprId, point: ExprId, k: i64) -> ExprId {
    let k = u64::try_from(k).unwrap_or(0);
    let mut body = f;
    for _ in 0..k {
        body = arena.intern(ExprNode::Derivative(body, var));
    }
    let at = arena.make(ExprNode::Subs(body, var, point));
    if k <= 1 {
        return at;
    }
    let inv = rat_expr(arena, Q::new(BigInt::one(), factorial(k)));
    arena.mul(&[inv, at])
}

// ═══════════════════════════════════════════════════════════════════════════
// Known-function detection
// ═══════════════════════════════════════════════════════════════════════════

/// `arg = c · var^m` with `m ≥ 1` and `c` free of `var`.
fn as_monomial(arena: &mut Arena, arg: ExprId, var: ExprId) -> Option<(ExprId, usize)> {
    let terms = crate::calculus::summation::sym_poly_in(arena, arg, var)?;
    if terms.len() != 1 || terms[0].0 == 0 {
        return None;
    }
    Some((terms[0].1, terms[0].0))
}

/// `arg = a + c · var^m` with `a ≠ 0`, `c` free of `var`, `m ≥ 1`.
fn as_shifted_monomial(
    arena: &mut Arena,
    arg: ExprId,
    var: ExprId,
) -> Option<(ExprId, ExprId, usize)> {
    let terms = crate::calculus::summation::sym_poly_in(arena, arg, var)?;
    if terms.len() != 2 || terms[0].0 != 0 {
        return None;
    }
    let a = terms[0].1;
    if is_zero_expr(arena, a) {
        return None;
    }
    Some((a, terms[1].1, terms[1].0))
}

fn detect_known(arena: &mut Arena, expr: ExprId, var: ExprId) -> Option<Known> {
    let node = arena.node(expr).clone();
    let elementary = |kind: FnKind, arg: ExprId, arena: &mut Arena| -> Option<Known> {
        let (c, m) = as_monomial(arena, arg, var)?;
        Some(Known::Elementary { kind, c, m })
    };
    match node {
        ExprNode::Exp(a) => elementary(FnKind::Exp, a, arena),
        ExprNode::Sin(a) => elementary(FnKind::Sin, a, arena),
        ExprNode::Cos(a) => elementary(FnKind::Cos, a, arena),
        ExprNode::Sinh(a) => elementary(FnKind::Sinh, a, arena),
        ExprNode::Cosh(a) => elementary(FnKind::Cosh, a, arena),
        ExprNode::Tan(a) => elementary(FnKind::Tan, a, arena),
        ExprNode::Tanh(a) => elementary(FnKind::Tanh, a, arena),
        ExprNode::Atan(a) => elementary(FnKind::Atan, a, arena),
        ExprNode::Atanh(a) => elementary(FnKind::Atanh, a, arena),
        ExprNode::Asin(a) => elementary(FnKind::Asin, a, arena),
        ExprNode::Asinh(a) => elementary(FnKind::Asinh, a, arena),
        ExprNode::Erf(a) => elementary(FnKind::Erf, a, arena),
        ExprNode::LambertW(a) => elementary(FnKind::LambertW, a, arena),
        ExprNode::Ln(inner) => {
            let (a, c, m) = as_shifted_monomial(arena, inner, var)?;
            Some(Known::Ln { a, c, m })
        }
        ExprNode::Pow(base, exp) => {
            if walk::contains(arena, exp, var) {
                return None;
            }
            let (a, c, m) = as_shifted_monomial(arena, base, var)?;
            Some(Known::Binomial {
                alpha: exp,
                a,
                c,
                m,
            })
        }
        _ => None,
    }
}

impl Known {
    fn coefficient(&self, arena: &mut Arena, k: usize) -> ExprId {
        match self {
            Known::Elementary { kind, c, m } => {
                if !k.is_multiple_of(*m) {
                    return arena.zero;
                }
                let n = k / m;
                let base = kind.coefficient(arena, n);
                if arena.is_zero_structural(base) {
                    return arena.zero;
                }
                let cn = pow_i(arena, *c, n as i64);
                mul_all(arena, &[base, cn])
            }
            Known::Ln { a, c, m } => {
                if k == 0 {
                    let l = arena.ln(*a);
                    return eval::eval(arena, l);
                }
                if !k.is_multiple_of(*m) {
                    return arena.zero;
                }
                let n = k / m;
                let coeff = rat_expr(arena, FnKind::Ln1p.rational_coefficient(n));
                let ratio = arena.div(*c, *a);
                let rn = pow_i(arena, ratio, n as i64);
                mul_all(arena, &[coeff, rn])
            }
            Known::Binomial { alpha, a, c, m } => {
                if !k.is_multiple_of(*m) {
                    return arena.zero;
                }
                let n = k / m;
                let bin = gen_binomial(arena, *alpha, n);
                let ratio = arena.div(*c, *a);
                let rn = pow_i(arena, ratio, n as i64);
                let aa = arena.pow(*a, *alpha);
                mul_all(arena, &[aa, bin, rn])
            }
        }
    }

    /// Closed-form general term in the symbolic index `k` (only for `m = 1`).
    fn general_term(&self, arena: &mut Arena, k: ExprId) -> Option<ExprId> {
        let one = arena.one;
        let two = arena.int(2);
        let m1 = arena.neg_one;
        let pi = arena.pi;
        // sin(kπ/2): 0, 1, 0, −1, …   cos(kπ/2): 1, 0, −1, 0, …
        let k_pi_2 = arena.mul(&[k, pi]);
        let k_pi_2 = arena.div(k_pi_2, two);
        let sin_k = arena.sin(k_pi_2);
        let cos_k = arena.cos(k_pi_2);
        let neg1_k = arena.pow(m1, k);
        // (1 − (−1)^k)/2 : 1 for odd k;   (1 + (−1)^k)/2 : 1 for even k
        let odd_ind = {
            let d = arena.sub(one, neg1_k);
            arena.div(d, two)
        };
        let even_ind = {
            let d = arena.add(&[one, neg1_k]);
            arena.div(d, two)
        };
        let kf = arena.factorial(k);
        let inv_kf = arena.div(one, kf);
        let inv_k = arena.div(one, k);
        let ge1 = arena.ge(k, one);
        let zero = arena.zero;
        let lt1 = arena.gt(one, k);
        let piecewise_ge1 =
            |arena: &mut Arena, v: ExprId| arena.piecewise(&[(v, ge1), (zero, lt1)]);
        match self {
            Known::Elementary { kind, c, m } => {
                if *m != 1 {
                    return None;
                }
                let ck = arena.pow(*c, k);
                let term = match kind {
                    FnKind::Exp => inv_kf,
                    FnKind::Sin => arena.mul(&[sin_k, inv_kf]),
                    FnKind::Cos => arena.mul(&[cos_k, inv_kf]),
                    FnKind::Sinh => arena.mul(&[odd_ind, inv_kf]),
                    FnKind::Cosh => arena.mul(&[even_ind, inv_kf]),
                    FnKind::Atan => {
                        let v = arena.mul(&[sin_k, inv_k]);
                        piecewise_ge1(arena, v)
                    }
                    FnKind::Atanh => {
                        let v = arena.mul(&[odd_ind, inv_k]);
                        piecewise_ge1(arena, v)
                    }
                    FnKind::Asin | FnKind::Asinh => {
                        // odd k = 2j+1: C(2j, j) / (4^j (2j+1)),  j = (k−1)/2
                        let km1 = arena.sub(k, one);
                        let j = arena.div(km1, two);
                        let bin = arena.binomial(km1, j);
                        let four_j = arena.pow(two, km1); // 4^j = 2^(k−1)
                        let den = arena.mul(&[four_j, k]);
                        let v = arena.div(bin, den);
                        let sel = if *kind == FnKind::Asin {
                            odd_ind
                        } else {
                            sin_k
                        };
                        let v = arena.mul(&[sel, v]);
                        piecewise_ge1(arena, v)
                    }
                    FnKind::Tan | FnKind::Tanh => {
                        // odd k: 2^(k+1)(2^(k+1) − 1) B_{k+1} / (k+1)!  (× (−1)^((k−1)/2) for tan)
                        let kp1 = arena.add(&[k, one]);
                        let p = arena.pow(two, kp1);
                        let pm1 = arena.sub(p, one);
                        let b = arena.bernoulli_number(kp1);
                        let f = arena.factorial(kp1);
                        let num = arena.mul(&[p, pm1, b]);
                        let v = arena.div(num, f);
                        let sel = if *kind == FnKind::Tan { sin_k } else { odd_ind };
                        arena.mul(&[sel, v])
                    }
                    FnKind::Erf => {
                        // odd k = 2j+1: (2/√π)(−1)^j / (j! (2j+1))
                        let km1 = arena.sub(k, one);
                        let j = arena.div(km1, two);
                        let jf = arena.factorial(j);
                        let den = arena.mul(&[jf, k]);
                        let sp = arena.sqrt(pi);
                        let pref = arena.div(two, sp);
                        let v = arena.div(pref, den);
                        let v = arena.mul(&[sin_k, v]);
                        piecewise_ge1(arena, v)
                    }
                    FnKind::LambertW => {
                        // (−k)^(k−1)/k!
                        let nk = arena.neg(k);
                        let km1 = arena.sub(k, one);
                        let p = arena.pow(nk, km1);
                        let v = arena.mul(&[p, inv_kf]);
                        piecewise_ge1(arena, v)
                    }
                    FnKind::Ln1p => return None,
                };
                let r = arena.mul(&[term, ck]);
                Some(eval::eval(arena, r))
            }
            Known::Ln { a, c, m } => {
                if *m != 1 {
                    return None;
                }
                // k ≥ 1: −(−c/a)^k / k ;  k = 0: ln a
                let ratio = arena.div(*c, *a);
                let nr = arena.neg(ratio);
                let p = arena.pow(nr, k);
                let v = arena.mul(&[p, inv_k]);
                let v = arena.neg(v);
                let la = arena.ln(*a);
                let eq0 = arena.eq_(k, zero);
                let r = arena.piecewise(&[(la, eq0), (v, ge1)]);
                Some(eval::eval(arena, r))
            }
            Known::Binomial { alpha, a, c, m } => {
                if *m != 1 {
                    return None;
                }
                // C(α, k) = (−1)^k (−α)_k / k!  (valid for every α)
                let neg_alpha = arena.neg(*alpha);
                let rf = arena.rising_factorial(neg_alpha, k);
                let bin = arena.mul(&[neg1_k, rf, inv_kf]);
                let ratio = arena.div(*c, *a);
                let rk = arena.pow(ratio, k);
                let aa = arena.pow(*a, *alpha);
                let r = arena.mul(&[aa, bin, rk]);
                Some(eval::eval(arena, r))
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Construction
// ═══════════════════════════════════════════════════════════════════════════

/// Build the formal power series of `expr` about `point` in `var`
/// (crate-internal entry used by `Ex::fps`).
///
/// Tried in order, on `f(x) = expr(x + point)`:
/// 1. a Taylor or Laurent series in `x` (closed form, polynomial, or the
///    meromorphic expansion engine);
/// 2. `f = x^ρ · h` with `ρ` a non-integer rational or symbolic exponent
///    (`√x·eˣ`, `x^a·eˣ`) and `h` as in 1;
/// 3. a Puiseux series: `x^(p/q)` in `h` becomes `t^(p·d/q)` and `x` becomes
///    `t^d` for `t = x^(1/d)` (an identity for principal branches, as
///    `arg t = arg(x)/d`), and `h(t^d)` is expanded as in 1 (`exp(√x)`,
///    `√(1 + √x)`); an even or odd `h(t^d)` drops to `d/2` (`cos √x` is a
///    power series in `x`, `sin √x = √x·(1 − x/6 + …)`);
/// 4. a sum, term by term, when the terms' exponents differ by rationals
///    (`x^a + x^(a+1)·eˣ`).
///
/// Otherwise the series is refused (never `nan` coefficients).
pub(crate) fn fps(ctx: &Context, expr: ExprId, var: ExprId, point: ExprId) -> FormalPowerSeries {
    let (whole, parts, zero) = {
        let mut inner = ctx.inner.write();
        let arena = &mut inner.arena;
        let zero = arena.zero;
        let shifted = if arena.is_zero_structural(point) {
            expr
        } else {
            let v = arena.add(&[var, point]);
            crate::transforms::subs::subs(arena, expr, var, v)
        };
        match build_term(ctx, arena, shifted, var, point) {
            Some(s) => (Some(s), Vec::new(), zero),
            None => {
                let mut parts = Vec::new();
                if let ExprNode::Add(args) = arena.node(shifted).clone() {
                    for t in args {
                        match build_term(ctx, arena, t, var, point) {
                            Some(s) => parts.push(s),
                            None => {
                                parts.clear();
                                break;
                            }
                        }
                    }
                }
                (None, parts, zero)
            }
        }
    };
    if let Some(s) = whole {
        return s;
    }
    let mut parts = parts.into_iter();
    if let Some(mut acc) = parts.next() {
        let mut ok = true;
        for p in parts {
            match acc.add(&p) {
                Ok(s) => acc = s,
                Err(_) => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            return acc;
        }
    }
    FormalPowerSeries::with_source(
        ctx,
        var,
        point,
        zero,
        Source::Refused {
            expr,
            reason: "no Laurent or Puiseux expansion exists at this point".into(),
        },
    )
}

/// Steps 1–3 of [`fps`] for one term `f` (already shifted to 0).
fn build_term(
    ctx: &Context,
    arena: &mut Arena,
    f: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<FormalPowerSeries> {
    let zero = arena.zero;
    let make = |offset: ExprId, ram: u32, (source, valuation): (Source, i64)| {
        let mut s = FormalPowerSeries::with_source(ctx, var, point, zero, source);
        s.offset = offset;
        s.ram = ram;
        s.valuation = valuation;
        s
    };
    if let Some(o) = ordinary(arena, f, var) {
        return Some(make(zero, 1, o));
    }
    let (offset, h) = split_monomial(arena, f, var);
    if h != f
        && let Some(o) = ordinary(arena, h, var)
    {
        return Some(make(offset, 1, o));
    }
    // Step 3: ramification by the lcm of the denominators of `x^(p/q)`.
    let mut d = 1u32;
    let mut pows: Vec<(ExprId, ExprId, Q)> = Vec::new();
    for id in walk::post_order_ids(arena, h) {
        if let ExprNode::Pow(b, e) = *arena.node(id)
            && let Some(c) = positive_multiple_of(arena, b, var)
            && let Some(q) = arena.as_num(e)
            && !q.is_integer()
        {
            let den = u32::try_from(q.denom()).ok()?;
            d = lcm_u32(d, den)?;
            let q = q.clone();
            let ce = arena.pow(c, e);
            pows.push((id, eval::eval(arena, ce), q));
        }
    }
    if d == 1 {
        return None;
    }
    let t = arena.symbol(RAMIFIED_PLACEHOLDER);
    if walk::contains(arena, h, t) {
        return None;
    }
    let d_q = Q::from_integer(BigInt::from(d));
    let mut g = h;
    for (node, ce, q) in pows {
        let m = q_to_i64(&(q * &d_q))?;
        let tm = pow_i(arena, t, m);
        let tm = mul_all(arena, &[ce, tm]);
        g = crate::transforms::subs::subs(arena, g, node, tm);
    }
    let td = pow_i(arena, t, i64::from(d));
    g = crate::transforms::subs::subs(arena, g, var, td);
    g = eval::eval(arena, g);
    if walk::contains(arena, g, var) {
        return None;
    }
    let s = make(offset, d, ordinary(arena, g, t)?);
    if !d.is_multiple_of(2) {
        return Some(s);
    }
    // `g(−t) = ±g(t)`: only even (odd) powers of `t` occur.
    let mt = arena.neg(t);
    let gm = crate::transforms::subs::subs(arena, g, t, mt);
    let gm = eval::eval(arena, gm);
    let ng = arena.neg(g);
    let ng = eval::eval(arena, ng);
    if gm == g {
        Some(s.reindexed(arena, offset, d / 2, 2, 1, 0))
    } else if gm == ng {
        let step = rat_expr(arena, Q::new(BigInt::one(), BigInt::from(d)));
        let o = arena.add(&[offset, step]);
        let o = eval::eval(arena, o);
        Some(s.reindexed(arena, o, d / 2, 2, 1, -1))
    } else {
        Some(s)
    }
}

/// A Taylor or Laurent series of `f` in `svar` about 0: a closed form, a
/// polynomial, or the meromorphic expansion engine.  Returns the source and
/// its valuation.
fn ordinary(arena: &mut Arena, f: ExprId, svar: ExprId) -> Option<(Source, i64)> {
    if let Some(k) = detect_known(arena, f, svar) {
        return Some((Source::Known(k), 0));
    }
    if let Some(terms) = crate::calculus::summation::sym_poly_in(arena, f, svar) {
        let deg = terms.iter().map(|(p, _)| *p).max().unwrap_or(0);
        let mut coeffs = vec![arena.zero; deg + 1];
        for (p, c) in terms {
            coeffs[p] = c;
        }
        return Some((Source::Polynomial(coeffs), 0));
    }
    // A formal power series stands for `f`, so `f` must be meromorphic at
    // the point: the real asymptotic expansion of `series` would give
    // `exp(−1/x²)` the coefficients 0, 0, 0, … (see the series contract).
    let ts = expand_meromorphic(arena, f, svar, 8).ok()?;
    let valuation = ts.shift();
    Some((
        Source::Engine {
            expr: f,
            svar,
            cache: Mutex::new(Some(ts)),
        },
        valuation,
    ))
}

/// `Some(c)` when `b = c·var` with `c` a positive rational (`c = 1` for
/// `var`): then `(c·x)^e = c^e·x^e` for principal powers.
fn positive_multiple_of(arena: &Arena, b: ExprId, var: ExprId) -> Option<ExprId> {
    if b == var {
        return Some(arena.one);
    }
    if let ExprNode::Mul(args) = arena.node(b)
        && args.len() == 2
    {
        for (c, v) in [(args[0], args[1]), (args[1], args[0])] {
            if v == var && arena.as_num(c).is_some_and(|q| *q > Q::zero()) {
                return Some(c);
            }
        }
    }
    None
}

/// `f = var^ρ · h` for the factors `(c·var)^e` (`e` free of `var`, `c > 0`) of a product,
/// when `ρ = Σ e` is not an integer; `(0, f)` otherwise.
fn split_monomial(arena: &mut Arena, f: ExprId, var: ExprId) -> (ExprId, ExprId) {
    let zero = arena.zero;
    let factors: Vec<ExprId> = match arena.node(f) {
        ExprNode::Mul(args) => args.to_vec(),
        _ => vec![f],
    };
    let mut exps = Vec::new();
    let mut rest = Vec::new();
    for t in factors {
        match *arena.node(t) {
            ExprNode::Pow(b, e)
                if !walk::contains(arena, e, var)
                    && positive_multiple_of(arena, b, var).is_some() =>
            {
                exps.push(e);
                if let Some(c) = positive_multiple_of(arena, b, var)
                    && c != arena.one
                {
                    let ce = arena.pow(c, e);
                    rest.push(eval::eval(arena, ce));
                }
            }
            // `(x^e)^n = x^(e·n)` for an integer `n` (`1/x^a`).
            ExprNode::Pow(b, n) if as_integer(arena, n).is_some() => match *arena.node(b) {
                ExprNode::Pow(bb, e) if bb == var && !walk::contains(arena, e, var) => {
                    let en = arena.mul(&[e, n]);
                    exps.push(eval::eval(arena, en));
                }
                _ => rest.push(t),
            },
            _ => rest.push(t),
        }
    }
    if exps.is_empty() {
        return (zero, f);
    }
    let rho = add_all(arena, &exps);
    if let Some(m) = as_integer(arena, rho) {
        // An integer total (`x^a/x^a`, `x^(1/2)·x^(1/2)` left unmerged) stays
        // in `h`, as a plain power of `var`.
        rest.push(pow_i(arena, var, m));
        return (zero, mul_all(arena, &rest));
    }
    (rho, mul_all(arena, &rest))
}

impl FormalPowerSeries {
    fn with_source(
        ctx: &Context,
        var: ExprId,
        point: ExprId,
        zero: ExprId,
        source: Source,
    ) -> Self {
        FormalPowerSeries {
            ctx: ctx.clone(),
            var,
            point,
            offset: zero,
            ram: 1,
            valuation: 0,
            log_terms: Vec::new(),
            source,
            memo: Mutex::new(Vec::new()),
            zero,
        }
    }

    /// A polynomial series `Σ coeffs[k] (var − point)^k` from explicit
    /// coefficients (all in the same context as `var`).
    ///
    /// # Panics
    ///
    /// Panics if the expressions come from different contexts.
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::formal_series::FormalPowerSeries;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let p = FormalPowerSeries::from_coefficients(&x, &ctx.int(0), &[ctx.int(1), ctx.int(2)]);
    /// assert_eq!(p.truncate(5).to_string(), "2*x + 1");
    /// ```
    #[must_use]
    pub fn from_coefficients(var: &Ex, point: &Ex, coeffs: &[Ex]) -> Self {
        let point_id = var.checked_id(point);
        let ids: Vec<ExprId> = coeffs.iter().map(|c| var.checked_id(c)).collect();
        let ctx = var.context();
        let zero = ctx.inner.read().arena.zero;
        Self::with_source(&ctx, var.raw_id(), point_id, zero, Source::Polynomial(ids))
    }

    /// A derived series in the frame (`offset`, `ram`) of `self`.
    fn derived(&self, source: Source, valuation: i64) -> Self {
        let mut s = Self::with_source(&self.ctx, self.var, self.point, self.zero, source);
        s.offset = self.offset;
        s.ram = self.ram;
        s.valuation = valuation;
        s
    }

    /// A refused series standing for `expr` (from an operation on a refused
    /// series).
    fn refused_like(&self, expr: ExprId, reason: &str) -> Self {
        let source = Source::Refused {
            expr,
            reason: reason.to_string(),
        };
        Self::with_source(&self.ctx, self.var, self.point, self.zero, source)
    }

    /// The same function under the frame `(offset, ram)`, coefficient `k`
    /// being `self`'s coefficient `(k·mul − shift)/div`.  An integer offset
    /// is folded into the indices (offset 0).
    fn reindexed(
        self,
        arena: &Arena,
        offset: ExprId,
        ram: u32,
        mul: u32,
        div: u32,
        shift: i64,
    ) -> Self {
        let (offset, shift) = match as_integer(arena, offset) {
            Some(m) if m != 0 => (
                self.zero,
                shift.saturating_add(m.saturating_mul(i64::from(ram) * i64::from(mul))),
            ),
            _ => (offset, shift),
        };
        if mul == 1 && div == 1 && shift == 0 && offset == self.offset && ram == self.ram {
            return self;
        }
        let num = self
            .valuation
            .saturating_mul(i64::from(div))
            .saturating_add(shift);
        let m = i64::from(mul);
        let valuation = num.div_euclid(m) + i64::from(num.rem_euclid(m) != 0);
        let (ctx, var, point, zero) = (self.ctx.clone(), self.var, self.point, self.zero);
        let log_terms = self.log_terms.clone();
        let source = Source::Reindex {
            inner: Box::new(self),
            mul,
            div,
            shift,
        };
        let mut s = Self::with_source(&ctx, var, point, zero, source);
        s.offset = offset;
        s.ram = ram;
        s.valuation = valuation;
        s.log_terms = log_terms;
        s
    }

    /// `self` in the frame `(offset, ram)`: `ram` a multiple of `self.ram`
    /// and the offsets differing by a multiple of `1/ram`.
    fn aligned(&self, offset: ExprId, ram: u32) -> Option<Self> {
        if !ram.is_multiple_of(self.ram) {
            return None;
        }
        let shift = {
            let mut inner = self.ctx.inner.write();
            let arena = &mut inner.arena;
            let n = arena.neg(offset);
            let delta = add_all(arena, &[self.offset, n]);
            let q = arena.as_num(delta)?.clone();
            q_to_i64(&(q * Q::from_integer(BigInt::from(ram))))?
        };
        // Clone (memo locks) without holding the context lock: `coeff`
        // takes the memo lock first.
        let s = self.clone();
        let inner = self.ctx.inner.read();
        Some(s.reindexed(&inner.arena, offset, ram, 1, ram / self.ram, shift))
    }

    /// A frame `(offset, ram)` holding both `self` and `other`, when their
    /// offsets differ by a rational number.
    fn common_frame(&self, other: &Self) -> Option<(ExprId, u32)> {
        let ram = lcm_u32(self.ram, other.ram)?;
        if self.offset == other.offset {
            return Some((self.offset, ram));
        }
        let mut inner = self.ctx.inner.write();
        let arena = &mut inner.arena;
        let n = arena.neg(self.offset);
        let delta = add_all(arena, &[other.offset, n]);
        let dq = arena.as_num(delta)?.clone();
        let den = |q: &Q| u32::try_from(q.denom()).ok();
        if let (Some(a), Some(b)) = (
            arena.as_num(self.offset).cloned(),
            arena.as_num(other.offset).cloned(),
        ) {
            let r = lcm_u32(lcm_u32(ram, den(&a)?)?, den(&b)?)?;
            return Some((arena.zero, r));
        }
        let r = lcm_u32(ram, den(&dq)?)?;
        let offset = if dq >= Q::zero() {
            self.offset
        } else {
            other.offset
        };
        Some((offset, r))
    }

    /// `self` with offset 0 (a rational offset folded into the indices).
    fn with_zero_offset(&self) -> Option<Self> {
        if self.offset == self.zero {
            return Some(self.clone());
        }
        let den = {
            let inner = self.ctx.inner.read();
            let q = inner.arena.as_num(self.offset)?;
            u32::try_from(q.denom()).ok()?
        };
        self.aligned(self.zero, lcm_u32(self.ram, den)?)
    }

    fn ensure_usable(&self, op: &'static str, logs_ok: bool) -> Result<(), SymplexError> {
        if let Source::Refused { reason, .. } = &self.source {
            return Err(SymplexError::ComputationFailed {
                operation: op,
                reason: format!("the series does not exist: {reason}"),
            });
        }
        if !logs_ok && !self.log_terms.is_empty() {
            return Err(SymplexError::InvalidArgument {
                operation: op,
                reason: "series has a logarithmic term".into(),
            });
        }
        Ok(())
    }

    fn is_zero_coeff(&self, c: Coeff) -> bool {
        if !c.exact {
            return false;
        }
        let mut inner = self.ctx.inner.write();
        is_zero_expr(&mut inner.arena, c.id)
    }

    fn wrap(&self, id: ExprId) -> Ex {
        Ex::from_raw_parts(self.ctx.id, Arc::clone(&self.ctx.inner), id)
    }

    /// The expansion variable.
    #[must_use]
    pub fn variable(&self) -> Ex {
        self.wrap(self.var)
    }

    /// The expansion point.
    #[must_use]
    pub fn point(&self) -> Ex {
        self.wrap(self.point)
    }

    /// The [`Context`] this series lives in.
    #[must_use]
    pub fn context(&self) -> Context {
        self.ctx.clone()
    }

    /// The exponent offset `ρ`: the series is
    /// `(x − a)^ρ · Σ_k a_k (x − a)^(k/d)`.  Zero for a Taylor or Laurent
    /// series; a non-integer rational or a symbolic exponent otherwise.
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let a = ctx.symbol("a");
    /// // x^a·eˣ = x^a · Σ x^k/k!
    /// let s = (&x.pow(&a) * &x.exp()).fps_maclaurin(&x);
    /// assert_eq!(s.offset().to_string(), "a");
    /// assert_eq!(s.coefficient(2).to_string(), "1/2");
    /// assert_eq!(s.exponent(2).to_string(), "a + 2");
    /// ```
    #[must_use]
    pub fn offset(&self) -> Ex {
        self.wrap(self.offset)
    }

    /// The ramification index `d ≥ 1`: coefficient `k` multiplies
    /// `(x − a)^(ρ + k/d)`.  `1` unless the series is in fractional powers
    /// beyond the offset (`exp(√x) = Σ x^(k/2)/k!` has `d = 2`).
    #[must_use]
    pub fn ramification(&self) -> u32 {
        self.ram
    }

    /// The exponent `ρ + k/d` of `x − a` that [`coefficient(k)`](Self::coefficient)
    /// multiplies (`k` for a Taylor series).
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// // exp(√x) = Σ x^(k/2)/k!
    /// let s = x.sqrt().exp().fps_maclaurin(&x);
    /// assert_eq!(s.ramification(), 2);
    /// assert_eq!(s.exponent(3).to_string(), "3/2");
    /// assert_eq!(s.coefficient(3).to_string(), "1/6");
    /// ```
    #[must_use]
    pub fn exponent(&self, k: usize) -> Ex {
        let id = {
            let mut inner = self.ctx.inner.write();
            self.exponent_id(&mut inner.arena, index(k))
        };
        self.wrap(id)
    }

    fn exponent_id(&self, arena: &mut Arena, k: i64) -> ExprId {
        let step = rat_expr(arena, Q::new(BigInt::from(k), BigInt::from(self.ram)));
        add_all(arena, &[self.offset, step])
    }

    /// The index of the integer exponent `e`, when the offset is rational.
    fn exponent_index_in(&self, arena: &Arena, e: i64) -> Option<i64> {
        let rho = arena.as_num(self.offset)?;
        q_to_i64(
            &((Q::from_integer(BigInt::from(e)) - rho) * Q::from_integer(BigInt::from(self.ram))),
        )
    }

    /// Whether `f` has no Laurent or Puiseux expansion at the point (`ln x`,
    /// `exp(1/x)` or `|x|` at 0).  The coefficients of a refused series are
    /// unevaluated stand-ins (`Subs(Derivative(…), x, a)`), never `nan`;
    /// [`truncate`](Self::truncate) is a formal `Series` node, and the
    /// fallible accessors return `Err`.
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let s = x.ln().fps_maclaurin(&x);
    /// assert!(s.is_refused());
    /// assert!(s.try_coefficient(0).is_err());
    /// assert!(!x.sqrt().fps_maclaurin(&x).is_refused());
    /// ```
    #[must_use]
    pub fn is_refused(&self) -> bool {
        matches!(self.source, Source::Refused { .. })
    }

    /// Check that `other` is compatible (same context, variable, point).
    fn check_compatible(&self, other: &Self, op: &'static str) -> Result<(), SymplexError> {
        // Cross-context is a logic error, like for `Ex` (panics with the same message).
        let _ = self.variable().checked_id(&other.variable());
        if self.var != other.var || self.point != other.point {
            return Err(SymplexError::InvalidArgument {
                operation: op,
                reason: "series must share the expansion variable and point".into(),
            });
        }
        Ok(())
    }

    // ── Coefficients ───────────────────────────────────────────────

    /// The `k`-th coefficient `a_k` as an exact expression: the coefficient
    /// of `(x − a)^k` for a Taylor or Laurent series, of
    /// `(x − a)^(ρ + k/d)` in general (see [`exponent`](Self::exponent)).
    ///
    /// For series with a pole at the expansion point this is the Laurent
    /// coefficient of `(x − a)^k` for `k ≥ 0`; use
    /// [`truncate`](Self::truncate) to see the negative powers.  Where no
    /// coefficient exists (a [refused](Self::is_refused) series) it is an
    /// unevaluated stand-in; [`try_coefficient`](Self::try_coefficient)
    /// returns `Err` instead.  A series computes the coefficients up to 2¹⁴
    /// indices past its first (the earlier ones are needed for the later):
    /// beyond, a polynomial's own coefficient, the stand-in `Res_{x=a} f·(x −
    /// a)^−(k+1)` for a series the expansion engine computes or that is
    /// refused, `nan` for the others (a closed form, arithmetic of series),
    /// and `Err` from `try_coefficient` except for a polynomial.
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let s = x.sin().fps_maclaurin(&x);
    /// assert_eq!(s.coefficient(3).to_string(), "-1/6");
    /// assert_eq!(s.coefficient(4).to_string(), "0");
    /// ```
    #[must_use]
    pub fn coefficient(&self, k: usize) -> Ex {
        self.wrap(self.coeff(index(k)).id)
    }

    /// Like [`coefficient`](Self::coefficient), but `Err` when the series is
    /// refused or the coefficient could not be computed.
    pub fn try_coefficient(&self, k: usize) -> Result<Ex, SymplexError> {
        self.ensure_usable("fps_coefficient", true)?;
        let c = self.coeff(index(k));
        if c.exact {
            Ok(self.wrap(c.id))
        } else {
            Err(SymplexError::ComputationFailed {
                operation: "fps_coefficient",
                reason: "the expansion failed at this order".into(),
            })
        }
    }

    /// The `k`-th coefficient as an exact rational, or `None` if it is not
    /// a rational number (e.g. `ln 2`, `2/√π`, or a symbolic parameter).
    #[must_use]
    pub fn coefficient_rational(&self, k: usize) -> Option<Q> {
        let c = self.coeff(index(k));
        if !c.exact {
            return None;
        }
        let inner = self.ctx.inner.read();
        inner.arena.as_num(c.id).cloned()
    }

    /// The first `n` coefficients `a_0 … a_{n−1}`.
    #[must_use]
    pub fn coefficients(&self, n: usize) -> Vec<Ex> {
        (0..n).map(|k| self.coefficient(k)).collect()
    }

    /// Coefficient of index `k` (memoised; computed sequentially from the
    /// valuation — recurrences need the earlier terms).
    fn coeff(&self, k: i64) -> Coeff {
        let zero = Coeff {
            id: self.zero,
            exact: true,
        };
        let Ok(idx) = usize::try_from(k.saturating_sub(self.valuation)) else {
            return zero;
        };
        let Some(idx) = cast::bounded_len(idx, MAX_COEFFICIENTS) else {
            return self.beyond_cap(k);
        };
        let mut memo = lock(&self.memo);
        while memo.len() <= idx {
            let next = self.valuation + memo.len() as i64;
            let c = self.compute(next, &memo);
            memo.push(c);
        }
        memo[idx]
    }

    /// The coefficient of index `k` more than [`MAX_COEFFICIENTS`] past the
    /// valuation, which is not computed: a polynomial's own entry (exact), a
    /// [`residue_standin`] for a series expanded by the engine or refused,
    /// `nan` for the others (never exact).
    fn beyond_cap(&self, k: i64) -> Coeff {
        let mut inner = self.ctx.inner.write();
        let arena = &mut inner.arena;
        let id = match &self.source {
            Source::Polynomial(c) => {
                let id = usize::try_from(k)
                    .ok()
                    .and_then(|k| c.get(k).copied())
                    .unwrap_or(self.zero);
                return Coeff { id, exact: true };
            }
            Source::Refused { expr, .. } => residue_standin(arena, *expr, self.var, self.point, k),
            Source::Engine { expr, svar, .. } => residue_standin(arena, *expr, *svar, self.zero, k),
            _ => arena.nan,
        };
        Coeff { id, exact: false }
    }

    /// Is the range `lo..=hi` longer than [`MAX_COEFFICIENTS`]?
    fn beyond_cap_range(lo: i64, hi: i64) -> bool {
        hi >= lo && cast::bounded_len(hi.abs_diff(lo), MAX_COEFFICIENTS).is_none()
    }

    /// Coefficients of indices `lo..=hi`.
    fn coeffs(&self, lo: i64, hi: i64) -> Vec<Coeff> {
        if hi < lo {
            return Vec::new();
        }
        let _ = self.coeff(hi);
        (lo..=hi).map(|k| self.coeff(k)).collect()
    }

    fn compute(&self, k: i64, prev: &[Coeff]) -> Coeff {
        let exact = |id: ExprId| Coeff { id, exact: true };
        let zero = exact(self.zero);
        match &self.source {
            Source::Known(known) => {
                let Ok(k) = usize::try_from(k) else {
                    return zero;
                };
                let mut inner = self.ctx.inner.write();
                exact(known.coefficient(&mut inner.arena, k))
            }
            Source::Engine { expr, svar, cache } => {
                let mut inner = self.ctx.inner.write();
                let arena = &mut inner.arena;
                let mut guard = lock(cache);
                let need = k + 1;
                if guard.as_ref().is_none_or(|ts| ts.known() < need) {
                    let order =
                        need.max(guard.as_ref().map_or(8, |ts| ts.known().saturating_mul(2)));
                    let ts = expand_meromorphic(arena, *expr, *svar, order).or_else(|e| {
                        if order > need {
                            expand_meromorphic(arena, *expr, *svar, need)
                        } else {
                            Err(e)
                        }
                    });
                    // On failure the earlier expansion stays valid.
                    if let Ok(ts) = ts {
                        *guard = Some(ts);
                    }
                }
                if let Some(ts) = guard.as_ref()
                    && ts.known() >= need
                {
                    let c = ts.coefficient(arena, k);
                    if !contains_singular(arena, c) {
                        return exact(c);
                    }
                }
                let id = taylor_standin(arena, *expr, *svar, self.zero, k);
                Coeff { id, exact: false }
            }
            Source::Polynomial(c) => usize::try_from(k)
                .ok()
                .and_then(|k| c.get(k).copied())
                .map_or(zero, exact),
            Source::Refused { expr, .. } => {
                if k < 0 {
                    return zero;
                }
                let mut inner = self.ctx.inner.write();
                let id = taylor_standin(&mut inner.arena, *expr, self.var, self.point, k);
                Coeff { id, exact: false }
            }
            Source::Add(a, b) | Source::Sub(a, b) => {
                let x = a.coeff(k);
                let y = b.coeff(k);
                let mut inner = self.ctx.inner.write();
                let arena = &mut inner.arena;
                let y_id = if matches!(self.source, Source::Sub(..)) {
                    arena.neg(y.id)
                } else {
                    y.id
                };
                Coeff {
                    id: add_all(arena, &[x.id, y_id]),
                    exact: x.exact && y.exact,
                }
            }
            Source::Mul(a, b) => {
                let (va, vb) = (a.valuation, b.valuation);
                let xs = a.coeffs(va, k - vb);
                let ys = b.coeffs(vb, k - va);
                let mut inner = self.ctx.inner.write();
                let arena = &mut inner.arena;
                let mut terms = Vec::new();
                let mut ex = true;
                let top = ys.len();
                for (i, x) in xs.iter().enumerate() {
                    let y = ys[top - 1 - i];
                    if (x.exact && arena.is_zero_structural(x.id))
                        || (y.exact && arena.is_zero_structural(y.id))
                    {
                        continue;
                    }
                    ex &= x.exact && y.exact;
                    terms.push(arena.mul(&[x.id, y.id]));
                }
                Coeff {
                    id: add_all(arena, &terms),
                    exact: ex,
                }
            }
            Source::Scale(a, c) => {
                let x = a.coeff(k);
                let mut inner = self.ctx.inner.write();
                Coeff {
                    id: mul_all(&mut inner.arena, &[*c, x.id]),
                    exact: x.exact,
                }
            }
            Source::Derivative(a) => {
                // d/dx (x−a)^(ρ+j/d) = (ρ + j/d)·(x−a)^(ρ+j/d−1),  j = k + d
                let d = i64::from(a.ram);
                let x = a.coeff(k + d);
                let mut inner = self.ctx.inner.write();
                let arena = &mut inner.arena;
                let e = a.exponent_id(arena, k + d);
                let mut terms = vec![arena.mul(&[e, x.id])];
                for &(p, r) in &a.log_terms {
                    // d/dx r(x−a)^p ln(x−a) contributes r(x−a)^(p−1)
                    if a.exponent_index_in(arena, p - 1) == Some(k) {
                        terms.push(r);
                    }
                }
                Coeff {
                    id: add_all(arena, &terms),
                    exact: x.exact,
                }
            }
            Source::Integral(a) => {
                // ∫ (x−a)^e = (x−a)^(e+1)/(e+1),  e = ρ + (k−d)/d
                let d = i64::from(a.ram);
                let x = a.coeff(k - d);
                let mut inner = self.ctx.inner.write();
                let arena = &mut inner.arena;
                let e = a.exponent_id(arena, k - d);
                let one = arena.one;
                let e1 = add_all(arena, &[e, one]);
                let mut terms = Vec::new();
                // e = −1: the constant of integration is 0 and the residue is
                // a logarithmic term (see `integral`).
                if !is_zero_expr(arena, e1) && !arena.is_zero_structural(x.id) {
                    let m1 = arena.neg_one;
                    let inv = arena.pow(e1, m1);
                    terms.push(arena.mul(&[x.id, inv]));
                }
                for &(p, r) in &a.log_terms {
                    // ∫ r(x−a)^p ln(x−a) = r/(p+1)·(x−a)^(p+1) ln(x−a) − r/(p+1)²·(x−a)^(p+1)
                    if a.exponent_index_in(arena, p + 1) == Some(k) {
                        let c = rat_expr(
                            arena,
                            Q::new(BigInt::from(-1), BigInt::from((p + 1) * (p + 1))),
                        );
                        terms.push(arena.mul(&[c, r]));
                    }
                }
                Coeff {
                    id: add_all(arena, &terms),
                    exact: x.exact,
                }
            }
            Source::Inverse(a, l) => {
                // b_j = (δ_j0 − Σ_{i=1}^{j} a_{l+i} b_{j−i}) / a_l,  j = k + l
                let j = k + l;
                let xs = a.coeffs(*l, l + j);
                let mut inner = self.ctx.inner.write();
                let arena = &mut inner.arena;
                let inv0 = {
                    let m1 = arena.neg_one;
                    let p = arena.pow(xs[0].id, m1);
                    eval::eval(arena, p)
                };
                let mut ex = xs[0].exact;
                if j == 0 {
                    return Coeff {
                        id: inv0,
                        exact: ex,
                    };
                }
                let mut terms = Vec::new();
                for (i, ai) in xs.iter().enumerate().skip(1) {
                    let bj = prev[usize::try_from(j).unwrap_or(0) - i];
                    if arena.is_zero_structural(ai.id) || arena.is_zero_structural(bj.id) {
                        continue;
                    }
                    ex &= ai.exact && bj.exact;
                    terms.push(arena.mul(&[ai.id, bj.id]));
                }
                let s = add_all(arena, &terms);
                let v = arena.mul(&[inv0, s]);
                let v = arena.neg(v);
                Coeff {
                    id: eval::eval(arena, v),
                    exact: ex,
                }
            }
            Source::Compose(f, g) => {
                // Σ_{n≤k} f_n · (g^n)_k with g_j = 0 for j ≤ 0.
                if k < 0 {
                    return zero;
                }
                let Some(ku) = cast::bounded_len(k, MAX_COEFFICIENTS) else {
                    return self.beyond_cap(k);
                };
                let fs = f.coeffs(0, k);
                let gs = g.coeffs(0, k);
                let ex = fs.iter().chain(gs.iter()).all(|c| c.exact);
                let gs: Vec<ExprId> = gs.iter().map(|c| c.id).collect();
                let mut inner = self.ctx.inner.write();
                let arena = &mut inner.arena;
                let n = ku + 1;
                let mut power: Vec<ExprId> = vec![arena.zero; n];
                power[0] = arena.one;
                let mut terms = Vec::new();
                for (i, fi) in fs.iter().enumerate() {
                    if i > 0 {
                        power = poly_mul(arena, &power, &gs, n);
                    }
                    if arena.is_zero_structural(fi.id) || arena.is_zero_structural(power[ku]) {
                        continue;
                    }
                    terms.push(arena.mul(&[fi.id, power[ku]]));
                }
                Coeff {
                    id: add_all(arena, &terms),
                    exact: ex,
                }
            }
            Source::Reversion(f) => {
                // Lagrange inversion: [x^k] g = (1/k) [x^{k−1}] (x/f(x))^k, k ≥ 1.
                if k < 0 {
                    return zero;
                }
                let Some(ku) = cast::bounded_len(k, MAX_COEFFICIENTS) else {
                    return self.beyond_cap(k);
                };
                if ku == 0 {
                    return zero;
                }
                let fs = f.coeffs(0, k);
                let ex = fs.iter().all(|c| c.exact);
                let mut inner = self.ctx.inner.write();
                let arena = &mut inner.arena;
                // h = f/x  (drop f_0 = 0)
                let h: Vec<ExprId> = fs[1..].iter().map(|c| c.id).collect();
                let inv_h = poly_inverse(arena, &h, ku);
                let mut power: Vec<ExprId> = vec![arena.zero; ku];
                power[0] = arena.one;
                for _ in 0..ku {
                    power = poly_mul(arena, &power, &inv_h, ku);
                }
                let inv_k = rat_expr(arena, Q::new(BigInt::one(), BigInt::from(ku)));
                Coeff {
                    id: mul_all(arena, &[inv_k, power[ku - 1]]),
                    exact: ex,
                }
            }
            Source::Reindex {
                inner,
                mul,
                div,
                shift,
            } => {
                let num = k.saturating_mul(i64::from(*mul)).saturating_sub(*shift);
                let dv = i64::from(*div);
                if num.rem_euclid(dv) != 0 {
                    return zero;
                }
                inner.coeff(num.div_euclid(dv))
            }
        }
    }

    // ── Presentation ───────────────────────────────────────────────

    /// Whether a closed-form general term is available (see
    /// [`general_term`](Self::general_term)).
    #[must_use]
    pub fn has_closed_form(&self) -> bool {
        match &self.source {
            Source::Known(_) | Source::Polynomial(_) => true,
            Source::Add(a, b) | Source::Sub(a, b) => a.has_closed_form() && b.has_closed_form(),
            Source::Scale(a, _) => a.has_closed_form(),
            Source::Derivative(a) => a.log_terms.is_empty() && a.has_closed_form(),
            Source::Integral(a) => {
                a.is_taylor_frame() && a.log_terms.is_empty() && a.has_closed_form()
            }
            Source::Reindex { inner, div, .. } => *div == 1 && inner.has_closed_form(),
            _ => false,
        }
    }

    fn is_taylor_frame(&self) -> bool {
        self.offset == self.zero && self.ram == 1
    }

    /// The general coefficient `a_k` as a closed-form expression in the
    /// symbolic index `k`, when known (the coefficient of
    /// `(x − a)^(ρ + k/d)`, see [`exponent`](Self::exponent)).
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let k = ctx.symbol("k");
    /// let e = x.exp().fps_maclaurin(&x);
    /// assert_eq!(e.general_term(&k).unwrap().to_string(), "1/k!");
    /// ```
    #[must_use]
    pub fn general_term(&self, k: &Ex) -> Option<Ex> {
        let k_id = self.variable().checked_id(k);
        let id = self.general_term_id(k_id)?;
        Some(self.wrap(id))
    }

    fn general_term_id(&self, k: ExprId) -> Option<ExprId> {
        match &self.source {
            Source::Known(known) => {
                let mut inner = self.ctx.inner.write();
                known.general_term(&mut inner.arena, k)
            }
            Source::Polynomial(c) => {
                let mut inner = self.ctx.inner.write();
                let arena = &mut inner.arena;
                let mut pairs = Vec::new();
                for (i, &ci) in c.iter().enumerate() {
                    if arena.is_zero_structural(ci) {
                        continue;
                    }
                    let ie = arena.int(i as i64);
                    let cond = arena.eq_(k, ie);
                    pairs.push((ci, cond));
                }
                let deg = arena.int(c.len() as i64);
                let rest = arena.ge(k, deg);
                let zero = arena.zero;
                pairs.push((zero, rest));
                Some(arena.piecewise(&pairs))
            }
            Source::Add(a, b) | Source::Sub(a, b) => {
                let x = a.general_term_id(k)?;
                let y = b.general_term_id(k)?;
                let mut inner = self.ctx.inner.write();
                let y = if matches!(self.source, Source::Sub(..)) {
                    inner.arena.neg(y)
                } else {
                    y
                };
                Some(add_all(&mut inner.arena, &[x, y]))
            }
            Source::Scale(a, c) => {
                let x = a.general_term_id(k)?;
                let mut inner = self.ctx.inner.write();
                Some(mul_all(&mut inner.arena, &[*c, x]))
            }
            Source::Derivative(a) => {
                if !a.log_terms.is_empty() {
                    return None;
                }
                // (ρ + (k+d)/d)·a_{k+d}
                let mut inner = self.ctx.inner.write();
                let d = inner.arena.int(i64::from(a.ram));
                let kpd = inner.arena.add(&[k, d]);
                drop(inner);
                let x = a.general_term_id(kpd)?;
                let mut inner = self.ctx.inner.write();
                let arena = &mut inner.arena;
                let e = if a.ram == 1 { kpd } else { arena.div(kpd, d) };
                let e = arena.add(&[a.offset, e]);
                Some(mul_all(arena, &[e, x]))
            }
            Source::Integral(a) => {
                if !(a.is_taylor_frame() && a.log_terms.is_empty()) {
                    return None;
                }
                let mut inner = self.ctx.inner.write();
                let one = inner.arena.one;
                let km1 = inner.arena.sub(k, one);
                drop(inner);
                let x = a.general_term_id(km1)?;
                let mut inner = self.ctx.inner.write();
                let arena = &mut inner.arena;
                let v = arena.div(x, k);
                let one = arena.one;
                let zero = arena.zero;
                let ge1 = arena.ge(k, one);
                let lt1 = arena.gt(one, k);
                Some(arena.piecewise(&[(v, ge1), (zero, lt1)]))
            }
            Source::Reindex {
                inner: s,
                mul,
                div,
                shift,
            } => {
                if *div != 1 {
                    return None;
                }
                let mut inner = self.ctx.inner.write();
                let arena = &mut inner.arena;
                let m = arena.int(i64::from(*mul));
                let km = arena.mul(&[m, k]);
                let sh = arena.int(-*shift);
                let j = arena.add(&[km, sh]);
                let j = eval::eval(arena, j);
                drop(inner);
                s.general_term_id(j)
            }
            _ => None,
        }
    }

    /// The truncated series: every term of order below `(x − point)^n`,
    /// i.e. `Σ_{k<n} a_k (x − point)^k` for a Taylor series.
    ///
    /// For series with a pole at the point, negative powers down to the
    /// pole order are included; Puiseux series keep the terms whose
    /// exponent (its rational part, for a symbolic offset) is below `n`, as
    /// SymPy's `fps(f).truncate(n)` does.  A [refused](Self::is_refused)
    /// series gives a formal `Series` node.
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let s = x.cos().fps_maclaurin(&x);
    /// assert_eq!(s.truncate(5).to_string(), "1/24*x^4 - 1/2*x^2 + 1");
    /// // √x·eˣ = √x + x^(3/2) + x^(5/2)/2 + …
    /// let p = (&x.sqrt() * &x.exp()).fps_maclaurin(&x);
    /// assert_eq!(p.truncate(3).to_string(), "sqrt(x) + x^(3/2) + 1/2*x^(5/2)");
    /// ```
    #[must_use]
    pub fn truncate(&self, n: usize) -> Ex {
        let id = self.truncate_id(n);
        self.wrap(id)
    }

    /// Like [`truncate`](Self::truncate), but `Err` when the series is
    /// refused or a coefficient could not be computed.
    pub fn try_truncate(&self, n: usize) -> Result<Ex, SymplexError> {
        self.ensure_usable("fps_truncate", true)?;
        let end = self.index_end(n);
        if Self::beyond_cap_range(self.valuation, end - 1)
            || self
                .coeffs(self.valuation, end - 1)
                .iter()
                .any(|c| !c.exact)
        {
            return Err(SymplexError::ComputationFailed {
                operation: "fps_truncate",
                reason: "the expansion failed at this order".into(),
            });
        }
        Ok(self.truncate(n))
    }

    /// One past the largest index whose exponent has rational part `< n`.
    fn index_end(&self, n: usize) -> i64 {
        let inner = self.ctx.inner.read();
        let r = rational_part(&inner.arena, self.offset);
        let x = (Q::from_integer(BigInt::from(n)) - r) * Q::from_integer(BigInt::from(self.ram));
        q_to_i64(&x.ceil())
            .unwrap_or(self.valuation)
            .max(self.valuation)
    }

    fn truncate_id(&self, n: usize) -> ExprId {
        if let Source::Refused { expr, .. } = &self.source {
            let mut inner = self.ctx.inner.write();
            let order = inner.arena.int(i64::try_from(n).unwrap_or(i64::MAX));
            return inner
                .arena
                .intern(ExprNode::Series(*expr, self.var, self.point, order));
        }
        let end = self.index_end(n);
        // More terms than a series computes: not computed (`try_truncate`
        // refuses).
        if Self::beyond_cap_range(self.valuation, end - 1) {
            return self.ctx.inner.read().arena.nan;
        }
        let coeffs = self.coeffs(self.valuation, end - 1);
        let mut inner = self.ctx.inner.write();
        let arena = &mut inner.arena;
        let base = if arena.is_zero_structural(self.point) {
            self.var
        } else {
            arena.sub(self.var, self.point)
        };
        let mut terms = Vec::new();
        for (i, c) in coeffs.iter().enumerate() {
            if arena.is_zero_structural(c.id) {
                continue;
            }
            let e = self.exponent_id(arena, self.valuation + i as i64);
            let t = if arena.is_zero_structural(e) {
                c.id
            } else if e == arena.one {
                arena.mul(&[c.id, base])
            } else {
                let p = arena.pow(base, e);
                let p = eval::eval(arena, p);
                arena.mul(&[c.id, p])
            };
            terms.push(t);
        }
        for &(p, r) in &self.log_terms {
            if usize::try_from(p).is_ok_and(|p| p < n) {
                let ln = arena.ln(base);
                let bp = pow_i(arena, base, p);
                terms.push(arena.mul(&[r, bp, ln]));
            }
        }
        add_all(arena, &terms)
    }

    // ── Arithmetic ─────────────────────────────────────────────────

    /// `self + other`.  Puiseux series are added in a common ramification;
    /// offsets must differ by a rational number (`Err` for `x^a + x^b`).
    pub fn add(&self, other: &Self) -> Result<Self, SymplexError> {
        self.add_sub(other, false, "fps_add")
    }

    /// `self − other` (see [`add`](Self::add)).
    pub fn sub(&self, other: &Self) -> Result<Self, SymplexError> {
        self.add_sub(other, true, "fps_sub")
    }

    fn add_sub(&self, other: &Self, sub: bool, op: &'static str) -> Result<Self, SymplexError> {
        self.check_compatible(other, op)?;
        self.ensure_usable(op, true)?;
        other.ensure_usable(op, true)?;
        let mismatch = || SymplexError::InvalidArgument {
            operation: op,
            reason: "the exponents of the two series differ by a non-rational amount".into(),
        };
        let (offset, ram) = self.common_frame(other).ok_or_else(mismatch)?;
        let a = self.aligned(offset, ram).ok_or_else(mismatch)?;
        let b = other.aligned(offset, ram).ok_or_else(mismatch)?;
        let mut logs = a.log_terms.clone();
        {
            let mut inner = self.ctx.inner.write();
            let arena = &mut inner.arena;
            for &(p, r) in &b.log_terms {
                let r = if sub { arena.neg(r) } else { r };
                match logs.iter_mut().find(|(q, _)| *q == p) {
                    Some(e) => e.1 = add_all(arena, &[e.1, r]),
                    None => logs.push((p, r)),
                }
            }
        }
        let valuation = a.valuation.min(b.valuation);
        let source = if sub {
            Source::Sub(Box::new(a.clone()), Box::new(b))
        } else {
            Source::Add(Box::new(a.clone()), Box::new(b))
        };
        let mut s = a.derived(source, valuation);
        s.log_terms = logs;
        Ok(s)
    }

    /// Cauchy product `self · other` (offsets add: `√x · √x·eˣ = x·eˣ`).
    pub fn mul(&self, other: &Self) -> Result<Self, SymplexError> {
        self.check_compatible(other, "fps_mul")?;
        self.ensure_usable("fps_mul", false)?;
        other.ensure_usable("fps_mul", false)?;
        let too_fine = || SymplexError::InvalidArgument {
            operation: "fps_mul",
            reason: "ramification too large".into(),
        };
        let ram = lcm_u32(self.ram, other.ram).ok_or_else(too_fine)?;
        let a = self.aligned(self.offset, ram).ok_or_else(too_fine)?;
        let b = other.aligned(other.offset, ram).ok_or_else(too_fine)?;
        let offset = {
            let mut inner = self.ctx.inner.write();
            add_all(&mut inner.arena, &[a.offset, b.offset])
        };
        let valuation = a.valuation.saturating_add(b.valuation);
        let mut s = a.derived(Source::Mul(Box::new(a.clone()), Box::new(b)), valuation);
        s.offset = offset;
        let inner = self.ctx.inner.read();
        Ok(s.reindexed(&inner.arena, offset, ram, 1, 1, 0))
    }

    /// `c · self` for a constant `c`.
    ///
    /// # Panics
    ///
    /// Panics if `c` belongs to a different context.
    #[must_use]
    pub fn scale(&self, c: &Ex) -> Self {
        let cid = self.variable().checked_id(c);
        let mut inner = self.ctx.inner.write();
        let arena = &mut inner.arena;
        if let Source::Refused { expr, reason } = &self.source {
            let e = mul_all(arena, &[cid, *expr]);
            drop(inner);
            return self.refused_like(e, reason);
        }
        let logs = self
            .log_terms
            .iter()
            .map(|&(p, r)| (p, mul_all(arena, &[cid, r])))
            .collect();
        drop(inner);
        let mut s = self.derived(Source::Scale(Box::new(self.clone()), cid), self.valuation);
        s.log_terms = logs;
        s
    }

    /// Composition `self(other)`; requires `self` to be a power series (no
    /// pole, no fractional powers) and `other` to have no constant term and
    /// no negative powers.  `other` may be a Puiseux series
    /// (`exp ∘ √x = exp(√x)`).
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let e = x.exp().fps_maclaurin(&x);
    /// let s = x.sin().fps_maclaurin(&x);
    /// // exp(sin x) = 1 + x + x²/2 − x⁴/8 + …
    /// let c = e.compose(&s).unwrap();
    /// assert_eq!(c.truncate(5).to_string(), "-1/8*x^4 + 1/2*x^2 + x + 1");
    /// ```
    pub fn compose(&self, other: &Self) -> Result<Self, SymplexError> {
        self.check_compatible(other, "fps_compose")?;
        self.ensure_usable("fps_compose", false)?;
        other.ensure_usable("fps_compose", false)?;
        if !self.is_taylor_frame() || !self.vanishes_up_to(-1) {
            return Err(SymplexError::InvalidArgument {
                operation: "fps_compose",
                reason: "outer series must be a power series".into(),
            });
        }
        let g = other
            .with_zero_offset()
            .filter(|g| g.vanishes_up_to(0))
            .ok_or_else(|| SymplexError::InvalidArgument {
                operation: "fps_compose",
                reason: "inner series must have zero constant term".into(),
            })?;
        let mut s = g.derived(
            Source::Compose(Box::new(self.clone()), Box::new(g.clone())),
            0,
        );
        s.offset = self.zero;
        Ok(s)
    }

    /// Every coefficient of index `≤ top` is provably zero.
    fn vanishes_up_to(&self, top: i64) -> bool {
        if self.valuation < -4096 {
            return false;
        }
        (self.valuation..=top).all(|i| self.is_zero_coeff(self.coeff(i)))
    }

    /// Term-wise derivative `d/dx`.
    #[must_use]
    pub fn derivative(&self) -> Self {
        if let Source::Refused { expr, reason } = &self.source {
            let d = {
                let mut inner = self.ctx.inner.write();
                inner.arena.intern(ExprNode::Derivative(*expr, self.var))
            };
            return self.refused_like(d, reason);
        }
        let mut valuation = self.valuation.saturating_sub(i64::from(self.ram));
        let mut logs = Vec::new();
        {
            let mut inner = self.ctx.inner.write();
            let arena = &mut inner.arena;
            for &(p, r) in &self.log_terms {
                if let Some(i) = self.exponent_index_in(arena, p - 1) {
                    valuation = valuation.min(i);
                }
                if p > 0 {
                    let pe = arena.int(p);
                    logs.push((p - 1, mul_all(arena, &[pe, r])));
                }
            }
        }
        let mut s = self.derived(Source::Derivative(Box::new(self.clone())), valuation);
        s.log_terms = logs;
        s
    }

    /// Term-wise antiderivative with zero constant term.  An
    /// `(x − a)^(−1)` term integrates to a logarithm, kept as a
    /// `r·ln(x − a)` term of [`truncate`](Self::truncate) (the
    /// coefficients are those of the power part).
    #[must_use]
    pub fn integral(&self) -> Self {
        if let Source::Refused { expr, reason } = &self.source {
            let i = {
                let mut inner = self.ctx.inner.write();
                inner.arena.intern(ExprNode::Integral(*expr, self.var))
            };
            return self.refused_like(i, reason);
        }
        let residue_index = {
            let inner = self.ctx.inner.read();
            self.exponent_index_in(&inner.arena, -1)
        };
        let mut logs = Vec::new();
        if let Some(i) = residue_index
            && i >= self.valuation
        {
            let c = self.coeff(i);
            if !self.is_zero_coeff(c) {
                logs.push((0, c.id));
            }
        }
        let mut valuation = self.valuation.saturating_add(i64::from(self.ram));
        {
            let mut inner = self.ctx.inner.write();
            let arena = &mut inner.arena;
            for &(p, r) in &self.log_terms {
                if let Some(i) = self.exponent_index_in(arena, p + 1) {
                    valuation = valuation.min(i);
                }
                let c = rat_expr(arena, Q::new(BigInt::one(), BigInt::from(p + 1)));
                logs.push((p + 1, mul_all(arena, &[c, r])));
            }
        }
        let mut s = self.derived(Source::Integral(Box::new(self.clone())), valuation);
        s.log_terms = logs;
        s
    }

    /// Multiplicative inverse `1/self`; requires a non-zero constant term
    /// or a pole (more generally a non-zero coefficient of index `≤ 0`).
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// // 1/cos x = sec x = 1 + x²/2 + 5x⁴/24 + …
    /// let sec = x.cos().fps_maclaurin(&x).inverse().unwrap();
    /// assert_eq!(sec.truncate(5).to_string(), "5/24*x^4 + 1/2*x^2 + 1");
    /// ```
    pub fn inverse(&self) -> Result<Self, SymplexError> {
        self.ensure_usable("fps_inverse", false)?;
        let not_invertible = || SymplexError::InvalidArgument {
            operation: "fps_inverse",
            reason: "series has zero constant term (not invertible as a power series)".into(),
        };
        if self.valuation < -4096 {
            return Err(not_invertible());
        }
        let mut lead = None;
        for i in self.valuation..=0 {
            let c = self.coeff(i);
            if !c.exact {
                return Err(not_invertible());
            }
            if !self.is_zero_coeff(c) {
                lead = Some(i);
                break;
            }
        }
        let l = lead.ok_or_else(not_invertible)?;
        let offset = {
            let mut inner = self.ctx.inner.write();
            let n = inner.arena.neg(self.offset);
            eval::eval(&mut inner.arena, n)
        };
        let mut s = self.derived(Source::Inverse(Box::new(self.clone()), l), -l);
        s.offset = offset;
        Ok(s)
    }

    /// Compositional inverse (series reversion) by Lagrange inversion;
    /// requires a power series with `a_0 = 0` and `a_1 ≠ 0`.
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// // reversion of sin is asin: x + x³/6 + 3x⁵/40
    /// let asin = x.sin().fps_maclaurin(&x).reversion().unwrap();
    /// assert_eq!(asin.truncate(6).to_string(), "3/40*x^5 + 1/6*x^3 + x");
    /// ```
    pub fn reversion(&self) -> Result<Self, SymplexError> {
        self.ensure_usable("fps_reversion", false)?;
        let ok = self.is_taylor_frame() && self.vanishes_up_to(0) && {
            let a1 = self.coeff(1);
            a1.exact && !self.is_zero_coeff(a1)
        };
        if !ok {
            return Err(SymplexError::InvalidArgument {
                operation: "fps_reversion",
                reason: "reversion needs a_0 = 0 and a_1 ≠ 0".into(),
            });
        }
        Ok(self.derived(Source::Reversion(Box::new(self.clone())), 0))
    }
}

/// A public `usize` index as a signed one.
fn index(k: usize) -> i64 {
    i64::try_from(k).unwrap_or(i64::MAX)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn rat(p: i64, q: i64) -> Q {
        Q::new(BigInt::from(p), BigInt::from(q))
    }

    #[test]
    fn exp_coefficients_and_general_term() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let k = ctx.symbol("k");
        let s = x.exp().fps_maclaurin(&x);
        assert!(s.has_closed_form());
        assert_eq!(s.coefficient_rational(0), Some(rat(1, 1)));
        assert_eq!(s.coefficient_rational(5), Some(rat(1, 120)));
        assert_eq!(s.general_term(&k).unwrap().to_string(), "1/k!");
    }

    #[test]
    fn sin_cos_sinh_cosh_coefficients() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let s = x.sin().fps_maclaurin(&x);
        assert_eq!(s.coefficient_rational(1), Some(rat(1, 1)));
        assert_eq!(s.coefficient_rational(2), Some(rat(0, 1)));
        assert_eq!(s.coefficient_rational(3), Some(rat(-1, 6)));
        assert_eq!(s.coefficient_rational(5), Some(rat(1, 120)));
        let c = x.cos().fps_maclaurin(&x);
        assert_eq!(c.coefficient_rational(2), Some(rat(-1, 2)));
        assert_eq!(c.coefficient_rational(4), Some(rat(1, 24)));
        let sh = x.sinh().fps_maclaurin(&x);
        assert_eq!(sh.coefficient_rational(3), Some(rat(1, 6)));
        let ch = x.cosh().fps_maclaurin(&x);
        assert_eq!(ch.coefficient_rational(2), Some(rat(1, 2)));
    }

    #[test]
    fn general_terms_match_coefficients() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let k = ctx.symbol("k");
        let funcs = [
            x.sin(),
            x.cos(),
            x.sinh(),
            x.cosh(),
            x.atan(),
            x.atanh(),
            x.tan(),
            x.tanh(),
            x.asin(),
            x.asinh(),
            x.lambertw(),
            (&ctx.int(1) + &x).ln(),
            (&ctx.int(1) - &x).powi(-1),
            (&ctx.int(1) + &x).pow(&ctx.rational(1, 2)),
        ];
        for f in funcs {
            let s = f.fps_maclaurin(&x);
            let gt = s
                .general_term(&k)
                .unwrap_or_else(|| panic!("no general term for {f}"));
            for i in 0..8usize {
                let a = s.coefficient(i).eval();
                let b = gt.subs_i64(&k, i as i64).eval();
                let av = a.eval_f64().unwrap();
                let bv = b
                    .eval_f64()
                    .unwrap_or_else(|_| panic!("general term of {f} at {i}: {b}"));
                assert!(
                    (av - bv).abs() < 1e-12,
                    "{f}: a_{i} = {a} but general term gives {b}"
                );
            }
        }
    }

    #[test]
    fn scaled_argument_and_shifted_log() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        // exp(2x): a_3 = 8/6 = 4/3
        let s = (&x * 2).exp().fps_maclaurin(&x);
        assert_eq!(s.coefficient_rational(3), Some(rat(4, 3)));
        // sin(x²): a_2 = 1, a_6 = −1/6, odd = 0
        let s = x.powi(2).sin().fps_maclaurin(&x);
        assert_eq!(s.coefficient_rational(2), Some(rat(1, 1)));
        assert_eq!(s.coefficient_rational(3), Some(rat(0, 1)));
        assert_eq!(s.coefficient_rational(6), Some(rat(-1, 6)));
        // ln(2 + x): a_0 = ln 2, a_1 = 1/2, a_2 = −1/8
        let s = (&ctx.int(2) + &x).ln().fps_maclaurin(&x);
        assert_eq!(s.coefficient(0).to_string(), "ln(2)");
        assert_eq!(s.coefficient_rational(1), Some(rat(1, 2)));
        assert_eq!(s.coefficient_rational(2), Some(rat(-1, 8)));
        // 1/(2 − x) = 1/2 + x/4 + x²/8
        let s = (&ctx.int(2) - &x).powi(-1).fps_maclaurin(&x);
        assert_eq!(s.coefficient_rational(0), Some(rat(1, 2)));
        assert_eq!(s.coefficient_rational(2), Some(rat(1, 8)));
    }

    #[test]
    fn engine_fallback_and_laurent() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let s = (&x.exp() + &x.sin()).fps_maclaurin(&x);
        assert!(!s.has_closed_form());
        assert_eq!(s.coefficient_rational(0), Some(rat(1, 1)));
        assert_eq!(s.coefficient_rational(1), Some(rat(2, 1)));
        assert_eq!(s.coefficient_rational(3), Some(rat(0, 1)));
        // 1/(x(1−x)): Laurent
        let f = &ctx.int(1) / &(&x * &(&ctx.int(1) - &x));
        let s = f.fps_maclaurin(&x);
        assert_eq!(s.coefficient_rational(0), Some(rat(1, 1)));
        assert_eq!(s.truncate(2).to_string(), "x + 1/x + 1");
    }

    #[test]
    fn arithmetic() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let e = x.exp().fps_maclaurin(&x);
        let s = x.sin().fps_maclaurin(&x);
        let c = x.cos().fps_maclaurin(&x);
        // sin² + cos² = 1
        let one = s.mul(&s).unwrap().add(&c.mul(&c).unwrap()).unwrap();
        assert_eq!(one.coefficient_rational(0), Some(rat(1, 1)));
        for k in 1..8 {
            assert_eq!(one.coefficient_rational(k), Some(rat(0, 1)), "k={k}");
        }
        // d/dx sin = cos
        let ds = s.derivative();
        for k in 0..8 {
            assert_eq!(ds.coefficient_rational(k), c.coefficient_rational(k));
        }
        // ∫ cos = sin
        let ic = c.integral();
        for k in 0..8 {
            assert_eq!(ic.coefficient_rational(k), s.coefficient_rational(k));
        }
        // e^x · e^{−x} = 1
        let em = (-&x).exp().fps_maclaurin(&x);
        let prod = e.mul(&em).unwrap();
        for k in 1..6 {
            assert_eq!(prod.coefficient_rational(k), Some(rat(0, 1)));
        }
        // 1/e^x = e^{−x}
        let inv = e.inverse().unwrap();
        for k in 0..6 {
            assert_eq!(inv.coefficient_rational(k), em.coefficient_rational(k));
        }
        // scale
        let two_e = e.scale(&ctx.int(2));
        assert_eq!(two_e.coefficient_rational(2), Some(rat(1, 1)));
        // exp(sin x) = 1 + x + x²/2 − x⁴/8 − x⁵/15
        let es = e.compose(&s).unwrap();
        assert_eq!(es.coefficient_rational(4), Some(rat(-1, 8)));
        assert_eq!(es.coefficient_rational(5), Some(rat(-1, 15)));
        // reversion(tan) = atan
        let at = x.tan().fps_maclaurin(&x).reversion().unwrap();
        assert_eq!(at.coefficient_rational(3), Some(rat(-1, 3)));
        assert_eq!(at.coefficient_rational(5), Some(rat(1, 5)));
        // reversion(e^x − 1) = ln(1+x)
        let em1 = e
            .sub(&FormalPowerSeries::from_coefficients(
                &x,
                &ctx.int(0),
                &[ctx.int(1)],
            ))
            .unwrap();
        let l = em1.reversion().unwrap();
        assert_eq!(l.coefficient_rational(2), Some(rat(-1, 2)));
        assert_eq!(l.coefficient_rational(3), Some(rat(1, 3)));
        assert_eq!(l.coefficient_rational(4), Some(rat(-1, 4)));
    }

    #[test]
    fn preconditions() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let s = x.sin().fps_maclaurin(&x);
        assert!(s.inverse().is_err());
        assert!(x.cos().fps_maclaurin(&x).reversion().is_err());
        assert!(
            x.exp()
                .fps_maclaurin(&x)
                .compose(&x.cos().fps_maclaurin(&x))
                .is_err()
        );
        let y = ctx.symbol("y");
        assert!(s.add(&y.sin().fps_maclaurin(&y)).is_err());
    }

    #[test]
    fn nonzero_point() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let one = ctx.int(1);
        // exp(x) about 1: coefficients e/k!
        let s = x.exp().fps(&x, &one);
        assert_eq!(s.coefficient(2).to_string(), "1/2*E");
        let t = s.truncate(3);
        assert!(t.to_string().contains("x - 1"), "{t}");
    }
}

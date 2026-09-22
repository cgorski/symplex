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
//!   `derivative`, `integral`, `inverse` (`1/f`, needs `f(a) ≠ 0`) or
//!   `reversion` (compositional inverse by Lagrange inversion, needs
//!   `f(a) = 0`, `f'(a) ≠ 0`) of other series.  Derived coefficients are
//!   computed from the operands' coefficients exactly, to any index.
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
//! ```

use std::sync::{Arc, Mutex};

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Zero};

use crate::api::context::Context;
use crate::api::expr::Ex;
use crate::base::arena::Arena;
use crate::base::combinatorics::factorial;
use crate::base::errors::SymplexError;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::Q;
use crate::base::walk;
use crate::calculus::series::{FnKind, TSeries, expand_maclaurin};
use crate::transforms::eval;

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

/// A formal power series `Σ a_k (x − point)^k` with exact, lazily computed
/// coefficients.  See the [module docs](self) for the design.
pub struct FormalPowerSeries {
    ctx: Context,
    var: ExprId,
    point: ExprId,
    source: Source,
    /// Memoised coefficients `a_0, a_1, …`.
    memo: Mutex<Vec<ExprId>>,
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
    Known(Known),
    /// Expanded on demand by the truncated-series engine.
    Engine {
        expr: ExprId,
        cache: Mutex<Option<TSeries>>,
    },
    Polynomial(Vec<ExprId>),
    Add(Box<FormalPowerSeries>, Box<FormalPowerSeries>),
    Sub(Box<FormalPowerSeries>, Box<FormalPowerSeries>),
    Mul(Box<FormalPowerSeries>, Box<FormalPowerSeries>),
    Scale(Box<FormalPowerSeries>, ExprId),
    Compose(Box<FormalPowerSeries>, Box<FormalPowerSeries>),
    Derivative(Box<FormalPowerSeries>),
    Integral(Box<FormalPowerSeries>),
    Inverse(Box<FormalPowerSeries>),
    Reversion(Box<FormalPowerSeries>),
}

impl Clone for FormalPowerSeries {
    fn clone(&self) -> Self {
        let source = match &self.source {
            Source::Known(k) => Source::Known(k.clone()),
            Source::Engine { expr, cache } => Source::Engine {
                expr: *expr,
                cache: Mutex::new(cache.lock().ok().and_then(|c| c.clone())),
            },
            Source::Polynomial(v) => Source::Polynomial(v.clone()),
            Source::Add(a, b) => Source::Add(a.clone(), b.clone()),
            Source::Sub(a, b) => Source::Sub(a.clone(), b.clone()),
            Source::Mul(a, b) => Source::Mul(a.clone(), b.clone()),
            Source::Scale(a, c) => Source::Scale(a.clone(), *c),
            Source::Compose(a, b) => Source::Compose(a.clone(), b.clone()),
            Source::Derivative(a) => Source::Derivative(a.clone()),
            Source::Integral(a) => Source::Integral(a.clone()),
            Source::Inverse(a) => Source::Inverse(a.clone()),
            Source::Reversion(a) => Source::Reversion(a.clone()),
        };
        FormalPowerSeries {
            ctx: self.ctx.clone(),
            var: self.var,
            point: self.point,
            source,
            memo: Mutex::new(self.memo.lock().map(|m| m.clone()).unwrap_or_default()),
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
pub(crate) fn fps(ctx: &Context, expr: ExprId, var: ExprId, point: ExprId) -> FormalPowerSeries {
    let source = {
        let mut inner = ctx.inner.write();
        let arena = &mut inner.arena;
        let at_zero = arena.is_zero_structural(point);
        let shifted = if at_zero {
            expr
        } else {
            let v = arena.add(&[var, point]);
            crate::transforms::subs::subs(arena, expr, var, v)
        };
        match detect_known(arena, shifted, var) {
            Some(k) => Source::Known(k),
            None => {
                // Polynomials get an explicit coefficient list.
                if let Some(terms) = crate::calculus::summation::sym_poly_in(arena, shifted, var) {
                    let deg = terms.iter().map(|(p, _)| *p).max().unwrap_or(0);
                    let mut coeffs = vec![arena.zero; deg + 1];
                    for (p, c) in terms {
                        coeffs[p] = c;
                    }
                    Source::Polynomial(coeffs)
                } else {
                    Source::Engine {
                        expr: shifted,
                        cache: Mutex::new(None),
                    }
                }
            }
        }
    };
    FormalPowerSeries {
        ctx: ctx.clone(),
        var,
        point,
        source,
        memo: Mutex::new(Vec::new()),
    }
}

impl FormalPowerSeries {
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
        FormalPowerSeries {
            ctx: var.context(),
            var: var.raw_id(),
            point: point_id,
            source: Source::Polynomial(ids),
            memo: Mutex::new(Vec::new()),
        }
    }

    fn derived(&self, source: Source) -> Self {
        FormalPowerSeries {
            ctx: self.ctx.clone(),
            var: self.var,
            point: self.point,
            source,
            memo: Mutex::new(Vec::new()),
        }
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

    /// The `k`-th coefficient `a_k` as an exact expression.
    ///
    /// For series with a pole at the expansion point this is the Laurent
    /// coefficient of `(x − a)^k` for `k ≥ 0`; use
    /// [`truncate`](Self::truncate) to see the negative powers.
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
        let id = self.coefficient_id(k);
        self.wrap(id)
    }

    /// The `k`-th coefficient as an exact rational, or `None` if it is not
    /// a rational number (e.g. `ln 2`, `2/√π`, or a symbolic parameter).
    #[must_use]
    pub fn coefficient_rational(&self, k: usize) -> Option<Q> {
        let id = self.coefficient_id(k);
        let inner = self.ctx.inner.read();
        inner.arena.as_num(id).cloned()
    }

    /// The first `n` coefficients `a_0 … a_{n−1}`.
    #[must_use]
    pub fn coefficients(&self, n: usize) -> Vec<Ex> {
        (0..n).map(|k| self.coefficient(k)).collect()
    }

    fn coefficient_id(&self, k: usize) -> ExprId {
        if let Ok(memo) = self.memo.lock()
            && k < memo.len()
        {
            return memo[k];
        }
        let id = self.compute_coefficient(k);
        if let Ok(mut memo) = self.memo.lock()
            && k == memo.len()
        {
            memo.push(id);
        }
        id
    }

    /// Ensure `a_0..a_{n-1}` are memoised (sequentially — recurrences need
    /// earlier terms) and return them.
    fn coefficient_ids(&self, n: usize) -> Vec<ExprId> {
        (0..n).map(|k| self.coefficient_id(k)).collect()
    }

    fn compute_coefficient(&self, k: usize) -> ExprId {
        match &self.source {
            Source::Known(known) => {
                let mut inner = self.ctx.inner.write();
                known.coefficient(&mut inner.arena, k)
            }
            Source::Engine { expr, cache } => {
                let mut inner = self.ctx.inner.write();
                let arena = &mut inner.arena;
                let mut guard = match cache.lock() {
                    Ok(g) => g,
                    Err(p) => p.into_inner(),
                };
                let need = k as i64 + 1;
                if guard.as_ref().is_none_or(|ts| ts.known() < need) {
                    let order = need.max(guard.as_ref().map_or(8, |ts| ts.known() * 2));
                    *guard = expand_maclaurin(arena, *expr, self.var, order, false).ok();
                }
                match guard.as_ref() {
                    Some(ts) if ts.known() >= need => ts.coefficient(arena, k as i64),
                    // The expansion does not exist (Puiseux / essential
                    // singularity); expose the failure as NaN.
                    _ => arena.nan,
                }
            }
            Source::Polynomial(c) => {
                let inner = self.ctx.inner.read();
                c.get(k).copied().unwrap_or(inner.arena.zero)
            }
            Source::Add(a, b) => {
                let x = a.coefficient_id(k);
                let y = b.coefficient_id(k);
                let mut inner = self.ctx.inner.write();
                add_all(&mut inner.arena, &[x, y])
            }
            Source::Sub(a, b) => {
                let x = a.coefficient_id(k);
                let y = b.coefficient_id(k);
                let mut inner = self.ctx.inner.write();
                let ny = inner.arena.neg(y);
                add_all(&mut inner.arena, &[x, ny])
            }
            Source::Mul(a, b) => {
                let xs = a.coefficient_ids(k + 1);
                let ys = b.coefficient_ids(k + 1);
                let mut inner = self.ctx.inner.write();
                let arena = &mut inner.arena;
                let mut terms = Vec::new();
                for i in 0..=k {
                    if arena.is_zero_structural(xs[i]) || arena.is_zero_structural(ys[k - i]) {
                        continue;
                    }
                    terms.push(arena.mul(&[xs[i], ys[k - i]]));
                }
                add_all(arena, &terms)
            }
            Source::Scale(a, c) => {
                let x = a.coefficient_id(k);
                let mut inner = self.ctx.inner.write();
                mul_all(&mut inner.arena, &[*c, x])
            }
            Source::Derivative(a) => {
                let x = a.coefficient_id(k + 1);
                let mut inner = self.ctx.inner.write();
                let kp1 = inner.arena.int(k as i64 + 1);
                mul_all(&mut inner.arena, &[kp1, x])
            }
            Source::Integral(a) => {
                if k == 0 {
                    let inner = self.ctx.inner.read();
                    return inner.arena.zero;
                }
                let x = a.coefficient_id(k - 1);
                let mut inner = self.ctx.inner.write();
                let inv = rat_expr(&mut inner.arena, Q::new(BigInt::one(), BigInt::from(k)));
                mul_all(&mut inner.arena, &[inv, x])
            }
            Source::Inverse(a) => {
                let xs = a.coefficient_ids(k + 1);
                let mut inner = self.ctx.inner.write();
                let b = poly_inverse(&mut inner.arena, &xs, k + 1);
                b[k]
            }
            Source::Compose(f, g) => {
                // Σ_{n≤k} f_n · (g^n)_k with g_0 = 0.
                let fs = f.coefficient_ids(k + 1);
                let gs = g.coefficient_ids(k + 1);
                let mut inner = self.ctx.inner.write();
                let arena = &mut inner.arena;
                let n = k + 1;
                let mut power: Vec<ExprId> = vec![arena.zero; n];
                power[0] = arena.one;
                let mut terms = Vec::new();
                for (i, &fi) in fs.iter().enumerate() {
                    if i > 0 {
                        power = poly_mul(arena, &power, &gs, n);
                    }
                    if arena.is_zero_structural(fi) || arena.is_zero_structural(power[k]) {
                        continue;
                    }
                    terms.push(arena.mul(&[fi, power[k]]));
                }
                add_all(arena, &terms)
            }
            Source::Reversion(f) => {
                // Lagrange inversion: [x^k] g = (1/k) [x^{k−1}] (x/f(x))^k, k ≥ 1.
                if k == 0 {
                    let inner = self.ctx.inner.read();
                    return inner.arena.zero;
                }
                let fs = f.coefficient_ids(k + 1);
                let mut inner = self.ctx.inner.write();
                let arena = &mut inner.arena;
                // h = f/x  (drop f_0 = 0)
                let h: Vec<ExprId> = fs[1..].to_vec();
                let inv_h = poly_inverse(arena, &h, k);
                let mut power: Vec<ExprId> = vec![arena.zero; k];
                power[0] = arena.one;
                for _ in 0..k {
                    power = poly_mul(arena, &power, &inv_h, k);
                }
                let inv_k = rat_expr(arena, Q::new(BigInt::one(), BigInt::from(k)));
                mul_all(arena, &[inv_k, power[k - 1]])
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
            Source::Scale(a, _) | Source::Derivative(a) | Source::Integral(a) => {
                a.has_closed_form()
            }
            _ => false,
        }
    }

    /// The general coefficient `a_k` as a closed-form expression in the
    /// symbolic index `k`, when known.
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
                let mut inner = self.ctx.inner.write();
                let one = inner.arena.one;
                let kp1 = inner.arena.add(&[k, one]);
                drop(inner);
                let x = a.general_term_id(kp1)?;
                let mut inner = self.ctx.inner.write();
                Some(mul_all(&mut inner.arena, &[kp1, x]))
            }
            Source::Integral(a) => {
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
            _ => None,
        }
    }

    /// The truncated series `Σ_{k<n} a_k (x − point)^k`.
    ///
    /// For series with a pole at the point, negative powers down to the
    /// pole order are included.
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let s = x.cos().fps_maclaurin(&x);
    /// assert_eq!(s.truncate(5).to_string(), "1/24*x^4 - 1/2*x^2 + 1");
    /// ```
    #[must_use]
    pub fn truncate(&self, n: usize) -> Ex {
        let coeffs = self.coefficient_ids(n);
        let mut inner = self.ctx.inner.write();
        let arena = &mut inner.arena;
        let base = if arena.is_zero_structural(self.point) {
            self.var
        } else {
            arena.sub(self.var, self.point)
        };
        let mut terms = Vec::new();
        // Negative powers for engine series with a pole.
        if let Source::Engine { cache, .. } = &self.source
            && let Ok(guard) = cache.lock()
            && let Some(ts) = guard.as_ref()
        {
            for e in ts.shift()..0 {
                let c = ts.coefficient(arena, e);
                if arena.is_zero_structural(c) {
                    continue;
                }
                let p = pow_i(arena, base, e);
                terms.push(arena.mul(&[c, p]));
            }
        }
        for (k, &c) in coeffs.iter().enumerate() {
            if arena.is_zero_structural(c) {
                continue;
            }
            let t = if k == 0 {
                c
            } else {
                let p = pow_i(arena, base, k as i64);
                arena.mul(&[c, p])
            };
            terms.push(t);
        }
        let id = add_all(arena, &terms);
        drop(inner);
        self.wrap(id)
    }

    // ── Arithmetic ─────────────────────────────────────────────────

    /// `self + other`.
    pub fn add(&self, other: &Self) -> Result<Self, SymplexError> {
        self.check_compatible(other, "fps_add")?;
        Ok(self.derived(Source::Add(Box::new(self.clone()), Box::new(other.clone()))))
    }

    /// `self − other`.
    pub fn sub(&self, other: &Self) -> Result<Self, SymplexError> {
        self.check_compatible(other, "fps_sub")?;
        Ok(self.derived(Source::Sub(Box::new(self.clone()), Box::new(other.clone()))))
    }

    /// Cauchy product `self · other`.
    pub fn mul(&self, other: &Self) -> Result<Self, SymplexError> {
        self.check_compatible(other, "fps_mul")?;
        Ok(self.derived(Source::Mul(Box::new(self.clone()), Box::new(other.clone()))))
    }

    /// `c · self` for a constant `c`.
    ///
    /// # Panics
    ///
    /// Panics if `c` belongs to a different context.
    #[must_use]
    pub fn scale(&self, c: &Ex) -> Self {
        let cid = self.variable().checked_id(c);
        self.derived(Source::Scale(Box::new(self.clone()), cid))
    }

    /// Composition `self(other)`; requires `other` to have zero constant term.
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
        let g0 = other.coefficient_id(0);
        let mut inner = self.ctx.inner.write();
        let zero = is_zero_expr(&mut inner.arena, g0);
        drop(inner);
        if !zero {
            return Err(SymplexError::InvalidArgument {
                operation: "fps_compose",
                reason: "inner series must have zero constant term".into(),
            });
        }
        Ok(self.derived(Source::Compose(
            Box::new(self.clone()),
            Box::new(other.clone()),
        )))
    }

    /// Term-wise derivative `d/dx`.
    #[must_use]
    pub fn derivative(&self) -> Self {
        self.derived(Source::Derivative(Box::new(self.clone())))
    }

    /// Term-wise antiderivative with zero constant term.
    #[must_use]
    pub fn integral(&self) -> Self {
        self.derived(Source::Integral(Box::new(self.clone())))
    }

    /// Multiplicative inverse `1/self`; requires a non-zero constant term.
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
        let a0 = self.coefficient_id(0);
        let mut inner = self.ctx.inner.write();
        let zero = is_zero_expr(&mut inner.arena, a0);
        drop(inner);
        if zero {
            return Err(SymplexError::InvalidArgument {
                operation: "fps_inverse",
                reason: "series has zero constant term (not invertible as a power series)".into(),
            });
        }
        Ok(self.derived(Source::Inverse(Box::new(self.clone()))))
    }

    /// Compositional inverse (series reversion) by Lagrange inversion;
    /// requires `a_0 = 0` and `a_1 ≠ 0`.
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
        let a0 = self.coefficient_id(0);
        let a1 = self.coefficient_id(1);
        let mut inner = self.ctx.inner.write();
        let z0 = is_zero_expr(&mut inner.arena, a0);
        let z1 = is_zero_expr(&mut inner.arena, a1);
        drop(inner);
        if !z0 || z1 {
            return Err(SymplexError::InvalidArgument {
                operation: "fps_reversion",
                reason: "reversion needs a_0 = 0 and a_1 ≠ 0".into(),
            });
        }
        Ok(self.derived(Source::Reversion(Box::new(self.clone()))))
    }
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

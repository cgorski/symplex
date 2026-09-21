//! Symbolic summation and products: Faulhaber, telescoping, hypergeometric,
//! infinite sums, and closed-form products.
//!
//! This module is the backend behind [`Ex::summation`](crate::api::expr::Ex::summation)
//! and [`Ex::product_over`](crate::api::expr::Ex::product_over).  It works on
//! `&mut Arena` + `ExprId` and is also used by `eval()` when it encounters a
//! `Sum` node (via the thin shim in `calculus::sum_eval`).
//!
//! # Summation strategies
//!
//! For `Σ_{k=lo}^{hi} f(k)` the dispatcher tries, in order:
//!
//! 1. **Empty / enumerable ranges** — concrete integer bounds with at most
//!    [`MAX_ENUMERATION_TERMS`] terms are summed directly (exactly).
//! 2. **Constant body** — `f` independent of `k` gives `f·(hi − lo + 1)`.
//! 3. **Rational functions of `k`** — partial fractions, then each pole
//!    family `c/(k+β)^m` is summed with harmonic numbers / digamma, and
//!    integer-shifted poles telescope exactly (`Σ 1/(k(k+1)) = 1 − 1/(n+1)`).
//! 4. **Telescoping** `g(k) − g(k+d)` for two-term bodies.
//! 5. **Linearity** — sums of terms are summed term-by-term; terms that
//!    cannot be summed are kept as an unevaluated `Sum`.
//! 6. **Polynomials in `k`** (any degree, symbolic coefficients allowed) via
//!    Faulhaber's formula with exact Bernoulli numbers.
//! 7. **Binomial identities** (`Σ C(n,k) = 2ⁿ`, `Σ k·C(n,k) = n·2ⁿ⁻¹`,
//!    `Σ C(n,k)² = C(2n,n)`, `Σ C(n,k) xᵏ = (1+x)ⁿ`, …).  Geometric
//!    exponents may carry a symbolic `k`-free offset, so the binomial
//!    theorem `Σ C(n,k) pᵏ (1−p)ⁿ⁻ᵏ = 1` closes for symbolic `n` and `p`.
//! 8. **Geometric / arithmetico-geometric** `Σ P(k)·rᵏ` with symbolic `r`
//!    (a `Piecewise` covers `r = 1`).
//! 9. **Gosper's algorithm** for hypergeometric terms (including
//!    `C(k+c, k)`-type binomials with `k` in both arguments).
//!
//! For infinite upper bounds the engine additionally recognises p-series
//! (`ζ(2m)` in closed form), alternating p-series (`η`, Dirichlet `β`),
//! convergent geometric series, the negative-binomial series
//! `Σ P(k)·C(k+c,k)·xᵏ` (`(1−x)^{−(c+1)}` for `P = 1`, any `c`), and a
//! table of classical power series
//! (`Σ xᵏ/k! = eˣ`, `Σ (−1)ᵏ x²ᵏ⁺¹/(2k+1)! = sin x`, `Σ xᵏ/k = −ln(1−x)`, …),
//! and proves divergence where it can.
//!
//! **Symbolic ratios.**  Convergence of an infinite geometric-type series
//! is decided only for a *numeric* ratio.  There is no `|r| < 1` assumption,
//! so `Σ_{k≥0} rᵏ`, `Σ k (1−p)ᵏ⁻¹ p` or `Σ C(k+c,k) xᵏ` with a symbolic
//! ratio stay unevaluated (SymPy returns a `Piecewise` over `Abs(r) < 1`
//! instead).  Sums whose ratio is numeric but whose other parameters are
//! symbolic (`Σ C(k+c,k) (1/3)ᵏ = (2/3)^{−c−1}`) do close.
//!
//! Values without an elementary closed form use the dedicated nodes:
//! `Σ 1/k³ = ζ(3)` (`Zeta`), `Σ (−1)^k/(2k+1)² = G` (`Catalan`).

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Signed, ToPrimitive, Zero};

use crate::base::arena::Arena;
use crate::base::bernoulli::bernoulli;
use crate::base::extended::Extended;
use crate::base::node::{ExprId, ExprNode};
use crate::base::walk;
use crate::calculus::gosper;
use crate::poly::Poly;
use crate::poly::polybridge;
use crate::transforms::{apart, eval, subs};

type Rat = Ratio<BigInt>;

/// Maximum number of terms that will be summed / multiplied by direct
/// enumeration when both bounds are concrete integers.
pub const MAX_ENUMERATION_TERMS: i64 = 1000;

/// Maximum integer shift between two poles that is telescoped explicitly.
const MAX_TELESCOPE_SHIFT: i64 = 200;

/// Maximum integer shift between Gamma-function arguments that is expanded
/// into a product of linear factors when normalising product results.
const MAX_GAMMA_SHIFT: i64 = 60;

// ═══════════════════════════════════════════════════════════════════════════
// Result type
// ═══════════════════════════════════════════════════════════════════════════

/// Outcome of a symbolic summation or product.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SumOutcome {
    /// A closed form was found.  It may still contain an unevaluated `Sum`
    /// for terms that could not be summed (partial linearity).
    Closed(ExprId),
    /// The sum / product was proven divergent.  `Some(±∞)` when the
    /// direction is known, `None` for oscillating divergence.
    Divergent(Option<ExprId>),
    /// No strategy applied — the caller should keep the formal node.
    Unevaluated,
}

// ═══════════════════════════════════════════════════════════════════════════
// Small helpers
// ═══════════════════════════════════════════════════════════════════════════

fn rat_i(n: i64) -> Rat {
    Ratio::from_integer(BigInt::from(n))
}

fn rat_expr(arena: &mut Arena, r: Rat) -> ExprId {
    let nid = arena.intern_num(r);
    arena.intern(ExprNode::Num(nid))
}

fn depends_on(arena: &Arena, e: ExprId, var: ExprId) -> bool {
    walk::contains(arena, e, var)
}

fn as_rat(arena: &Arena, e: ExprId) -> Option<Rat> {
    arena.as_num(e).cloned()
}

fn as_i64(arena: &Arena, e: ExprId) -> Option<i64> {
    arena.as_num(e).and_then(|r| {
        if r.is_integer() {
            r.numer().to_i64()
        } else {
            None
        }
    })
}

fn eval_rat(arena: &mut Arena, e: ExprId) -> Option<Rat> {
    let v = eval::eval(arena, e);
    as_rat(arena, v)
}

fn mul_all(arena: &mut Arena, factors: &[ExprId]) -> ExprId {
    match factors.len() {
        0 => arena.one,
        1 => factors[0],
        _ => arena.mul(factors),
    }
}

fn add_all(arena: &mut Arena, terms: &[ExprId]) -> ExprId {
    match terms.len() {
        0 => arena.zero,
        1 => terms[0],
        _ => arena.add(terms),
    }
}

fn add_terms(arena: &Arena, e: ExprId) -> Vec<ExprId> {
    match arena.node(e) {
        ExprNode::Add(ch) => ch.to_vec(),
        _ => vec![e],
    }
}

fn mul_factors(arena: &Arena, e: ExprId) -> Vec<ExprId> {
    match arena.node(e) {
        ExprNode::Mul(ch) => ch.to_vec(),
        _ => vec![e],
    }
}

/// `pow(base, r)` for a rational exponent.
fn pow_rat(arena: &mut Arena, base: ExprId, r: &Rat) -> ExprId {
    if r.is_zero() {
        return arena.one;
    }
    if r.is_one() {
        return base;
    }
    let e = rat_expr(arena, r.clone());
    arena.pow(base, e)
}

/// `r^n` for rational `r` and integer `n` (exact).
fn rat_pow_i(r: &Rat, n: i64) -> Rat {
    let mut acc = Rat::one();
    let base = if n < 0 { Rat::one() / r } else { r.clone() };
    for _ in 0..n.unsigned_abs() {
        acc *= &base;
    }
    acc
}

fn factorial_big(n: u64) -> BigInt {
    let mut acc = BigInt::one();
    for i in 2..=n {
        acc *= BigInt::from(i);
    }
    acc
}

fn binomial_big(n: u64, k: u64) -> BigInt {
    if k > n {
        return BigInt::zero();
    }
    let mut acc = BigInt::one();
    for i in 0..k {
        acc = acc * BigInt::from(n - i) / BigInt::from(i + 1);
    }
    acc
}

/// Put a sum of fractions over a common denominator (no-op for non-sums).
fn combine_fractions(arena: &mut Arena, e: ExprId) -> ExprId {
    if matches!(arena.node(e), ExprNode::Add(_)) {
        let t = polybridge::together(arena, e);
        eval::eval(arena, t)
    } else {
        e
    }
}

/// Fractional part `β − ⌊β⌋` of a rational.
fn frac_part(r: &Rat) -> Rat {
    r - r.floor()
}

/// Is `x + c` (as an expression) — builds `x + c` with rational `c`.
fn add_rat(arena: &mut Arena, x: ExprId, c: &Rat) -> ExprId {
    if c.is_zero() {
        return x;
    }
    let ce = rat_expr(arena, c.clone());
    arena.add(&[x, ce])
}

/// Evaluate `e` and, if the result is a rational number, return it.
fn const_value(arena: &mut Arena, e: ExprId) -> Option<Rat> {
    eval_rat(arena, e)
}

/// `hi − lo + 1`.
fn range_count(arena: &mut Arena, lo: ExprId, hi: ExprId) -> ExprId {
    let d = arena.sub(hi, lo);
    let one = arena.one;
    arena.add(&[d, one])
}

/// Extract `(a, b)` such that `e = a·var + b` with rational `a ≠ 0`, `b`.
fn linear_in(arena: &Arena, e: ExprId, var: ExprId) -> Option<(Rat, Rat)> {
    let p = polybridge::expr_to_poly(arena, e, var)?;
    match p.degree() {
        Some(1) => Some((p.coeff(1), p.coeff(0))),
        _ => None,
    }
}

/// Extract `(a, b)` such that `e = a·var + b` with rational `a ≠ 0` and a
/// `var`-free intercept `b` that may be symbolic (`n − k` → `(−1, n)`).
fn linear_in_sym(arena: &mut Arena, e: ExprId, var: ExprId) -> Option<(Rat, ExprId)> {
    if let Some((a, b)) = linear_in(arena, e, var) {
        return Some((a, rat_expr(arena, b)));
    }
    let monomials = sym_poly_in(arena, e, var)?;
    let mut slope: Option<Rat> = None;
    let mut intercept = arena.zero;
    for (deg, coeff) in monomials {
        match deg {
            0 => intercept = coeff,
            1 => slope = Some(as_rat(arena, coeff)?),
            _ => return None,
        }
    }
    let a = slope?;
    if a.is_zero() {
        return None;
    }
    Some((a, intercept))
}

/// `base^b` for a `var`-free exponent (`1` when `b` is zero).
fn pow_const(arena: &mut Arena, base: ExprId, b: ExprId) -> ExprId {
    if let Some(r) = as_rat(arena, b) {
        return pow_rat(arena, base, &r);
    }
    arena.pow(base, b)
}

/// `r·e` for a rational `r` and a `var`-free expression `e`.
fn scale_const(arena: &mut Arena, e: ExprId, r: &Rat) -> ExprId {
    if r.is_one() {
        return e;
    }
    if let Some(v) = as_rat(arena, e) {
        return rat_expr(arena, v * r);
    }
    let re = rat_expr(arena, r.clone());
    arena.mul(&[re, e])
}

/// A "structural" zero test: expand + eval, and if that does not reach a
/// literal zero, fall back to the unified simplifier.
fn is_zero_expr(arena: &mut Arena, e: ExprId) -> bool {
    let ex = arena.expand_expr(e);
    let ev = eval::eval(arena, ex);
    if arena.is_zero_structural(ev) {
        return true;
    }
    if let Some(r) = as_rat(arena, ev) {
        return r.is_zero();
    }
    let simp = crate::simplify::simplify_engine::unified_simplify(
        arena,
        ev,
        &crate::simplify::simplify_engine::SimplifyOpts::single_pass(),
    );
    arena.is_zero_structural(simp.expr)
}

/// Sign of a k-free expression: `Some(true)` positive, `Some(false)`
/// negative, `None` unknown / zero.
fn const_sign(arena: &mut Arena, e: ExprId) -> Option<bool> {
    if let Some(r) = const_value(arena, e) {
        if r.is_positive() {
            return Some(true);
        }
        if r.is_negative() {
            return Some(false);
        }
        return None;
    }
    let f = crate::transforms::evalf::eval_const_f64(arena, e)?;
    if f > 0.0 {
        Some(true)
    } else if f < 0.0 {
        Some(false)
    } else {
        None
    }
}

/// Numeric magnitude test `|e| < 1` for a k-free expression.
///
/// Exact for rationals; for other constants uses a 16-digit evaluation
/// with a safety margin and returns `None` when the value is within
/// `1e-9` of the unit circle.
fn abs_less_than_one(arena: &mut Arena, e: ExprId) -> Option<bool> {
    if let Some(r) = const_value(arena, e) {
        return Some(r.abs() < Rat::one());
    }
    let f = crate::transforms::evalf::eval_const_f64(arena, e)?;
    if !f.is_finite() {
        return Some(false);
    }
    let a = f.abs();
    if a < 1.0 - 1e-9 {
        Some(true)
    } else if a > 1.0 + 1e-9 {
        Some(false)
    } else {
        None
    }
}

fn infinity_of_sign(arena: &Arena, positive: Option<bool>) -> Option<ExprId> {
    match positive {
        Some(true) => Some(arena.infinity),
        Some(false) => Some(arena.neg_infinity),
        None => None,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Bounds
// ═══════════════════════════════════════════════════════════════════════════

/// A summation limit: `±∞`, or any other expression (an integer, a
/// symbol, …) as `Finite`.
fn classify_bound(arena: &Arena, b: ExprId) -> Extended<ExprId> {
    match arena.node(b) {
        ExprNode::Infinity => Extended::PosInf,
        ExprNode::NegInfinity => Extended::NegInf,
        _ => Extended::Finite(b),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Public (crate) entry points
// ═══════════════════════════════════════════════════════════════════════════

/// Evaluate `Σ_{var=lower}^{upper} body` symbolically.
///
/// `lower` / `upper` may be concrete integers, symbolic expressions, or
/// `Infinity` / `NegInfinity`.  See the module docs for the strategies.
pub(crate) fn summation(
    arena: &mut Arena,
    body: ExprId,
    var: ExprId,
    lower: ExprId,
    upper: ExprId,
) -> SumOutcome {
    if !matches!(arena.node(var), ExprNode::Symbol(_)) {
        return SumOutcome::Unevaluated;
    }
    tracing::debug!("summation: dispatching");
    match (classify_bound(arena, lower), classify_bound(arena, upper)) {
        (Extended::Finite(lo), Extended::Finite(hi)) => finite_sum(arena, body, var, lo, hi),
        (Extended::Finite(lo), Extended::PosInf) => infinite_sum(arena, body, var, lo),
        (Extended::NegInf, Extended::Finite(hi)) => {
            // k = −j:  Σ_{k=−∞}^{hi} f(k) = Σ_{j=−hi}^{∞} f(−j)
            let neg_var = arena.neg(var);
            let reflected = subs::subs(arena, body, var, neg_var);
            let lo2 = arena.neg(hi);
            let lo2 = eval::eval(arena, lo2);
            infinite_sum(arena, reflected, var, lo2)
        }
        (Extended::NegInf, Extended::PosInf) => {
            let zero = arena.zero;
            let one = arena.one;
            let right = infinite_sum(arena, body, var, zero);
            let neg_var = arena.neg(var);
            let reflected = subs::subs(arena, body, var, neg_var);
            let left = infinite_sum(arena, reflected, var, one);
            combine_outcomes(arena, right, left)
        }
        _ => SumOutcome::Unevaluated,
    }
}

/// Combine two independent partial results by addition.
fn combine_outcomes(arena: &mut Arena, a: SumOutcome, b: SumOutcome) -> SumOutcome {
    use SumOutcome::*;
    match (a, b) {
        (Closed(x), Closed(y)) => {
            let s = arena.add(&[x, y]);
            Closed(eval::eval(arena, s))
        }
        (Divergent(d), Closed(_)) | (Closed(_), Divergent(d)) => Divergent(d),
        (Divergent(Some(x)), Divergent(Some(y))) if x == y => Divergent(Some(x)),
        (Divergent(_), Divergent(_)) => Divergent(None),
        _ => Unevaluated,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Finite sums
// ═══════════════════════════════════════════════════════════════════════════

fn finite_sum(arena: &mut Arena, body: ExprId, var: ExprId, lo: ExprId, hi: ExprId) -> SumOutcome {
    if let (Some(a), Some(b)) = (as_i64(arena, lo), as_i64(arena, hi)) {
        if b < a {
            return SumOutcome::Closed(arena.zero);
        }
        if b - a < MAX_ENUMERATION_TERMS {
            tracing::debug!("summation: enumerating {} terms", b - a + 1);
            return SumOutcome::Closed(enumerate_sum(arena, body, var, a, b));
        }
    }
    match finite_closed(arena, body, var, lo, hi) {
        Some(id) => SumOutcome::Closed(eval::eval(arena, id)),
        None => SumOutcome::Unevaluated,
    }
}

fn enumerate_sum(arena: &mut Arena, body: ExprId, var: ExprId, a: i64, b: i64) -> ExprId {
    let mut terms = Vec::with_capacity((b - a + 1) as usize);
    for k in a..=b {
        let kk = arena.int(k);
        let t = subs::subs(arena, body, var, kk);
        terms.push(eval::eval(arena, t));
    }
    let s = add_all(arena, &terms);
    eval::eval(arena, s)
}

fn enumerate_product(arena: &mut Arena, body: ExprId, var: ExprId, a: i64, b: i64) -> ExprId {
    let mut factors = Vec::with_capacity((b - a + 1) as usize);
    for k in a..=b {
        let kk = arena.int(k);
        let t = subs::subs(arena, body, var, kk);
        factors.push(eval::eval(arena, t));
    }
    let p = mul_all(arena, &factors);
    eval::eval(arena, p)
}

/// Closed form of a finite sum with (possibly symbolic) bounds.
fn finite_closed(
    arena: &mut Arena,
    body: ExprId,
    var: ExprId,
    lo: ExprId,
    hi: ExprId,
) -> Option<ExprId> {
    // 0. Body independent of the summation variable.
    if !depends_on(arena, body, var) {
        let n = range_count(arena, lo, hi);
        return Some(arena.mul(&[body, n]));
    }

    let node = arena.node(body).clone();

    // 1. Sums: whole-rational, telescoping, linearity.
    if let ExprNode::Add(ref terms) = node {
        let terms: Vec<ExprId> = terms.to_vec();
        if let Some(r) = rational_sum_finite(arena, body, var, lo, hi) {
            return Some(r);
        }
        if let Some(r) = telescoping_finite(arena, &terms, var, lo, hi) {
            return Some(r);
        }
        if let Some(p) = sym_poly_in(arena, body, var) {
            return Some(faulhaber_sym(arena, &p, lo, hi));
        }
        let mut done = Vec::new();
        let mut failed = Vec::new();
        for &t in &terms {
            match finite_closed(arena, t, var, lo, hi) {
                Some(v) => done.push(v),
                None => failed.push(t),
            }
        }
        if done.is_empty() {
            return None;
        }
        if !failed.is_empty() {
            let rest = add_all(arena, &failed);
            done.push(arena.intern(ExprNode::Sum(rest, var, lo, hi)));
        }
        return Some(add_all(arena, &done));
    }

    // 2. Constant-factor extraction.
    if let ExprNode::Mul(ref factors) = node {
        let factors: Vec<ExprId> = factors.to_vec();
        let (consts, varf): (Vec<ExprId>, Vec<ExprId>) = factors
            .iter()
            .copied()
            .partition(|&f| !depends_on(arena, f, var));
        if !consts.is_empty() && !varf.is_empty() {
            let inner = mul_all(arena, &varf);
            let s = finite_closed(arena, inner, var, lo, hi)?;
            let mut all = consts;
            all.push(s);
            return Some(arena.mul(&all));
        }
    }

    // 3. Polynomial in k (symbolic coefficients allowed).
    if let Some(p) = sym_poly_in(arena, body, var) {
        return Some(faulhaber_sym(arena, &p, lo, hi));
    }

    // 4. Rational function of k.
    if let Some(r) = rational_sum_finite(arena, body, var, lo, hi) {
        return Some(r);
    }

    // 5. Binomial identities.
    if let Some(r) = binomial_sum(arena, body, var, lo, hi) {
        return Some(r);
    }

    // 6. Geometric / arithmetico-geometric.
    if let Some(r) = geometric_poly_finite(arena, body, var, lo, hi) {
        return Some(r);
    }

    // 7. Gosper.
    if let Some(r) = gosper::gosper_sum(arena, body, var, lo, hi) {
        tracing::debug!("summation: Gosper succeeded");
        return Some(r);
    }

    None
}

// ═══════════════════════════════════════════════════════════════════════════
// Polynomials in k with symbolic coefficients
// ═══════════════════════════════════════════════════════════════════════════

/// Decompose `expr` (after expansion) as `Σ c_j·var^j` where each `c_j` is
/// free of `var`.  Returns `None` if `expr` is not polynomial in `var`.
pub(crate) fn sym_poly_in(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
) -> Option<Vec<(usize, ExprId)>> {
    let expanded = arena.expand_expr(expr);
    let terms = add_terms(arena, expanded);
    let mut acc: Vec<(usize, Vec<ExprId>)> = Vec::new();
    for t in terms {
        let (power, coeff) = monomial_in(arena, t, var)?;
        if let Some(slot) = acc.iter_mut().find(|(p, _)| *p == power) {
            slot.1.push(coeff);
        } else {
            acc.push((power, vec![coeff]));
        }
    }
    acc.sort_by_key(|(p, _)| *p);
    let mut out = Vec::with_capacity(acc.len());
    for (p, coeffs) in acc {
        let c = add_all(arena, &coeffs);
        let c = eval::eval(arena, c);
        if !arena.is_zero_structural(c) {
            out.push((p, c));
        }
    }
    Some(out)
}

/// `t = c·var^p` with `c` free of `var`.
fn monomial_in(arena: &mut Arena, t: ExprId, var: ExprId) -> Option<(usize, ExprId)> {
    if !depends_on(arena, t, var) {
        return Some((0, t));
    }
    if t == var {
        return Some((1, arena.one));
    }
    match arena.node(t).clone() {
        ExprNode::Pow(base, exp) if base == var => {
            let n = as_i64(arena, exp)?;
            if n < 0 {
                return None;
            }
            Some((n as usize, arena.one))
        }
        ExprNode::Mul(ref factors) => {
            let mut power = 0usize;
            let mut consts = Vec::new();
            let mut seen_var = false;
            for &f in factors.iter() {
                if !depends_on(arena, f, var) {
                    consts.push(f);
                } else if f == var {
                    if seen_var {
                        return None;
                    }
                    seen_var = true;
                    power = 1;
                } else if let ExprNode::Pow(base, exp) = arena.node(f).clone()
                    && base == var
                {
                    let n = as_i64(arena, exp)?;
                    if n < 0 || seen_var {
                        return None;
                    }
                    seen_var = true;
                    power = n as usize;
                } else {
                    return None;
                }
            }
            let c = mul_all(arena, &consts);
            Some((power, c))
        }
        _ => None,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Faulhaber
// ═══════════════════════════════════════════════════════════════════════════

/// Coefficients (ascending) of the Faulhaber polynomial
/// `S_p(n) = Σ_{k=1}^{n} k^p` using `B₁ = +1/2`:
///
/// `S_p(n) = 1/(p+1) · Σ_{j=0}^{p} C(p+1, j) · B⁺_j · n^{p+1−j}`.
pub(crate) fn faulhaber_coefficients(p: usize) -> Poly {
    let mut coeffs = vec![Rat::zero(); p + 2];
    let inv = Rat::one() / rat_i(p as i64 + 1);
    for j in 0..=p {
        let mut bj = bernoulli(j);
        if j == 1 {
            bj = -bj; // B₁⁺ = +1/2
        }
        if bj.is_zero() {
            continue;
        }
        let c = Rat::from_integer(binomial_big((p + 1) as u64, j as u64));
        let power = p + 1 - j;
        coeffs[power] += &inv * c * bj;
    }
    Poly::from_coeffs(coeffs)
}

/// `Σ_{k=1}^{n} k^p` as an expression in `n`.
pub(crate) fn faulhaber_from_one(arena: &mut Arena, p: usize, n: ExprId) -> ExprId {
    let poly = faulhaber_coefficients(p);
    poly_at(arena, &poly, n)
}

/// Evaluate a rational-coefficient polynomial at an arbitrary expression.
fn poly_at(arena: &mut Arena, poly: &Poly, x: ExprId) -> ExprId {
    let mut terms = Vec::new();
    for (i, c) in poly.coeffs().iter().enumerate() {
        if c.is_zero() {
            continue;
        }
        let ce = rat_expr(arena, c.clone());
        let t = if i == 0 {
            ce
        } else {
            let xi = pow_rat(arena, x, &rat_i(i as i64));
            arena.mul(&[ce, xi])
        };
        terms.push(t);
    }
    let s = add_all(arena, &terms);
    eval::eval(arena, s)
}

/// `Σ_{k=lo}^{hi} P(k)` for a rational-coefficient polynomial `P`.
fn faulhaber_poly(arena: &mut Arena, poly: &Poly, lo: ExprId, hi: ExprId) -> ExprId {
    let sym: Vec<(usize, ExprId)> = poly
        .coeffs()
        .iter()
        .enumerate()
        .filter(|(_, c)| !c.is_zero())
        .map(|(i, c)| (i, rat_expr(arena, c.clone())))
        .collect();
    faulhaber_sym(arena, &sym, lo, hi)
}

/// `Σ_{k=lo}^{hi} Σ_j c_j k^j = Σ_j c_j (S_j(hi) − S_j(lo − 1))`.
fn faulhaber_sym(arena: &mut Arena, poly: &[(usize, ExprId)], lo: ExprId, hi: ExprId) -> ExprId {
    let lo_is_one = as_i64(arena, lo) == Some(1);
    let one = arena.one;
    let lo_m1 = arena.sub(lo, one);
    let lo_m1 = eval::eval(arena, lo_m1);
    let mut terms = Vec::new();
    for &(p, c) in poly {
        let s_hi = faulhaber_from_one(arena, p, hi);
        let s = if lo_is_one {
            s_hi
        } else {
            let s_lo = faulhaber_from_one(arena, p, lo_m1);
            arena.sub(s_hi, s_lo)
        };
        terms.push(arena.mul(&[c, s]));
    }
    let total = add_all(arena, &terms);
    let expanded = arena.expand_expr(total);
    eval::eval(arena, expanded)
}

// ═══════════════════════════════════════════════════════════════════════════
// Rational functions of k: partial fractions + harmonic / digamma
// ═══════════════════════════════════════════════════════════════════════════

/// A pole term `c·(k + β)^(−m)`.
#[derive(Debug, Clone)]
struct PoleTerm {
    c: Rat,
    beta: Rat,
    m: u32,
}

/// Decompose a rational function of `var` (rational coefficients) into a
/// polynomial part and pole terms.  Returns `None` if `body` is not a
/// rational function with a non-constant denominator, or has a pole term
/// that is not of the form `c/(αk+β)^m`.
fn rational_decompose(
    arena: &mut Arena,
    body: ExprId,
    var: ExprId,
) -> Option<(Poly, Vec<PoleTerm>)> {
    let body = combine_fractions(arena, body);
    let (n, d) = polybridge::as_numer_denom(arena, body);
    let np = polybridge::expr_to_poly(arena, n, var)?;
    let dp = polybridge::expr_to_poly(arena, d, var)?;
    if dp.is_zero() || dp.is_constant() || np.is_zero() {
        return None;
    }
    let decomposed = apart::apart(arena, body, var);
    let terms = add_terms(arena, decomposed);
    let mut poly_part = Poly::zero();
    let mut poles = Vec::new();
    for t in terms {
        if let Some(p) = polybridge::expr_to_poly(arena, t, var) {
            poly_part = &poly_part + &p;
            continue;
        }
        poles.push(as_pole_term(arena, t, var)?);
    }
    Some((poly_part, poles))
}

fn as_pole_term(arena: &mut Arena, t: ExprId, var: ExprId) -> Option<PoleTerm> {
    let (c, rest) = arena.as_coeff_term(t);
    let ExprNode::Pow(base, exp) = arena.node(rest).clone() else {
        return None;
    };
    let e = as_i64(arena, exp)?;
    if e >= 0 {
        return None;
    }
    let m = (-e) as u32;
    let (alpha, beta0) = linear_in(arena, base, var)?;
    // c·(αk+β')^(−m) = c·α^(−m)·(k + β'/α)^(−m)
    let coeff = c * rat_pow_i(&alpha, -(m as i64));
    Some(PoleTerm {
        c: coeff,
        beta: beta0 / alpha,
        m,
    })
}

/// Group poles by `(m, frac(β))`; each group sorted by `β`.
fn group_poles(poles: &[PoleTerm]) -> Vec<Vec<PoleTerm>> {
    let mut groups: Vec<Vec<PoleTerm>> = Vec::new();
    for p in poles {
        let fp = frac_part(&p.beta);
        if let Some(g) = groups
            .iter_mut()
            .find(|g| g[0].m == p.m && frac_part(&g[0].beta) == fp)
        {
            g.push(p.clone());
        } else {
            groups.push(vec![p.clone()]);
        }
    }
    for g in &mut groups {
        g.sort_by(|a, b| a.beta.cmp(&b.beta));
    }
    groups
}

/// `G_m(x)` — the "partial sum function" for `Σ 1/(k+β)^m`:
/// `harmonic(x)` for integer offsets and `m = 1`, `digamma(x+1)` otherwise.
fn partial_sum_fn(arena: &mut Arena, x: ExprId, integer_offsets: bool) -> ExprId {
    if integer_offsets {
        arena.harmonic(x)
    } else {
        let one = arena.one;
        let x1 = arena.add(&[x, one]);
        arena.digamma(x1)
    }
}

/// `T(N) = Σ_i c_i G_m(N + β_i)` re-expressed relative to the smallest `β`
/// in the group, so that integer-shifted poles telescope exactly.
///
/// Poles more than [`MAX_TELESCOPE_SHIFT`] beyond the smallest `β` are not
/// expanded term by term; for `m = 1` they keep their own
/// `harmonic` / `digamma` value instead (still exact, just less compact).
/// Returns `None` if the group needs a generalised harmonic number
/// (`m ≥ 2` with non-zero total coefficient, or a far `m ≥ 2` pole).
fn pole_group_partial(arena: &mut Arena, group: &[PoleTerm], n_expr: ExprId) -> Option<ExprId> {
    let m = group[0].m;
    let b0 = group[0].beta.clone();
    let integer_offsets = b0.is_integer();
    let mut terms = Vec::new();
    let mut csum = Rat::zero();
    let mut far = Vec::new();
    for p in group {
        let d = (&p.beta - &b0).to_integer().to_i64()?;
        if d > MAX_TELESCOPE_SHIFT {
            far.push(p);
            continue;
        }
        csum += &p.c;
        for j in 1..=d {
            let shift = &b0 + rat_i(j);
            let x = add_rat(arena, n_expr, &shift);
            let inv = pow_rat(arena, x, &rat_i(-(m as i64)));
            let ce = rat_expr(arena, p.c.clone());
            terms.push(arena.mul(&[ce, inv]));
        }
    }
    if !csum.is_zero() {
        if m != 1 {
            return None;
        }
        let x = add_rat(arena, n_expr, &b0);
        let g = partial_sum_fn(arena, x, integer_offsets);
        let ce = rat_expr(arena, csum);
        terms.push(arena.mul(&[ce, g]));
    }
    for p in far {
        if m != 1 {
            return None;
        }
        let x = add_rat(arena, n_expr, &p.beta);
        let g = partial_sum_fn(arena, x, integer_offsets);
        let ce = rat_expr(arena, p.c.clone());
        terms.push(arena.mul(&[ce, g]));
    }
    Some(add_all(arena, &terms))
}

fn rational_sum_finite(
    arena: &mut Arena,
    body: ExprId,
    var: ExprId,
    lo: ExprId,
    hi: ExprId,
) -> Option<ExprId> {
    let (poly_part, poles) = rational_decompose(arena, body, var)?;
    let mut parts = Vec::new();
    if !poly_part.is_zero() {
        parts.push(faulhaber_poly(arena, &poly_part, lo, hi));
    }
    let one = arena.one;
    let lo_m1 = arena.sub(lo, one);
    let lo_m1 = eval::eval(arena, lo_m1);
    for group in group_poles(&poles) {
        let t_hi = pole_group_partial(arena, &group, hi)?;
        let t_lo = pole_group_partial(arena, &group, lo_m1)?;
        parts.push(arena.sub(t_hi, t_lo));
    }
    let total = add_all(arena, &parts);
    Some(eval::eval(arena, total))
}

/// Infinite rational sums `Σ_{k=lo}^{∞} N(k)/D(k)`.
fn rational_sum_infinite(
    arena: &mut Arena,
    body: ExprId,
    var: ExprId,
    lo: ExprId,
) -> Option<SumOutcome> {
    let (poly_part, poles) = rational_decompose(arena, body, var)?;
    if !poly_part.is_zero() {
        let lc = poly_part.leading_coeff().cloned().unwrap_or_else(Rat::zero);
        return Some(SumOutcome::Divergent(infinity_of_sign(
            arena,
            Some(lc.is_positive()),
        )));
    }
    // Simple poles: total coefficient must vanish for convergence.
    let simple_total: Rat = poles.iter().filter(|p| p.m == 1).map(|p| p.c.clone()).sum();
    if !simple_total.is_zero() {
        return Some(SumOutcome::Divergent(infinity_of_sign(
            arena,
            Some(simple_total.is_positive()),
        )));
    }
    // Integer-offset simple poles: use harmonic numbers only if their
    // own coefficient sum vanishes (so Euler's γ cancels); otherwise digamma.
    let int_simple_total: Rat = poles
        .iter()
        .filter(|p| p.m == 1 && p.beta.is_integer())
        .map(|p| p.c.clone())
        .sum();
    let use_harmonic = int_simple_total.is_zero();

    let mut parts = Vec::new();
    for group in group_poles(&poles) {
        let m = group[0].m;
        let b0 = group[0].beta.clone();
        let csum: Rat = group.iter().map(|p| p.c.clone()).sum();
        if m == 1 {
            // Σ_{k=lo}^{∞} Σ_i c_i/(k+β_i) = −Σ_i c_i ψ(lo + β_i)   (Σ c_i = 0 overall)
            // Within the group, ψ(x + d) = ψ(x) + Σ_{j<d} 1/(x+j), so only the
            // group's total coefficient multiplies a ψ / harmonic value.
            // Poles shifted by more than MAX_TELESCOPE_SHIFT keep their own ψ.
            let x0 = add_rat(arena, lo, &b0);
            let psi = |arena: &mut Arena, x: ExprId, integer: bool| -> ExprId {
                if use_harmonic && integer {
                    // ψ(x) = H_{x−1} − γ; γ cancels overall.
                    let one = arena.one;
                    let xm1 = arena.sub(x, one);
                    let xm1 = eval::eval(arena, xm1);
                    arena.harmonic(xm1)
                } else {
                    arena.digamma(x)
                }
            };
            let mut near_sum = Rat::zero();
            for p in &group {
                let d = (&p.beta - &b0).to_integer().to_i64()?;
                if d > MAX_TELESCOPE_SHIFT {
                    let x = add_rat(arena, lo, &p.beta);
                    let g = psi(arena, x, p.beta.is_integer());
                    let ce = rat_expr(arena, -p.c.clone());
                    parts.push(arena.mul(&[ce, g]));
                    continue;
                }
                near_sum += &p.c;
                for j in 0..d {
                    let x = add_rat(arena, x0, &rat_i(j));
                    let inv = pow_rat(arena, x, &(-Rat::one()));
                    let ce = rat_expr(arena, -p.c.clone());
                    parts.push(arena.mul(&[ce, inv]));
                }
            }
            if !near_sum.is_zero() {
                let g = psi(arena, x0, b0.is_integer());
                let ce = rat_expr(arena, -near_sum);
                parts.push(arena.mul(&[ce, g]));
            }
        } else if csum.is_zero() {
            // Pure telescoping: −Σ_i c_i Σ_{j=0}^{d_i−1} (lo + β_0 + j)^(−m)
            for p in &group {
                let d = (&p.beta - &b0).to_integer().to_i64()?;
                if d > MAX_TELESCOPE_SHIFT {
                    return None;
                }
                for j in 0..d {
                    let shift = &b0 + rat_i(j);
                    let x = add_rat(arena, lo, &shift);
                    let inv = pow_rat(arena, x, &rat_i(-(m as i64)));
                    let ce = rat_expr(arena, -p.c.clone());
                    parts.push(arena.mul(&[ce, inv]));
                }
            }
        } else {
            // Each term is a Hurwitz zeta value ζ(m, lo+β).
            for p in &group {
                let x = add_rat(arena, lo, &p.beta);
                let q = eval::eval(arena, x);
                let qr = as_rat(arena, q)?;
                let z = hurwitz_zeta_closed(arena, m as usize, &qr)?;
                let ce = rat_expr(arena, p.c.clone());
                parts.push(arena.mul(&[ce, z]));
            }
        }
    }
    let total = add_all(arena, &parts);
    Some(SumOutcome::Closed(eval::eval(arena, total)))
}

// ═══════════════════════════════════════════════════════════════════════════
// Zeta-type constants
// ═══════════════════════════════════════════════════════════════════════════

/// `ζ(2m) = (−1)^{m+1} B_{2m} (2π)^{2m} / (2·(2m)!)` — returns the rational
/// factor `r` such that `ζ(2m) = r·π^{2m}`.
pub(crate) fn zeta_even_rational(m: usize) -> Rat {
    let two_m = 2 * m;
    let b = bernoulli(two_m);
    let sign = if m % 2 == 1 { Rat::one() } else { -Rat::one() };
    let two_pow = rat_pow_i(&rat_i(2), two_m as i64);
    let denom = Rat::from_integer(factorial_big(two_m as u64)) * rat_i(2);
    sign * b * two_pow / denom
}

/// Euler numbers `E_0, E_2, E_4, …` (E_{2n} for the given `n`).
pub(crate) fn euler_number(n: usize) -> BigInt {
    // Σ_{k=0}^{n} C(2n, 2k) E_{2k} = 0 for n ≥ 1, E_0 = 1.
    let mut e = vec![BigInt::one()];
    for nn in 1..=n {
        let mut acc = BigInt::zero();
        for (k, ek) in e.iter().enumerate() {
            acc += binomial_big(2 * nn as u64, 2 * k as u64) * ek;
        }
        e.push(-acc);
    }
    e[n].clone()
}

/// Exact value of `ζ(p)` for integer `p ≥ 2`.
///
/// Even `p` gives the elementary closed form (`ζ(2) = π²/6`, `ζ(4) = π⁴/90`,
/// …, via Bernoulli numbers); odd `p ≥ 3` has no elementary closed form and
/// is returned as the `Zeta(p)` node (`ζ(3)` is Apéry's constant).
/// Returns `None` for `p < 2` (the harmonic series diverges).
pub(crate) fn zeta_value(arena: &mut Arena, p: usize) -> Option<ExprId> {
    if p < 2 {
        return None;
    }
    if p % 2 == 1 {
        let pe = arena.int(p as i64);
        return Some(arena.zeta(pe));
    }
    let r = zeta_even_rational(p / 2);
    let re = rat_expr(arena, r);
    let pi = arena.pi;
    let pip = pow_rat(arena, pi, &rat_i(p as i64));
    let v = arena.mul(&[re, pip]);
    Some(eval::eval(arena, v))
}

/// Dirichlet eta `η(p) = Σ_{k≥1} (−1)^{k+1}/k^p = (1 − 2^{1−p}) ζ(p)`; `η(1) = ln 2`.
fn eta_value(arena: &mut Arena, p: usize) -> Option<ExprId> {
    if p == 1 {
        let two = arena.int(2);
        return Some(arena.ln(two));
    }
    let z = zeta_value(arena, p)?;
    let factor = Rat::one() - rat_pow_i(&rat_i(2), 1 - p as i64);
    let fe = rat_expr(arena, factor);
    let v = arena.mul(&[fe, z]);
    Some(eval::eval(arena, v))
}

/// Dirichlet beta `β(p) = Σ_{k≥0} (−1)^k/(2k+1)^p`; closed form for odd `p`:
/// `β(2m+1) = (−1)^m E_{2m} π^{2m+1} / (4^{m+1} (2m)!)`, and `β(2) = G`
/// (Catalan's constant).  Even `p ≥ 4` has no known closed form.
fn dirichlet_beta_value(arena: &mut Arena, p: usize) -> Option<ExprId> {
    if p == 2 {
        return Some(arena.catalan);
    }
    if p.is_multiple_of(2) {
        return None;
    }
    let m = (p - 1) / 2;
    let e = euler_number(m);
    let sign = if m.is_multiple_of(2) {
        BigInt::one()
    } else {
        -BigInt::one()
    };
    let denom = rat_pow_i(&rat_i(4), m as i64 + 1) * Rat::from_integer(factorial_big(2 * m as u64));
    let r = Rat::from_integer(sign * e) / denom;
    let re = rat_expr(arena, r);
    let pi = arena.pi;
    let pip = pow_rat(arena, pi, &rat_i(p as i64));
    let v = arena.mul(&[re, pip]);
    Some(eval::eval(arena, v))
}

/// Hurwitz zeta `ζ(m, q) = Σ_{k≥0} 1/(k+q)^m` for integer `m ≥ 2` and the
/// rational offsets we can handle exactly: positive integers and
/// half-integers, with `m` even.
fn hurwitz_zeta_closed(arena: &mut Arena, m: usize, q: &Rat) -> Option<ExprId> {
    if m < 2 || !q.is_positive() {
        return None;
    }
    if q.is_integer() {
        let qi = q.to_integer().to_i64()?;
        if qi > MAX_TELESCOPE_SHIFT {
            return None;
        }
        let z = zeta_value(arena, m)?;
        let mut acc = Rat::zero();
        for j in 1..qi {
            acc += rat_pow_i(&rat_i(j), -(m as i64));
        }
        let ce = rat_expr(arena, -acc);
        let v = arena.add(&[z, ce]);
        return Some(eval::eval(arena, v));
    }
    // half-integer q = n + 1/2 with n ≥ 0: ζ(m, 1/2) = (2^m − 1) ζ(m)
    let two_q = q * rat_i(2);
    if two_q.is_integer() {
        let n = ((q - Rat::new(BigInt::one(), BigInt::from(2))).to_integer()).to_i64()?;
        if !(0..=MAX_TELESCOPE_SHIFT).contains(&n) {
            return None;
        }
        let z = zeta_value(arena, m)?;
        let factor = rat_pow_i(&rat_i(2), m as i64) - Rat::one();
        let mut acc = Rat::zero();
        for j in 0..n {
            let x = rat_i(j) + Rat::new(BigInt::one(), BigInt::from(2));
            acc += rat_pow_i(&x, -(m as i64));
        }
        let fe = rat_expr(arena, factor);
        let ce = rat_expr(arena, -acc);
        let fz = arena.mul(&[fe, z]);
        let v = arena.add(&[fz, ce]);
        return Some(eval::eval(arena, v));
    }
    None
}

// ═══════════════════════════════════════════════════════════════════════════
// Telescoping  g(k) − g(k+d)
// ═══════════════════════════════════════════════════════════════════════════

/// If `terms = [A, B]` with `B(k+d) = −A(k)` (or vice versa) for some
/// `d ∈ {1,2,3}`, returns `(g, d)` such that `body = g(k) − g(k+d)`.
fn telescoping_form(arena: &mut Arena, terms: &[ExprId], var: ExprId) -> Option<(ExprId, i64)> {
    if terms.len() != 2 {
        return None;
    }
    let (a, b) = (terms[0], terms[1]);
    for d in 1..=3i64 {
        let de = arena.int(d);
        let shifted_var = arena.add(&[var, de]);
        let b_shift = subs::subs(arena, b, var, shifted_var);
        let test = arena.add(&[a, b_shift]);
        if is_zero_expr(arena, test) {
            return Some((b, d));
        }
        let a_shift = subs::subs(arena, a, var, shifted_var);
        let test = arena.add(&[b, a_shift]);
        if is_zero_expr(arena, test) {
            return Some((a, d));
        }
    }
    None
}

fn telescoping_finite(
    arena: &mut Arena,
    terms: &[ExprId],
    var: ExprId,
    lo: ExprId,
    hi: ExprId,
) -> Option<ExprId> {
    let (g, d) = telescoping_form(arena, terms, var)?;
    // Σ_{k=lo}^{hi} [g(k) − g(k+d)] = Σ_{j=0}^{d−1} [g(lo+j) − g(hi+1+j)]
    let mut parts = Vec::new();
    for j in 0..d {
        let je = arena.int(j);
        let lo_j = arena.add(&[lo, je]);
        let hi_j = arena.int(j + 1);
        let hi_j = arena.add(&[hi, hi_j]);
        let g_lo = subs::subs(arena, g, var, lo_j);
        let g_hi = subs::subs(arena, g, var, hi_j);
        parts.push(arena.sub(g_lo, g_hi));
    }
    let total = add_all(arena, &parts);
    Some(eval::eval(arena, total))
}

/// Infinite telescoping: needs `lim_{N→∞} g(N)`.  Only rational-function
/// tails are trusted (their limit is computed exactly here).
fn telescoping_infinite(
    arena: &mut Arena,
    terms: &[ExprId],
    var: ExprId,
    lo: ExprId,
) -> Option<SumOutcome> {
    let (g, d) = telescoping_form(arena, terms, var)?;
    let limit = limit_at_infinity(arena, g, var)?;
    let mut parts = Vec::new();
    for j in 0..d {
        let je = arena.int(j);
        let lo_j = arena.add(&[lo, je]);
        let g_lo = subs::subs(arena, g, var, lo_j);
        parts.push(arena.sub(g_lo, limit));
    }
    let total = add_all(arena, &parts);
    Some(SumOutcome::Closed(eval::eval(arena, total)))
}

/// Exact `lim_{k→∞} f(k)` for the shapes we can decide without the general
/// limit engine: rational functions of `k` (finite limits only), sums of
/// such terms, and hypergeometric-type terms whose exact Stirling growth
/// analysis shows they tend to zero (`P(k)·r^k` with `|r| < 1`,
/// `1/(k+1)!`, `k!/k^k`, …).
pub(crate) fn limit_at_infinity(arena: &mut Arena, f: ExprId, var: ExprId) -> Option<ExprId> {
    if !depends_on(arena, f, var) {
        return Some(f);
    }
    if let Some(l) = rational_limit_at_infinity(arena, f, var) {
        return l.map(|r| rat_expr(arena, r));
    }
    if let ExprNode::Add(ref terms) = arena.node(f).clone() {
        let terms: Vec<ExprId> = terms.to_vec();
        let mut limits = Vec::with_capacity(terms.len());
        for t in terms {
            limits.push(limit_at_infinity(arena, t, var)?);
        }
        let s = add_all(arena, &limits);
        return Some(eval::eval(arena, s));
    }
    // Constant factors: c · g(k)
    if let ExprNode::Mul(ref factors) = arena.node(f).clone() {
        let factors: Vec<ExprId> = factors.to_vec();
        let (consts, varf): (Vec<ExprId>, Vec<ExprId>) = factors
            .iter()
            .copied()
            .partition(|&g| !depends_on(arena, g, var));
        if !consts.is_empty() && !varf.is_empty() {
            let inner = mul_all(arena, &varf);
            let l = limit_at_infinity(arena, inner, var)?;
            let mut all = consts;
            all.push(l);
            let p = arena.mul(&all);
            return Some(eval::eval(arena, p));
        }
    }
    // Growth analysis needs monomial factors: replace every polynomial
    // factor `P(k)` (rational coefficients) by its leading term `lc·k^d`,
    // which has the same asymptotic growth.
    let dominant = dominant_factor_form(arena, f, var);
    if crate::calculus::convergence::growth_exponents(arena, dominant, var)
        .and_then(|g| g.tends_to_zero())
        == Some(true)
    {
        return Some(arena.zero);
    }
    None
}

/// Replace polynomial factors of `f` (with rational coefficients, degree
/// ≥ 1) by their leading monomials.  Only the growth rate is preserved.
fn dominant_factor_form(arena: &mut Arena, f: ExprId, var: ExprId) -> ExprId {
    let factors = mul_factors(arena, f);
    let mut out = Vec::with_capacity(factors.len());
    let mut changed = false;
    for g in factors {
        let replaced = match arena.node(g).clone() {
            ExprNode::Add(_) => polybridge::expr_to_poly(arena, g, var).and_then(|p| {
                let d = p.degree()?;
                let lc = p.leading_coeff()?.clone();
                let ce = rat_expr(arena, lc);
                let kd = pow_rat(arena, var, &rat_i(d as i64));
                Some(arena.mul(&[ce, kd]))
            }),
            ExprNode::Pow(base, exp)
                if matches!(arena.node(base), ExprNode::Add(_)) && !depends_on(arena, exp, var) =>
            {
                polybridge::expr_to_poly(arena, base, var).and_then(|p| {
                    let d = p.degree()?;
                    let lc = p.leading_coeff()?.clone();
                    let ce = rat_expr(arena, lc);
                    let kd = pow_rat(arena, var, &rat_i(d as i64));
                    let mono = arena.mul(&[ce, kd]);
                    Some(arena.pow(mono, exp))
                })
            }
            _ => None,
        };
        match replaced {
            Some(r) => {
                changed = true;
                out.push(r);
            }
            None => out.push(g),
        }
    }
    if changed {
        let p = mul_all(arena, &out);
        eval::eval(arena, p)
    } else {
        f
    }
}

/// `lim_{k→∞}` of a rational function with rational coefficients.
/// `Some(None)` means the limit is infinite.
fn rational_limit_at_infinity(arena: &mut Arena, f: ExprId, var: ExprId) -> Option<Option<Rat>> {
    let (n, d) = polybridge::as_numer_denom(arena, f);
    let np = polybridge::expr_to_poly(arena, n, var)?;
    let dp = polybridge::expr_to_poly(arena, d, var)?;
    if dp.is_zero() {
        return None;
    }
    if np.is_zero() {
        return Some(Some(Rat::zero()));
    }
    let dn = np.degree()?;
    let dd = dp.degree()?;
    if dn < dd {
        Some(Some(Rat::zero()))
    } else if dn == dd {
        Some(Some(np.coeff(dn) / dp.coeff(dd)))
    } else {
        Some(None)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Term shape analysis (hypergeometric monomials)
// ═══════════════════════════════════════════════════════════════════════════

/// Normalised multiplicative structure of a summand `t(k)`:
///
/// ```text
/// t(k) = constant · (−1)^k[if alternating] · numeric_base^k · Π baseᵢ^(aᵢ·k)
///        · Π (k + βⱼ)^pⱼ · Π ((αₗ k + βₗ)!)^eₗ · Π C(nᵣ, k)^eᵣ
/// ```
///
/// Every factor is either `k`-free (folded into `constant`) or one of the
/// recognised shapes; otherwise [`term_shape`] returns `None`.
#[derive(Debug, Clone)]
pub(crate) struct TermShape {
    /// Product of all `k`-free factors (including `base^b` leftovers).
    pub(crate) constant: ExprId,
    /// A `(−1)^k` alternation is present.
    pub(crate) alternating: bool,
    /// Exact rational `Π rᵢ^{aᵢ}` over numeric bases with integer `aᵢ`.
    pub(crate) numeric_base: Rat,
    /// Symbolic geometric bases `(base, a)` meaning `base^(a·k)`.
    pub(crate) bases: Vec<(ExprId, Rat)>,
    /// Monic linear powers `(β, p)` meaning `(k + β)^p`, sorted by `β`.
    pub(crate) lin_pows: Vec<(Rat, Rat)>,
    /// Factorial factors `(α, β, e)` meaning `((α·k + β)!)^e`, sorted.
    pub(crate) facts: Vec<(Rat, Rat, i64)>,
    /// Binomial factors `(n, e)` meaning `C(n, k)^e` with `n` free of `k`.
    pub(crate) binomials: Vec<(ExprId, i64)>,
}

impl TermShape {
    fn is_pure_lin_pows(&self) -> bool {
        self.bases.is_empty()
            && self.numeric_base.is_one()
            && self.facts.is_empty()
            && self.binomials.is_empty()
    }

    /// Combined geometric base `Y = numeric_base · Π baseᵢ^{aᵢ}` (with the
    /// sign alternation folded in), or `None` when there is none.
    pub(crate) fn geometric_base(&self, arena: &mut Arena) -> Option<ExprId> {
        if self.bases.is_empty() && self.numeric_base.is_one() && !self.alternating {
            return None;
        }
        let mut factors = Vec::new();
        let mut num = self.numeric_base.clone();
        if self.alternating {
            num = -num;
        }
        if !num.is_one() {
            factors.push(rat_expr(arena, num));
        }
        for (b, a) in &self.bases {
            factors.push(pow_rat(arena, *b, a));
        }
        let y = mul_all(arena, &factors);
        Some(eval::eval(arena, y))
    }
}

/// Analyse the multiplicative structure of `body` with respect to `var`.
pub(crate) fn term_shape(arena: &mut Arena, body: ExprId, var: ExprId) -> Option<TermShape> {
    let mut shape = TermShape {
        constant: arena.one,
        alternating: false,
        numeric_base: Rat::one(),
        bases: Vec::new(),
        lin_pows: Vec::new(),
        facts: Vec::new(),
        binomials: Vec::new(),
    };
    let mut consts: Vec<ExprId> = Vec::new();
    let factors = mul_factors(arena, body);
    for f in factors {
        shape_factor(arena, f, var, &Rat::one(), &mut shape, &mut consts)?;
    }
    shape.constant = mul_all(arena, &consts);
    shape.constant = eval::eval(arena, shape.constant);
    // Merge & sort.
    shape.lin_pows = merge_pairs(shape.lin_pows);
    shape.facts.sort_by_key(|a| (a.0.clone(), a.1.clone()));
    let mut merged_facts: Vec<(Rat, Rat, i64)> = Vec::new();
    for (a, b, e) in shape.facts {
        if let Some(last) = merged_facts.last_mut()
            && last.0 == a
            && last.1 == b
        {
            last.2 += e;
        } else {
            merged_facts.push((a, b, e));
        }
    }
    merged_facts.retain(|f| f.2 != 0);
    shape.facts = merged_facts;
    let mut merged_bases: Vec<(ExprId, Rat)> = Vec::new();
    for (b, a) in shape.bases {
        if let Some(slot) = merged_bases.iter_mut().find(|(bb, _)| *bb == b) {
            slot.1 += a;
        } else {
            merged_bases.push((b, a));
        }
    }
    merged_bases.retain(|(_, a)| !a.is_zero());
    merged_bases.sort_by_key(|(b, _)| b.0);
    shape.bases = merged_bases;
    let mut merged_bin: Vec<(ExprId, i64)> = Vec::new();
    for (n, e) in shape.binomials {
        if let Some(slot) = merged_bin.iter_mut().find(|(nn, _)| *nn == n) {
            slot.1 += e;
        } else {
            merged_bin.push((n, e));
        }
    }
    merged_bin.retain(|(_, e)| *e != 0);
    shape.binomials = merged_bin;
    Some(shape)
}

fn merge_pairs(mut v: Vec<(Rat, Rat)>) -> Vec<(Rat, Rat)> {
    v.sort_by(|a, b| a.0.cmp(&b.0));
    let mut out: Vec<(Rat, Rat)> = Vec::new();
    for (b, p) in v {
        if let Some(last) = out.last_mut()
            && last.0 == b
        {
            last.1 += p;
        } else {
            out.push((b, p));
        }
    }
    out.retain(|(_, p)| !p.is_zero());
    out
}

/// Classify one multiplicative factor raised to the rational power `outer`.
fn shape_factor(
    arena: &mut Arena,
    f: ExprId,
    var: ExprId,
    outer: &Rat,
    shape: &mut TermShape,
    consts: &mut Vec<ExprId>,
) -> Option<()> {
    if !depends_on(arena, f, var) {
        let c = pow_rat(arena, f, outer);
        consts.push(c);
        return Some(());
    }
    if f == var {
        shape.lin_pows.push((Rat::zero(), outer.clone()));
        return Some(());
    }
    match arena.node(f).clone() {
        ExprNode::Pow(base, exp) => {
            if !depends_on(arena, exp, var) {
                let p = as_rat(arena, exp)?;
                let total = outer * p;
                shape_factor(arena, base, var, &total, shape, consts)
            } else {
                if depends_on(arena, base, var) {
                    return None;
                }
                let (a, b) = linear_in_sym(arena, exp, var)?;
                let a = a * outer;
                let b = scale_const(arena, b, outer);
                shape_geometric(arena, base, &a, b, shape, consts)
            }
        }
        ExprNode::Exp(arg) => {
            let (a, b) = linear_in_sym(arena, arg, var)?;
            let e = arena.e_const;
            let a = a * outer;
            let b = scale_const(arena, b, outer);
            shape_geometric(arena, e, &a, b, shape, consts)
        }
        ExprNode::Factorial(arg) => {
            let e = outer.to_integer();
            if !outer.is_integer() {
                return None;
            }
            let (alpha, beta) = linear_in(arena, arg, var)?;
            shape.facts.push((alpha, beta, e.to_i64()?));
            Some(())
        }
        ExprNode::Gamma(arg) => {
            if !outer.is_integer() {
                return None;
            }
            let (alpha, beta) = linear_in(arena, arg, var)?;
            shape
                .facts
                .push((alpha, beta - Rat::one(), outer.to_integer().to_i64()?));
            Some(())
        }
        ExprNode::Binomial(n, kk) => {
            if !outer.is_integer() {
                return None;
            }
            let e = outer.to_integer().to_i64()?;
            if kk == var && !depends_on(arena, n, var) {
                shape.binomials.push((n, e));
                return Some(());
            }
            // C(2k, k) = (2k)!/(k!)²
            if kk == var
                && let Some((a, b)) = linear_in(arena, n, var)
            {
                shape.facts.push((a, b, e));
                shape.facts.push((Rat::one(), Rat::zero(), -e));
                let nm = arena.sub(n, var);
                let (a2, b2) = linear_in(arena, nm, var)?;
                shape.facts.push((a2, b2, -e));
                return Some(());
            }
            None
        }
        ExprNode::Mul(ref inner) => {
            let inner: Vec<ExprId> = inner.to_vec();
            for g in inner {
                shape_factor(arena, g, var, outer, shape, consts)?;
            }
            Some(())
        }
        ExprNode::Add(_) => {
            // Polynomial in k with rational coefficients → linear factors.
            let p = polybridge::expr_to_poly(arena, f, var)?;
            shape_polynomial(arena, &p, outer, shape, consts)
        }
        ExprNode::Neg(inner) => {
            let m1 = arena.neg_one;
            let c = pow_rat(arena, m1, outer);
            consts.push(c);
            shape_factor(arena, inner, var, outer, shape, consts)
        }
        _ => None,
    }
}

/// `base^(a·k + b)` with `base` and `b` free of `k` (`b` may be symbolic).
fn shape_geometric(
    arena: &mut Arena,
    base: ExprId,
    a: &Rat,
    b: ExprId,
    shape: &mut TermShape,
    consts: &mut Vec<ExprId>,
) -> Option<()> {
    if a.is_zero() {
        let c = pow_const(arena, base, b);
        consts.push(c);
        return Some(());
    }
    if let Some(r) = as_rat(arena, base) {
        if r.is_zero() {
            return None;
        }
        if a.is_integer() {
            let ai = a.to_integer().to_i64()?;
            if r == -Rat::one() {
                if ai.rem_euclid(2) == 1 {
                    shape.alternating = !shape.alternating;
                }
            } else {
                shape.numeric_base *= rat_pow_i(&r, ai);
            }
            if !arena.is_zero_structural(b) {
                let c = pow_const(arena, base, b);
                consts.push(c);
            }
            return Some(());
        }
        // Non-integer multiple of k (e.g. 2^(k/2)) — keep as symbolic base.
        shape.bases.push((base, a.clone()));
        if !arena.is_zero_structural(b) {
            let c = pow_const(arena, base, b);
            consts.push(c);
        }
        return Some(());
    }
    shape.bases.push((base, a.clone()));
    if !arena.is_zero_structural(b) {
        let c = pow_const(arena, base, b);
        consts.push(c);
    }
    Some(())
}

/// `P(k)^outer` for a rational-coefficient polynomial: factor over ℤ and
/// require every irreducible factor to be linear.
fn shape_polynomial(
    arena: &mut Arena,
    p: &Poly,
    outer: &Rat,
    shape: &mut TermShape,
    consts: &mut Vec<ExprId>,
) -> Option<()> {
    let (content, factors) = p.factor_over_z();
    if content.is_zero() {
        return None;
    }
    if !content.is_one() {
        let c = rat_expr(arena, content);
        let c = pow_rat(arena, c, outer);
        consts.push(c);
    }
    for (fac, mult) in factors {
        if fac.degree() != Some(1) {
            return None;
        }
        let alpha = fac.coeff(1);
        let beta = fac.coeff(0);
        let e = outer * rat_i(mult as i64);
        if !alpha.is_one() {
            if alpha.is_negative() && !e.is_integer() {
                return None;
            }
            let ae = rat_expr(arena, alpha.clone());
            let c = pow_rat(arena, ae, &e);
            consts.push(c);
        }
        shape.lin_pows.push((beta / alpha, e));
    }
    Some(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Binomial sums
// ═══════════════════════════════════════════════════════════════════════════

/// Stirling numbers of the second kind `S(m, j)` for `0 ≤ j ≤ m`.
fn stirling_second_row(m: usize) -> Vec<Rat> {
    // S(0,0) = 1; S(m, j) = j·S(m−1, j) + S(m−1, j−1).
    let mut row = vec![Rat::one()];
    for _ in 0..m {
        let mut next = vec![Rat::zero(); row.len() + 1];
        for (j, s) in row.iter().enumerate() {
            next[j] += rat_i(j as i64) * s;
            next[j + 1] += s;
        }
        row = next;
    }
    row
}

/// Split `k`-free factors into those of the form `b^n` (exponent
/// structurally equal to `n`) and the rest, returning `(Π b, Π rest)`.
///
/// In the binomial identities `n` is the sum's upper bound, hence an
/// integer, so `q^n` may be folded into the closed form's base without a
/// branch-cut concern: `(1 + x)^n·q^n = ((1 + x)·q)^n`.  This is what turns
/// `Σ C(n,k) p^k q^{n−k}` into `(p + q)^n` rather than `(1 + p/q)^n·q^n`.
fn split_pow_n(arena: &mut Arena, factors: &[ExprId], n: ExprId) -> (ExprId, ExprId) {
    let mut bases = Vec::new();
    let mut rest = Vec::new();
    for &f in factors {
        match arena.node(f) {
            ExprNode::Pow(b, e) if *e == n => bases.push(*b),
            _ => rest.push(f),
        }
    }
    let q = mul_all(arena, &bases);
    let r = mul_all(arena, &rest);
    (q, r)
}

/// `e·q` over a common denominator (identity when `q` is 1).
fn scale_and_tidy(arena: &mut Arena, e: ExprId, q: ExprId) -> ExprId {
    let m = if q == arena.one {
        e
    } else {
        arena.mul(&[e, q])
    };
    let t = arena.together_expr(m);
    eval::eval(arena, t)
}

/// `Σ_{k=0}^{n} P(k)·C(n,k)·x^k` for a polynomial `P` (symbolic `k`-free
/// coefficients allowed) via the falling-factorial basis:
/// `k^(j)·C(n,k) = n^(j)·C(n−j, k−j)`, so the sum is
/// `Σ_j s_j·n^(j)·x^j·(1+x)^{n−j}` where `P(k) = Σ_j s_j·k^(j)`.
fn binomial_poly_sum(
    arena: &mut Arena,
    body: ExprId,
    var: ExprId,
    lo: ExprId,
    hi: ExprId,
) -> Option<ExprId> {
    if as_i64(arena, lo) != Some(0) {
        return None;
    }
    let mut consts = Vec::new();
    let mut poly_factors = Vec::new();
    let mut binomial_n: Option<ExprId> = None;
    let mut x_parts = Vec::new();
    for f in mul_factors(arena, body) {
        if !depends_on(arena, f, var) {
            consts.push(f);
            continue;
        }
        match arena.node(f).clone() {
            ExprNode::Binomial(n, kk) if kk == var && !depends_on(arena, n, var) => {
                if binomial_n.is_some() {
                    return None;
                }
                binomial_n = Some(n);
            }
            ExprNode::Pow(base, exp)
                if !depends_on(arena, base, var) && depends_on(arena, exp, var) =>
            {
                let (a, b) = linear_in_sym(arena, exp, var)?;
                x_parts.push(pow_rat(arena, base, &a));
                if !arena.is_zero_structural(b) {
                    consts.push(pow_const(arena, base, b));
                }
            }
            _ => {
                sym_poly_in(arena, f, var)?;
                poly_factors.push(f);
            }
        }
    }
    let n = binomial_n?;
    if poly_factors.is_empty() {
        return None; // the shape table handles the pure cases
    }
    let diff = arena.sub(hi, n);
    let diff = eval::eval(arena, diff);
    if !arena.is_zero_structural(diff) {
        return None;
    }
    let p_expr = mul_all(arena, &poly_factors);
    let monomials = sym_poly_in(arena, p_expr, var)?;
    let deg = monomials.iter().map(|(d, _)| *d).max()?;
    // x (the geometric base) and 1 + x.
    let x = if x_parts.is_empty() {
        arena.one
    } else {
        let xx = mul_all(arena, &x_parts);
        eval::eval(arena, xx)
    };
    if as_rat(arena, x) == Some(-Rat::one()) {
        return None; // (1+x)^{n−j} = 0^{n−j} needs a case split; leave to Gosper
    }
    // Constant factors q^n are absorbed into x^j·(1+x)^{n−j} as (xq)^j·((1+x)q)^{n−j}.
    let (q, rest) = split_pow_n(arena, &consts, n);
    let mut consts = vec![rest];
    let one = arena.one;
    let one_plus_x = arena.add(&[one, x]);
    let one_plus_x = scale_and_tidy(arena, one_plus_x, q);
    let x = scale_and_tidy(arena, x, q);
    // Falling-factorial coefficients s_j = Σ_m c_m S(m, j).
    let mut s: Vec<Vec<ExprId>> = vec![Vec::new(); deg + 1];
    for (m, c) in &monomials {
        let row = stirling_second_row(*m);
        for (j, st) in row.iter().enumerate() {
            if st.is_zero() {
                continue;
            }
            let se = rat_expr(arena, st.clone());
            s[j].push(arena.mul(&[se, *c]));
        }
    }
    let mut terms = Vec::new();
    let mut falling = arena.one; // n^(j)
    for (j, parts) in s.iter().enumerate() {
        if j > 0 {
            let shift = arena.int(-(j as i64 - 1));
            let factor = arena.add(&[n, shift]);
            falling = arena.mul(&[falling, factor]);
        }
        if parts.is_empty() {
            continue;
        }
        let sj = add_all(arena, parts);
        let xj = pow_rat(arena, x, &rat_i(j as i64));
        let nj = arena.int(-(j as i64));
        let n_minus_j = arena.add(&[n, nj]);
        let tail = arena.pow(one_plus_x, n_minus_j);
        terms.push(arena.mul(&[sj, falling, xj, tail]));
    }
    let total = add_all(arena, &terms);
    consts.push(total);
    let result = mul_all(arena, &consts);
    Some(eval::eval(arena, result))
}

fn binomial_sum(
    arena: &mut Arena,
    body: ExprId,
    var: ExprId,
    lo: ExprId,
    hi: ExprId,
) -> Option<ExprId> {
    if let Some(r) = binomial_poly_sum(arena, body, var, lo, hi) {
        return Some(r);
    }
    let shape = term_shape(arena, body, var)?;
    if shape.binomials.len() != 1 || !shape.facts.is_empty() {
        return None;
    }
    let (n, e) = shape.binomials[0];
    // Require lo = 0 and hi = n.
    if as_i64(arena, lo) != Some(0) {
        return None;
    }
    let diff = arena.sub(hi, n);
    let diff = eval::eval(arena, diff);
    if !arena.is_zero_structural(diff) {
        return None;
    }
    let one = arena.one;
    let two = arena.int(2);
    let n_m1 = arena.sub(n, one);
    let n_p1 = arena.add(&[n, one]);
    let x = shape.geometric_base(arena); // includes sign alternation
    // Constant factors q^n (e.g. the q^{n} of Σ C(n,k) p^k q^{n−k}) are folded
    // into the closed form's base where the geometric identities allow it.
    let const_factors = mul_factors(arena, shape.constant);
    let (q, constant) = split_pow_n(arena, &const_factors, n);
    let has_q = q != one;
    let mut folded_q = false;
    let result = match (e, shape.lin_pows.as_slice(), x) {
        // Σ C(n,k) = 2^n
        (1, [], None) => arena.pow(two, n),
        // Σ k C(n,k) = n 2^(n−1)
        (1, [(b, p)], None) if b.is_zero() && p.is_one() => {
            let t = arena.pow(two, n_m1);
            arena.mul(&[n, t])
        }
        // Σ k² C(n,k) = n(n+1) 2^(n−2)
        (1, [(b, p)], None) if b.is_zero() && *p == rat_i(2) => {
            let n_m2 = arena.int(-2);
            let n_m2 = arena.add(&[n, n_m2]);
            let t = arena.pow(two, n_m2);
            arena.mul(&[n, n_p1, t])
        }
        // Σ C(n,k)/(k+1) = (2^(n+1) − 1)/(n+1)
        (1, [(b, p)], None) if b.is_one() && *p == -Rat::one() => {
            let t = arena.pow(two, n_p1);
            let num = arena.sub(t, one);
            arena.div(num, n_p1)
        }
        // Σ C(n,k)² = C(2n, n)
        (2, [], None) => {
            let two_n = arena.mul(&[two, n]);
            arena.binomial(two_n, n)
        }
        // Σ C(n,k) x^k = (1+x)^n      (x = −1 gives 0 for n ≥ 1, 1 for n = 0)
        (1, [], Some(x)) => {
            if as_rat(arena, x) == Some(-Rat::one()) {
                let zero = arena.zero;
                let cond = arena.eq_(n, zero);
                let ncond = arena.ne_(n, zero);
                arena.piecewise(&[(one, cond), (zero, ncond)])
            } else {
                // (1+x)^n·q^n = ((1+x)q)^n
                folded_q = true;
                let s = arena.add(&[one, x]);
                let base = scale_and_tidy(arena, s, q);
                arena.pow(base, n)
            }
        }
        // Σ k C(n,k) x^k = n x (1+x)^(n−1)   (·q^n = n·(xq)·((1+x)q)^(n−1))
        (1, [(b, p)], Some(x)) if b.is_zero() && p.is_one() => {
            folded_q = true;
            let s = arena.add(&[one, x]);
            let base = scale_and_tidy(arena, s, q);
            let xq = scale_and_tidy(arena, x, q);
            let t = arena.pow(base, n_m1);
            arena.mul(&[n, xq, t])
        }
        _ => return None,
    };
    let mut factors = vec![constant, result];
    if has_q && !folded_q {
        factors.push(arena.pow(q, n));
    }
    let total = arena.mul(&factors);
    Some(eval::eval(arena, total))
}

// ═══════════════════════════════════════════════════════════════════════════
// Geometric / arithmetico-geometric sums
// ═══════════════════════════════════════════════════════════════════════════

/// `(P as monomial list, Y, const_factor)` — see [`geometric_split`].
type GeometricSplit = (Vec<(usize, ExprId)>, ExprId, ExprId);

/// Split `body = P(k)·Y^k·const` where `Y` collects every exponential factor.
/// Returns `(P as monomial list, Y, const_factor)`; `P` may have symbolic
/// coefficients.
fn geometric_split(arena: &mut Arena, body: ExprId, var: ExprId) -> Option<GeometricSplit> {
    let factors = mul_factors(arena, body);
    let mut y_parts = Vec::new();
    let mut c_parts = Vec::new();
    let mut p_parts = Vec::new();
    for f in factors {
        if !depends_on(arena, f, var) {
            c_parts.push(f);
            continue;
        }
        match arena.node(f).clone() {
            ExprNode::Pow(base, exp)
                if depends_on(arena, exp, var) && !depends_on(arena, base, var) =>
            {
                let (a, b) = linear_in_sym(arena, exp, var)?;
                y_parts.push(pow_rat(arena, base, &a));
                if !arena.is_zero_structural(b) {
                    c_parts.push(pow_const(arena, base, b));
                }
            }
            ExprNode::Exp(arg) => {
                let (a, b) = linear_in_sym(arena, arg, var)?;
                let e = arena.e_const;
                y_parts.push(pow_rat(arena, e, &a));
                if !arena.is_zero_structural(b) {
                    c_parts.push(pow_const(arena, e, b));
                }
            }
            _ => p_parts.push(f),
        }
    }
    if y_parts.is_empty() {
        return None;
    }
    let y = mul_all(arena, &y_parts);
    let y = eval::eval(arena, y);
    let p_expr = mul_all(arena, &p_parts);
    let poly = sym_poly_in(arena, p_expr, var)?;
    if poly.is_empty() {
        return None;
    }
    let c = mul_all(arena, &c_parts);
    Some((poly, y, c))
}

/// Apply `Σ_j p_j (r·d/dr)^j` to `s0(r)` and substitute `r = y`.
fn apply_euler_operator(
    arena: &mut Arena,
    poly: &[(usize, ExprId)],
    s0: ExprId,
    r: ExprId,
    y: ExprId,
) -> ExprId {
    let max_p = poly.iter().map(|(p, _)| *p).max().unwrap_or(0);
    let mut derivs = Vec::with_capacity(max_p + 1);
    derivs.push(s0);
    for _ in 0..max_p {
        let prev = *derivs.last().unwrap_or(&s0);
        let d = crate::transforms::diff::diff(arena, prev, r);
        let rd = arena.mul(&[r, d]);
        let rd = eval::eval(arena, rd);
        derivs.push(rd);
    }
    let mut terms = Vec::new();
    for &(p, c) in poly {
        terms.push(arena.mul(&[c, derivs[p]]));
    }
    let total = add_all(arena, &terms);
    let at_y = subs::subs(arena, total, r, y);
    let at_y = eval::eval(arena, at_y);
    let together = arena.together_expr(at_y);
    eval::eval(arena, together)
}

fn geometric_poly_finite(
    arena: &mut Arena,
    body: ExprId,
    var: ExprId,
    lo: ExprId,
    hi: ExprId,
) -> Option<ExprId> {
    let (poly, y, c) = geometric_split(arena, body, var)?;
    if as_rat(arena, y) == Some(Rat::one()) {
        return None; // degenerate; polynomial path applies
    }
    // Numeric ratio with rational coefficients → Gosper gives the tidiest form.
    if as_rat(arena, y).is_some()
        && let Some(g) = gosper::gosper_sum(arena, body, var, lo, hi)
    {
        return Some(g);
    }
    let r = arena.symbol("_r");
    let one = arena.one;
    let r_lo = arena.pow(r, lo);
    let hi1 = arena.add(&[hi, one]);
    let r_hi1 = arena.pow(r, hi1);
    let num = arena.sub(r_lo, r_hi1);
    let den = arena.sub(one, r);
    let s0 = arena.div(num, den);
    let formula = apply_euler_operator(arena, &poly, s0, r, y);
    let formula = arena.mul(&[c, formula]);
    if as_rat(arena, y).is_some() || const_value(arena, y).is_some() {
        return Some(formula);
    }
    // Symbolic ratio: Piecewise on y = 1.  Both conditions are explicit so
    // that `eval` only selects a branch once `y` is known.
    let poly_sum = faulhaber_sym(arena, &poly, lo, hi);
    let poly_sum = arena.mul(&[c, poly_sum]);
    let cond = arena.eq_(y, one);
    let ncond = arena.ne_(y, one);
    Some(arena.piecewise(&[(poly_sum, cond), (formula, ncond)]))
}

fn geometric_poly_infinite(
    arena: &mut Arena,
    body: ExprId,
    var: ExprId,
    lo: ExprId,
) -> Option<SumOutcome> {
    let (poly, y, c) = geometric_split(arena, body, var)?;
    match abs_less_than_one(arena, y) {
        Some(true) => {}
        Some(false) => {
            // Divergent: direction known only for positive ratio & rational sign.
            let y_pos = const_sign(arena, y) == Some(true);
            let lead = poly.last().map(|(_, c)| *c);
            let sign = if y_pos {
                lead.and_then(|l| {
                    let lc = arena.mul(&[c, l]);
                    const_sign(arena, lc)
                })
            } else {
                None
            };
            return Some(SumOutcome::Divergent(infinity_of_sign(arena, sign)));
        }
        None => return Some(SumOutcome::Unevaluated),
    }
    let r = arena.symbol("_r");
    let one = arena.one;
    let r_lo = arena.pow(r, lo);
    let den = arena.sub(one, r);
    let s0 = arena.div(r_lo, den);
    let formula = apply_euler_operator(arena, &poly, s0, r, y);
    let total = arena.mul(&[c, formula]);
    Some(SumOutcome::Closed(eval::eval(arena, total)))
}

// ═══════════════════════════════════════════════════════════════════════════
// Infinite sums
// ═══════════════════════════════════════════════════════════════════════════

fn infinite_sum(arena: &mut Arena, body: ExprId, var: ExprId, lo: ExprId) -> SumOutcome {
    // 0. Constant body.
    if !depends_on(arena, body, var) {
        let v = eval::eval(arena, body);
        if arena.is_zero_structural(v) {
            return SumOutcome::Closed(v);
        }
        let sign = const_sign(arena, v);
        return SumOutcome::Divergent(infinity_of_sign(arena, sign));
    }
    let node = arena.node(body).clone();

    // 1. Rational functions (handles cancellation between divergent pieces).
    if let Some(r) = rational_sum_infinite(arena, body, var, lo) {
        return r;
    }

    // 2. Sums.
    if let ExprNode::Add(ref terms) = node {
        let terms: Vec<ExprId> = terms.to_vec();
        if let Some(r) = telescoping_infinite(arena, &terms, var, lo) {
            return r;
        }
        let mut closed = Vec::new();
        let mut divergent: Vec<Option<ExprId>> = Vec::new();
        let mut unknown = 0usize;
        for &t in &terms {
            match infinite_sum(arena, t, var, lo) {
                SumOutcome::Closed(v) => closed.push(v),
                SumOutcome::Divergent(d) => divergent.push(d),
                SumOutcome::Unevaluated => unknown += 1,
            }
        }
        if unknown == 0 && divergent.is_empty() {
            let s = add_all(arena, &closed);
            return SumOutcome::Closed(eval::eval(arena, s));
        }
        if unknown == 0 && divergent.len() == 1 {
            return SumOutcome::Divergent(divergent[0]);
        }
        if unknown == 0 && !divergent.is_empty() {
            // Several divergent pieces: decisive only if they all point the same way.
            let first = divergent[0];
            if first.is_some() && divergent.iter().all(|d| *d == first) {
                return SumOutcome::Divergent(first);
            }
        }
        return SumOutcome::Unevaluated;
    }

    // 3. Constant-factor extraction.
    if let ExprNode::Mul(ref factors) = node {
        let factors: Vec<ExprId> = factors.to_vec();
        let (consts, varf): (Vec<ExprId>, Vec<ExprId>) = factors
            .iter()
            .copied()
            .partition(|&f| !depends_on(arena, f, var));
        if !consts.is_empty() && !varf.is_empty() {
            let c = mul_all(arena, &consts);
            let c = eval::eval(arena, c);
            if arena.is_zero_structural(c) {
                return SumOutcome::Closed(c);
            }
            let inner = mul_all(arena, &varf);
            return match infinite_sum(arena, inner, var, lo) {
                SumOutcome::Closed(v) => {
                    let s = arena.mul(&[c, v]);
                    SumOutcome::Closed(eval::eval(arena, s))
                }
                SumOutcome::Divergent(Some(inf)) => match const_sign(arena, c) {
                    Some(true) => SumOutcome::Divergent(Some(inf)),
                    Some(false) => {
                        let flipped = if inf == arena.infinity {
                            arena.neg_infinity
                        } else {
                            arena.infinity
                        };
                        SumOutcome::Divergent(Some(flipped))
                    }
                    None => SumOutcome::Divergent(None),
                },
                other => other,
            };
        }
    }

    // 4. Shape-based recognisers: p-series, then geometric / arithmetico-
    //    geometric (symbolic polynomial multipliers), then the power-series table.
    let shape = term_shape(arena, body, var);
    if let Some(shape) = &shape
        && let Some(r) = p_series_infinite(arena, shape, lo)
    {
        return r;
    }
    if let Some(r) = geometric_poly_infinite(arena, body, var, lo) {
        return r;
    }
    if let Some(shape) = &shape
        && let Some(r) = power_series_infinite(arena, shape, var, lo)
    {
        return r;
    }

    // 5. Negative-binomial series Σ P(k)·C(k+c, k)·xᵏ.
    if let Some(r) = negative_binomial_series(arena, body, var, lo) {
        return r;
    }

    // 6. Gosper antidifference with an exactly computable tail limit.
    if let Some(r) = gosper_infinite(arena, body, var, lo) {
        return r;
    }

    // 7. Convergence tests can still prove divergence.
    if crate::calculus::convergence::is_convergent(arena, body, var) == Some(false) {
        let sign = eventual_sign(arena, body, var);
        return SumOutcome::Divergent(infinity_of_sign(arena, sign));
    }

    SumOutcome::Unevaluated
}

/// Eventual sign of `body(k)` for large `k`, when structurally obvious:
/// a hypergeometric monomial with a non-alternating sign and positive
/// geometric base has the sign of its constant.
fn eventual_sign(arena: &mut Arena, body: ExprId, var: ExprId) -> Option<bool> {
    let shape = term_shape(arena, body, var)?;
    if shape.alternating {
        return None;
    }
    for (b, a) in &shape.bases {
        if a.is_integer() && a.to_integer().to_i64()? % 2 == 0 {
            continue;
        }
        if const_sign(arena, *b) != Some(true) {
            return None;
        }
    }
    if shape.numeric_base.is_negative() {
        return None;
    }
    const_sign(arena, shape.constant)
}

/// Gosper for infinite sums: `Σ_{k=lo}^{∞} t(k) = lim_{N→∞} g(N+1) − g(lo)`
/// when the tail limit is exactly computable (rational tail, or a
/// `P(N)·r^N` tail with `|r| < 1`).
fn gosper_infinite(arena: &mut Arena, body: ExprId, var: ExprId, lo: ExprId) -> Option<SumOutcome> {
    let n = arena.symbol("_N");
    let g = gosper::gosper_sum(arena, body, var, lo, n)?;
    let limit = limit_at_infinity(arena, g, n)?;
    Some(SumOutcome::Closed(limit))
}

// ── p-series ────────────────────────────────────────────────────────────────

/// `Σ_{k=lo}^{∞} c·(±1)^k·(k+β)^(−p)`.
fn p_series_infinite(arena: &mut Arena, shape: &TermShape, lo: ExprId) -> Option<SumOutcome> {
    if !shape.is_pure_lin_pows() || shape.lin_pows.len() != 1 {
        return None;
    }
    let (beta, exp) = shape.lin_pows[0].clone();
    let c = shape.constant;
    if exp >= -Rat::one() {
        // Terms do not decay fast enough (or grow).
        if shape.alternating {
            if exp >= Rat::zero() {
                return Some(SumOutcome::Divergent(None));
            }
            // Alternating with (k+β)^(−p), 0 < p ≤ 1: conditionally convergent.
            // Closed forms only for p = 1 (handled below); otherwise leave.
        } else {
            let sign = const_sign(arena, c);
            return Some(SumOutcome::Divergent(infinity_of_sign(arena, sign)));
        }
    }
    if !exp.is_integer() {
        return None;
    }
    let p = (-exp.to_integer().to_i64()?) as usize;
    let lo_r = const_value(arena, lo)?;
    let q = &lo_r + &beta; // first argument k+β
    let value = if !shape.alternating {
        // Integer or half-integer offsets have Hurwitz-zeta closed forms.
        if beta.is_integer() || (&beta * rat_i(2)).is_integer() {
            hurwitz_zeta_closed(arena, p, &q)?
        } else {
            return None;
        }
    } else if beta.is_integer() {
        // Σ_{k=lo}^{∞} (−1)^k (k+β)^(−p) = (−1)^β [−η(p) − Σ_{j=1}^{q−1} (−1)^j j^(−p)]
        if !q.is_positive() {
            return None;
        }
        let qi = q.to_integer().to_i64()?;
        if qi > MAX_TELESCOPE_SHIFT {
            return None;
        }
        let eta = eta_value(arena, p)?;
        let mut acc = Rat::zero();
        for j in 1..qi {
            let s = if j % 2 == 0 { Rat::one() } else { -Rat::one() };
            acc += s * rat_pow_i(&rat_i(j), -(p as i64));
        }
        let neg_eta = arena.neg(eta);
        let ce = rat_expr(arena, -acc);
        let inner = arena.add(&[neg_eta, ce]);
        let sign_beta = if beta.to_integer().to_i64()?.rem_euclid(2) == 0 {
            arena.one
        } else {
            arena.neg_one
        };
        let v = arena.mul(&[sign_beta, inner]);
        eval::eval(arena, v)
    } else if (&beta * rat_i(2)).is_integer() {
        // β = n + 1/2:  (k + β)^(−p) = 2^p (2(k+n) + 1)^(−p).  With j = k + n,
        // Σ_{k=lo}^{∞} (−1)^k (k+β)^(−p) = 2^p (−1)^n [β(p) − Σ_{j=0}^{s−1} (−1)^j (2j+1)^(−p)],
        // where s = lo + n is the first index of the tail.
        let half = Rat::new(BigInt::one(), BigInt::from(2));
        let n = &beta - &half;
        let s = &q - &half;
        if !s.is_integer() || s.is_negative() {
            return None;
        }
        let si = s.to_integer().to_i64()?;
        if si > MAX_TELESCOPE_SHIFT {
            return None;
        }
        let ni = n.to_integer().to_i64()?;
        let beta_p = dirichlet_beta_value(arena, p)?;
        let mut acc = Rat::zero();
        for j in 0..si {
            let sg = if j % 2 == 0 { Rat::one() } else { -Rat::one() };
            acc += sg * rat_pow_i(&rat_i(2 * j + 1), -(p as i64));
        }
        let ce = rat_expr(arena, -acc);
        let inner = arena.add(&[beta_p, ce]);
        let scale = rat_pow_i(&rat_i(2), p as i64)
            * if ni.rem_euclid(2) == 0 {
                Rat::one()
            } else {
                -Rat::one()
            };
        let se = rat_expr(arena, scale);
        let v = arena.mul(&[se, inner]);
        eval::eval(arena, v)
    } else {
        return None;
    };
    let total = arena.mul(&[c, value]);
    Some(SumOutcome::Closed(eval::eval(arena, total)))
}

// ── negative-binomial series ────────────────────────────────────────────────────

/// `Σ_{k=lo}^{∞} P(k)·C(k+c, k)·xᵏ`, where the binomial coefficient may also
/// be spelled `C(k+c, c)` and `c` is any `k`-free expression (numeric or
/// symbolic — `r − 1` for the NegativeBinomial pmf `C(k+r−1, k) pʳ (1−p)ᵏ`).
///
/// The base identity is the generalised binomial series, valid for every
/// `c` when `|x| < 1`:
///
/// ```text
/// Σ_{k≥0} C(k+c, k) xᵏ = (1 − x)^{−(c+1)}          (SymPy: (1 - x)**(-c - 1))
/// ```
///
/// A polynomial multiplier `P(k)` (symbolic coefficients allowed) is applied
/// through the Euler operator `P(x·d/dx)` exactly as for arithmetico-
/// geometric sums, so `Σ k·C(k+c,k) xᵏ = (c+1)·x·(1−x)^{−(c+2)}`; a lower
/// bound `lo > 0` subtracts the skipped leading terms.
///
/// Convergence follows this module's geometric-series convention: `|x| < 1`
/// is decided only for a numeric ratio.  A *symbolic* ratio (no `|x| < 1`
/// assumption is available) leaves the sum unevaluated — SymPy returns a
/// `Piecewise` over `Abs(x) < 1` instead.  `|x| ≥ 1` with a numeric `c ≥ 0`
/// is reported as divergent.
fn negative_binomial_series(
    arena: &mut Arena,
    body: ExprId,
    var: ExprId,
    lo: ExprId,
) -> Option<SumOutcome> {
    let lo_i = as_i64(arena, lo)?;
    if !(0..=MAX_ENUMERATION_TERMS).contains(&lo_i) {
        return None;
    }
    let mut consts = Vec::new();
    let mut poly_factors = Vec::new();
    let mut x_parts = Vec::new();
    let mut shift: Option<ExprId> = None;
    for f in mul_factors(arena, body) {
        if !depends_on(arena, f, var) {
            consts.push(f);
            continue;
        }
        match arena.node(f).clone() {
            ExprNode::Binomial(n, m) => {
                if shift.is_some() {
                    return None;
                }
                shift = Some(binomial_shift(arena, n, m, var)?);
            }
            ExprNode::Pow(base, exp)
                if !depends_on(arena, base, var) && depends_on(arena, exp, var) =>
            {
                let (a, b) = linear_in_sym(arena, exp, var)?;
                x_parts.push(pow_rat(arena, base, &a));
                if !arena.is_zero_structural(b) {
                    consts.push(pow_const(arena, base, b));
                }
            }
            ExprNode::Exp(arg) => {
                let (a, b) = linear_in_sym(arena, arg, var)?;
                let e = arena.e_const;
                x_parts.push(pow_rat(arena, e, &a));
                if !arena.is_zero_structural(b) {
                    consts.push(pow_const(arena, e, b));
                }
            }
            _ => {
                sym_poly_in(arena, f, var)?;
                poly_factors.push(f);
            }
        }
    }
    let c = shift?;
    if x_parts.is_empty() {
        return None;
    }
    let x = mul_all(arena, &x_parts);
    let x = eval::eval(arena, x);
    match abs_less_than_one(arena, x) {
        Some(true) => {}
        Some(false) => {
            // |C(k+c, k)| ≥ 1 for a numeric c ≥ 0, so the terms cannot tend
            // to zero; for other `c` the question is left open.
            let c_val = const_value(arena, c)?;
            if c_val.is_negative() {
                return None;
            }
            let sign = if const_sign(arena, x) == Some(true) {
                // Leading coefficient of P (1 when there is no polynomial).
                let lead = if poly_factors.is_empty() {
                    arena.one
                } else {
                    let p_expr = mul_all(arena, &poly_factors);
                    let poly = sym_poly_in(arena, p_expr, var)?;
                    poly.last().map(|(_, c)| *c)?
                };
                let mut all = consts.clone();
                all.push(lead);
                let lc = mul_all(arena, &all);
                const_sign(arena, lc)
            } else {
                None
            };
            return Some(SumOutcome::Divergent(infinity_of_sign(arena, sign)));
        }
        None => return Some(SumOutcome::Unevaluated),
    }
    // (1 − r)^{−(c+1)} in a placeholder r, then P(r d/dr) and r → x.
    let r = arena.symbol("_r");
    let one = arena.one;
    let omr = arena.sub(one, r);
    let c1 = arena.add(&[c, one]);
    let neg_c1 = arena.neg(c1);
    let neg_c1 = eval::eval(arena, neg_c1);
    let s0 = arena.pow(omr, neg_c1);
    let f = if poly_factors.is_empty() {
        let at_x = subs::subs(arena, s0, r, x);
        eval::eval(arena, at_x)
    } else {
        let p_expr = mul_all(arena, &poly_factors);
        let poly = sym_poly_in(arena, p_expr, var)?;
        apply_euler_operator(arena, &poly, s0, r, x)
    };
    tracing::debug!("summation: negative-binomial series");
    consts.push(f);
    let mut total = mul_all(arena, &consts);
    if lo_i > 0 {
        let mut skipped = Vec::with_capacity(lo_i as usize);
        for k in 0..lo_i {
            let ke = arena.int(k);
            let t = subs::subs(arena, body, var, ke);
            skipped.push(eval::eval(arena, t));
        }
        let s = add_all(arena, &skipped);
        total = arena.sub(total, s);
    }
    Some(SumOutcome::Closed(eval::eval(arena, total)))
}

/// The `k`-free shift `c` of a binomial coefficient written as `C(k + c, k)`
/// or `C(k + c, c)`; `None` for any other shape.
fn binomial_shift(arena: &mut Arena, n: ExprId, m: ExprId, var: ExprId) -> Option<ExprId> {
    if m == var {
        let c = arena.sub(n, var);
        let c = eval::eval(arena, c);
        return (!depends_on(arena, c, var)).then_some(c);
    }
    if depends_on(arena, m, var) {
        return None;
    }
    // C(n, m) with n − m = k.
    let d = arena.sub(n, m);
    let d = arena.sub(d, var);
    is_zero_expr(arena, d).then_some(m)
}

// ── power-series table ────────────────────────────────────────────────────────────

/// Convergence domain of a power series in `x`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Domain {
    /// Converges for every `x`.
    Everywhere,
    /// `|x| < 1`.
    OpenUnit,
    /// `|x| ≤ 1`.
    ClosedUnit,
    /// `|x| < 1` or `x = −1`.
    OpenUnitOrMinusOne,
}

/// One entry of the classical power-series table: the canonical term
///
/// ```text
/// u(k) = table_const · (−1)^k[alt] · base_const^k · x^(A·k + B) · Π (k+β)^p · Π ((αk+β)!)^e
/// ```
///
/// with `Σ_{k=k0}^{∞} u(k) = F(x)`.
struct SeriesEntry {
    name: &'static str,
    alternating: bool,
    base_const: Rat,
    lin_pows: &'static [(i64, i64, i64)], // (β numer, β denom, p)
    facts: &'static [(i64, i64, i64)],    // (α, β, e)
    a: i64,
    b: i64,
    k0: i64,
    table_const: Rat,
    domain: Domain,
    build: fn(&mut Arena, ExprId) -> ExprId,
}

fn series_table() -> Vec<SeriesEntry> {
    let one = Rat::one;
    let half = || Rat::new(BigInt::one(), BigInt::from(2));
    vec![
        SeriesEntry {
            name: "exp",
            alternating: false,
            base_const: one(),
            lin_pows: &[],
            facts: &[(1, 0, -1)],
            a: 1,
            b: 0,
            k0: 0,
            table_const: one(),
            domain: Domain::Everywhere,
            build: |a, x| a.exp(x),
        },
        SeriesEntry {
            name: "sin",
            alternating: true,
            base_const: one(),
            lin_pows: &[],
            facts: &[(2, 1, -1)],
            a: 2,
            b: 1,
            k0: 0,
            table_const: one(),
            domain: Domain::Everywhere,
            build: |a, x| a.sin(x),
        },
        SeriesEntry {
            name: "cos",
            alternating: true,
            base_const: one(),
            lin_pows: &[],
            facts: &[(2, 0, -1)],
            a: 2,
            b: 0,
            k0: 0,
            table_const: one(),
            domain: Domain::Everywhere,
            build: |a, x| a.cos(x),
        },
        SeriesEntry {
            name: "sinh",
            alternating: false,
            base_const: one(),
            lin_pows: &[],
            facts: &[(2, 1, -1)],
            a: 2,
            b: 1,
            k0: 0,
            table_const: one(),
            domain: Domain::Everywhere,
            build: |a, x| a.sinh(x),
        },
        SeriesEntry {
            name: "cosh",
            alternating: false,
            base_const: one(),
            lin_pows: &[],
            facts: &[(2, 0, -1)],
            a: 2,
            b: 0,
            k0: 0,
            table_const: one(),
            domain: Domain::Everywhere,
            build: |a, x| a.cosh(x),
        },
        SeriesEntry {
            name: "-ln(1-x)",
            alternating: false,
            base_const: one(),
            lin_pows: &[(0, 1, -1)],
            facts: &[],
            a: 1,
            b: 0,
            k0: 1,
            table_const: one(),
            domain: Domain::OpenUnitOrMinusOne,
            build: |a, x| {
                let one = a.one;
                let omx = a.sub(one, x);
                let omx = eval::eval(a, omx);
                // −ln(p/q) = ln(q/p) for rational arguments.
                if let Some(r) = as_rat(a, omx)
                    && r.is_positive()
                {
                    let inv = rat_expr(a, Rat::one() / r);
                    return a.ln(inv);
                }
                let l = a.ln(omx);
                a.neg(l)
            },
        },
        SeriesEntry {
            name: "atan",
            alternating: true,
            base_const: one(),
            lin_pows: &[(1, 2, -1)],
            facts: &[],
            a: 2,
            b: 1,
            k0: 0,
            table_const: half(),
            domain: Domain::ClosedUnit,
            build: |a, x| a.atan(x),
        },
        SeriesEntry {
            name: "atanh",
            alternating: false,
            base_const: one(),
            lin_pows: &[(1, 2, -1)],
            facts: &[],
            a: 2,
            b: 1,
            k0: 0,
            table_const: half(),
            domain: Domain::OpenUnit,
            build: |a, x| a.atanh(x),
        },
        SeriesEntry {
            name: "1/(1-x)",
            alternating: false,
            base_const: one(),
            lin_pows: &[],
            facts: &[],
            a: 1,
            b: 0,
            k0: 0,
            table_const: one(),
            domain: Domain::OpenUnit,
            build: |a, x| {
                let one = a.one;
                let omx = a.sub(one, x);
                a.div(one, omx)
            },
        },
        // asin x = Σ (2k)!/(4^k (k!)² (2k+1)) x^(2k+1)
        SeriesEntry {
            name: "asin",
            alternating: false,
            base_const: Rat::new(BigInt::one(), BigInt::from(4)),
            lin_pows: &[(1, 2, -1)],
            facts: &[(1, 0, -2), (2, 0, 1)],
            a: 2,
            b: 1,
            k0: 0,
            table_const: half(),
            domain: Domain::ClosedUnit,
            build: |a, x| a.asin(x),
        },
        SeriesEntry {
            name: "asinh",
            alternating: true,
            base_const: Rat::new(BigInt::one(), BigInt::from(4)),
            lin_pows: &[(1, 2, -1)],
            facts: &[(1, 0, -2), (2, 0, 1)],
            a: 2,
            b: 1,
            k0: 0,
            table_const: half(),
            domain: Domain::ClosedUnit,
            build: |a, x| a.asinh(x),
        },
        // erf x = (2/√π) Σ (−1)^k x^(2k+1)/(k! (2k+1))
        SeriesEntry {
            name: "erf",
            alternating: true,
            base_const: one(),
            lin_pows: &[(1, 2, -1)],
            facts: &[(1, 0, -1)],
            a: 2,
            b: 1,
            k0: 0,
            table_const: half(),
            domain: Domain::Everywhere,
            build: |a, x| {
                let e = a.erf(x);
                let pi = a.pi;
                let sp = a.sqrt(pi);
                let two = a.int(2);
                let half_sqrt_pi = a.div(sp, two);
                a.mul(&[half_sqrt_pi, e])
            },
        },
        // J₀(x) = Σ (−1)^k x^(2k)/(4^k (k!)²),  I₀(x) = Σ x^(2k)/(4^k (k!)²)
        SeriesEntry {
            name: "besselj0",
            alternating: true,
            base_const: Rat::new(BigInt::one(), BigInt::from(4)),
            lin_pows: &[],
            facts: &[(1, 0, -2)],
            a: 2,
            b: 0,
            k0: 0,
            table_const: one(),
            domain: Domain::Everywhere,
            build: |a, x| {
                let z = a.zero;
                a.besselj(z, x)
            },
        },
        SeriesEntry {
            name: "besseli0",
            alternating: false,
            base_const: Rat::new(BigInt::one(), BigInt::from(4)),
            lin_pows: &[],
            facts: &[(1, 0, -2)],
            a: 2,
            b: 0,
            k0: 0,
            table_const: one(),
            domain: Domain::Everywhere,
            build: |a, x| {
                let z = a.zero;
                a.besseli(z, x)
            },
        },
    ]
}

fn entry_lin_pows(e: &SeriesEntry) -> Vec<(Rat, Rat)> {
    e.lin_pows
        .iter()
        .map(|&(bn, bd, p)| (Rat::new(BigInt::from(bn), BigInt::from(bd)), rat_i(p)))
        .collect()
}

fn entry_facts(e: &SeriesEntry) -> Vec<(Rat, Rat, i64)> {
    let mut v: Vec<(Rat, Rat, i64)> = e
        .facts
        .iter()
        .map(|&(a, b, ee)| (rat_i(a), rat_i(b), ee))
        .collect();
    v.sort_by_key(|a| (a.0.clone(), a.1.clone()));
    v
}

/// Does `x` lie in the entry's convergence domain?  `None` = undecidable.
fn in_domain(arena: &mut Arena, x: ExprId, domain: Domain) -> Option<bool> {
    if domain == Domain::Everywhere {
        return Some(true);
    }
    if let Some(r) = const_value(arena, x) {
        let a = r.abs();
        let one = Rat::one();
        return Some(match domain {
            Domain::Everywhere => true,
            Domain::OpenUnit => a < one,
            Domain::ClosedUnit => a <= one,
            Domain::OpenUnitOrMinusOne => a < one || r == -one,
        });
    }
    // Non-rational constant: decide strictly inside / outside the unit disc
    // (a transcendental constant cannot sit exactly on the unit circle in a
    // way we could certify, so the boundary cases stay undecided).
    abs_less_than_one(arena, x)
}

/// Recognise `Σ_{k=lo}^{∞} t(k)` against the classical power-series table.
fn power_series_infinite(
    arena: &mut Arena,
    shape: &TermShape,
    var: ExprId,
    lo: ExprId,
) -> Option<SumOutcome> {
    if !shape.binomials.is_empty() {
        return None;
    }
    let lo_i = as_i64(arena, lo)?;
    // Extra k^p factor → Euler operator ((x d/dx − B)/A)^p on F.
    let mut extra_power: i64 = 0;
    let mut lin = shape.lin_pows.clone();
    if let Some(pos) = lin
        .iter()
        .position(|(b, p)| b.is_zero() && p.is_integer() && p.is_positive())
    {
        let (_, p) = lin.remove(pos);
        extra_power = p.to_integer().to_i64()?;
        if extra_power > 6 {
            return None;
        }
    }
    let _ = var;
    for entry in series_table() {
        // Structural match of algebraic / factorial parts.
        if entry_lin_pows(&entry) != lin || entry_facts(&entry) != shape.facts {
            continue;
        }
        // Alternation: identical, or absorbed into x → −x when A is odd.
        let mut flip_x = false;
        if entry.alternating != shape.alternating {
            if entry.a % 2 == 1 {
                flip_x = true;
            } else {
                continue;
            }
        }
        // Geometric part: numeric_base · Π base^a = base_const · x^A  ⇒  x = (numeric_base/base_const · Π base^a)^(1/A)
        let ratio = &shape.numeric_base / &entry.base_const;
        let a_rat = rat_i(entry.a);
        let mut x_factors = Vec::new();
        if !ratio.is_one() {
            if ratio.is_negative() && entry.a % 2 == 0 {
                continue;
            }
            let re = rat_expr(arena, ratio.clone());
            let inv_a = Rat::one() / &a_rat;
            x_factors.push(pow_rat(arena, re, &inv_a));
        }
        let mut ok = true;
        for (base, a) in &shape.bases {
            let q = a / &a_rat;
            if !q.is_integer() {
                ok = false;
                break;
            }
            x_factors.push(pow_rat(arena, *base, &q));
        }
        if !ok {
            continue;
        }
        let x = mul_all(arena, &x_factors);
        let x = eval::eval(arena, x);
        let x = if flip_x {
            let nx = arena.neg(x);
            eval::eval(arena, nx)
        } else {
            x
        };
        // Constant adjustment: t(k) = C_body · u(k) / (table_const · x^B · (−1)^B[if flipped])
        let xb = pow_rat(arena, x, &rat_i(entry.b));
        let tc = rat_expr(arena, entry.table_const.clone());
        let mut denom_factors = vec![tc, xb];
        if flip_x && entry.b % 2 == 1 {
            denom_factors.push(arena.neg_one);
        }
        let denom = arena.mul(&denom_factors);
        let coeff = arena.div(shape.constant, denom);
        let coeff = eval::eval(arena, coeff);
        // Domain.
        match in_domain(arena, x, entry.domain) {
            Some(true) => {}
            Some(false) => {
                let sign = if !shape.alternating && const_sign(arena, x) == Some(true) {
                    const_sign(arena, coeff)
                } else {
                    None
                };
                return Some(SumOutcome::Divergent(infinity_of_sign(arena, sign)));
            }
            None => return Some(SumOutcome::Unevaluated),
        }
        if lo_i < entry.k0 {
            return None;
        }
        tracing::debug!("summation: power-series table hit `{}`", entry.name);
        let f = if extra_power > 0 {
            // Apply ((x d/dx − B)/A)^extra_power on a symbolic placeholder,
            // then substitute the actual argument.
            let xs = arena.symbol("_x");
            let mut g = (entry.build)(arena, xs);
            for _ in 0..extra_power {
                let d = crate::transforms::diff::diff(arena, g, xs);
                let xd = arena.mul(&[xs, d]);
                let be = arena.int(entry.b);
                let bg = arena.mul(&[be, g]);
                let num = arena.sub(xd, bg);
                let ae = arena.int(entry.a);
                g = arena.div(num, ae);
                g = eval::eval(arena, g);
            }
            let at_x = subs::subs(arena, g, xs, x);
            eval::eval(arena, at_x)
        } else {
            (entry.build)(arena, x)
        };
        // Subtract the skipped leading terms u(k0..lo−1) — build them from
        // the canonical term via the entry's structure: u(k) = t(k)/coeff.
        let mut skipped = Vec::new();
        for k in entry.k0..lo_i {
            let ke = arena.int(k);
            // u(k) = table_const · (−1)^k · base_const^k · x^(Ak+B) · Π(k+β)^p · Π((αk+β)!)^e · k^extra
            let mut fs = vec![tc];
            if entry.alternating && k % 2 != 0 {
                fs.push(arena.neg_one);
            }
            if !entry.base_const.is_one() {
                fs.push(rat_expr(arena, rat_pow_i(&entry.base_const, k)));
            }
            fs.push(pow_rat(arena, x, &rat_i(entry.a * k + entry.b)));
            for (beta, p) in entry_lin_pows(&entry) {
                let kb = rat_i(k) + beta;
                if kb.is_zero() && p.is_negative() {
                    return None;
                }
                fs.push(rat_expr(arena, rat_pow_i(&kb, p.to_integer().to_i64()?)));
            }
            for (al, be, e) in entry_facts(&entry) {
                let arg = (al * rat_i(k) + be).to_integer().to_i64()?;
                if arg < 0 {
                    return None;
                }
                let fv = Rat::from_integer(factorial_big(arg as u64));
                fs.push(rat_expr(arena, rat_pow_i(&fv, e)));
            }
            if extra_power > 0 {
                fs.push(pow_rat(arena, ke, &rat_i(extra_power)));
            }
            skipped.push(arena.mul(&fs));
        }
        let mut total_terms = vec![f];
        for s in skipped {
            total_terms.push(arena.neg(s));
        }
        let inner = add_all(arena, &total_terms);
        let total = arena.mul(&[coeff, inner]);
        return Some(SumOutcome::Closed(eval::eval(arena, total)));
    }
    None
}

// ═══════════════════════════════════════════════════════════════════════════
// Products
// ═══════════════════════════════════════════════════════════════════════════

/// Evaluate `Π_{var=lower}^{upper} body` symbolically.
pub(crate) fn product(
    arena: &mut Arena,
    body: ExprId,
    var: ExprId,
    lower: ExprId,
    upper: ExprId,
) -> SumOutcome {
    if !matches!(arena.node(var), ExprNode::Symbol(_)) {
        return SumOutcome::Unevaluated;
    }
    match (classify_bound(arena, lower), classify_bound(arena, upper)) {
        (Extended::Finite(lo), Extended::Finite(hi)) => finite_product(arena, body, var, lo, hi),
        (Extended::Finite(lo), Extended::PosInf) => infinite_product(arena, body, var, lo),
        _ => SumOutcome::Unevaluated,
    }
}

fn finite_product(
    arena: &mut Arena,
    body: ExprId,
    var: ExprId,
    lo: ExprId,
    hi: ExprId,
) -> SumOutcome {
    if let (Some(a), Some(b)) = (as_i64(arena, lo), as_i64(arena, hi)) {
        if b < a {
            return SumOutcome::Closed(arena.one);
        }
        if b - a < MAX_ENUMERATION_TERMS {
            return SumOutcome::Closed(enumerate_product(arena, body, var, a, b));
        }
    }
    match product_closed(arena, body, var, lo, hi) {
        Some(id) => {
            let n = gamma_ratio_normalize(arena, id);
            let n = gamma_to_factorial(arena, n);
            let n = flatten_nested_pows(arena, n);
            SumOutcome::Closed(eval::eval(arena, n))
        }
        None => SumOutcome::Unevaluated,
    }
}

fn arena_neg_one(arena: &Arena) -> ExprId {
    arena.neg_one
}

/// Rewrite `(b^p)^q` factors with rational `p, q` as `b^(pq)` so that
/// same-base powers (e.g. `√π · (√π)⁻¹`) cancel in the canonical `Mul`.
fn flatten_nested_pows(arena: &mut Arena, expr: ExprId) -> ExprId {
    let factors = mul_factors(arena, expr);
    let mut out: Vec<ExprId> = Vec::with_capacity(factors.len());
    // Numeric-base powers with symbolic exponents, grouped by base: (base, [exps]).
    let mut numeric_pows: Vec<(ExprId, Vec<ExprId>)> = Vec::new();
    let mut changed = false;
    for f in factors {
        if let ExprNode::Pow(base, exp) = arena.node(f).clone() {
            // (b^p)^q → b^(pq) when q is an integer (always valid) or both
            // exponents are rational.
            if let ExprNode::Pow(b2, e2) = arena.node(base).clone()
                && let Some(q) = as_rat(arena, exp)
                && (q.is_integer() || as_rat(arena, e2).is_some())
            {
                let qe = rat_expr(arena, q);
                let pq = arena.mul(&[e2, qe]);
                let pq = eval::eval(arena, pq);
                let flat = arena.pow(b2, pq);
                // Re-classify the flattened factor (it may now be a numeric-base power).
                if let ExprNode::Pow(nb, ne) = arena.node(flat).clone()
                    && arena.as_num(nb).is_some()
                    && arena.as_num(ne).is_none()
                {
                    if let Some(slot) = numeric_pows.iter_mut().find(|(b, _)| *b == nb) {
                        slot.1.push(ne);
                    } else {
                        numeric_pows.push((nb, vec![ne]));
                    }
                } else {
                    out.push(flat);
                }
                changed = true;
                continue;
            }
            if arena.as_num(base).is_some() && arena.as_num(exp).is_none() {
                if let Some(slot) = numeric_pows.iter_mut().find(|(b, _)| *b == base) {
                    slot.1.push(exp);
                    changed = true;
                } else {
                    numeric_pows.push((base, vec![exp]));
                }
                continue;
            }
        }
        out.push(f);
    }
    for (base, exps) in numeric_pows {
        let e = add_all(arena, &exps);
        let e = eval::eval(arena, e);
        out.push(arena.pow(base, e));
    }
    if changed {
        let r = mul_all(arena, &out);
        eval::eval(arena, r)
    } else {
        expr
    }
}

fn product_closed(
    arena: &mut Arena,
    body: ExprId,
    var: ExprId,
    lo: ExprId,
    hi: ExprId,
) -> Option<ExprId> {
    if !depends_on(arena, body, var) {
        let n = range_count(arena, lo, hi);
        return Some(arena.pow(body, n));
    }
    match arena.node(body).clone() {
        ExprNode::Mul(ref factors) => {
            let factors: Vec<ExprId> = factors.to_vec();
            let mut parts = Vec::new();
            for f in factors {
                parts.push(product_closed(arena, f, var, lo, hi)?);
            }
            Some(arena.mul(&parts))
        }
        ExprNode::Neg(inner) => {
            let m1 = arena.neg_one;
            let n = range_count(arena, lo, hi);
            let s = arena.pow(m1, n);
            let p = product_closed(arena, inner, var, lo, hi)?;
            Some(arena.mul(&[s, p]))
        }
        ExprNode::Pow(base, exp) if !depends_on(arena, exp, var) => {
            let p = product_closed(arena, base, var, lo, hi)?;
            Some(arena.pow(p, exp))
        }
        ExprNode::Pow(base, exp) if !depends_on(arena, base, var) => {
            let s = match summation(arena, exp, var, lo, hi) {
                SumOutcome::Closed(s) if !walk::has_unevaluated(arena, s) => s,
                _ => return None,
            };
            Some(arena.pow(base, s))
        }
        ExprNode::Exp(arg) => {
            let s = match summation(arena, arg, var, lo, hi) {
                SumOutcome::Closed(s) if !walk::has_unevaluated(arena, s) => s,
                _ => return None,
            };
            Some(arena.exp(s))
        }
        _ => rational_product(arena, body, var, lo, hi)
            .or_else(|| linear_symbolic_product(arena, body, var, lo, hi)),
    }
}

/// `Π_{k=lo}^{hi} (αk + β)` with rational `α ≠ 0` and a `k`-free (possibly
/// symbolic) `β`: `α^count · Γ(hi + 1 + β/α) / Γ(lo + β/α)`.
fn linear_symbolic_product(
    arena: &mut Arena,
    body: ExprId,
    var: ExprId,
    lo: ExprId,
    hi: ExprId,
) -> Option<ExprId> {
    let terms = sym_poly_in(arena, body, var)?;
    let alpha_e = terms.iter().find(|(p, _)| *p == 1).map(|(_, c)| *c)?;
    if terms.iter().any(|(p, _)| *p > 1) {
        return None;
    }
    let alpha = as_rat(arena, alpha_e)?;
    if alpha.is_zero() {
        return None;
    }
    let beta = terms
        .iter()
        .find(|(p, _)| *p == 0)
        .map(|(_, c)| *c)
        .unwrap_or(arena.zero);
    let count = range_count(arena, lo, hi);
    let mut factors = Vec::new();
    if !alpha.is_one() {
        factors.push(arena.pow(alpha_e, count));
    }
    let inv_alpha = rat_expr(arena, Rat::one() / &alpha);
    let shift = arena.mul(&[beta, inv_alpha]);
    let one = arena.one;
    let hi1 = arena.add(&[hi, one]);
    let top_arg = arena.add(&[hi1, shift]);
    let bot_arg = arena.add(&[lo, shift]);
    let top = arena.gamma(top_arg);
    let bot = arena.gamma(bot_arg);
    factors.push(top);
    factors.push(pow_rat(arena, bot, &(-Rat::one())));
    Some(arena.mul(&factors))
}

/// `Π_{k=lo}^{hi} N(k)/D(k)` for rational functions whose numerator and
/// denominator factor into linear factors over ℚ.
fn rational_product(
    arena: &mut Arena,
    body: ExprId,
    var: ExprId,
    lo: ExprId,
    hi: ExprId,
) -> Option<ExprId> {
    let body = combine_fractions(arena, body);
    let (n, d) = polybridge::as_numer_denom(arena, body);
    let np = polybridge::expr_to_poly(arena, n, var)?;
    let dp = polybridge::expr_to_poly(arena, d, var)?;
    if np.is_zero() || dp.is_zero() {
        return None;
    }
    let count = range_count(arena, lo, hi);
    let mut factors = Vec::new();
    for (poly, sign) in [(np, 1i64), (dp, -1i64)] {
        let (content, irreducibles) = poly.factor_over_z();
        if !content.is_one() {
            let ce = rat_expr(arena, content);
            let p = pow_rat(arena, ce, &rat_i(sign));
            factors.push(arena.pow(p, count));
        }
        for (fac, mult) in irreducibles {
            if fac.degree() != Some(1) {
                return None;
            }
            let alpha = fac.coeff(1);
            let beta = fac.coeff(0);
            let e = rat_i(sign * mult as i64);
            // Π (αk+β) = α^count · Γ(hi + 1 + β/α) / Γ(lo + β/α)
            if !alpha.is_one() {
                let ae = rat_expr(arena, alpha.clone());
                let ap = pow_rat(arena, ae, &e);
                factors.push(arena.pow(ap, count));
            }
            let shift = &beta / &alpha;
            let one = arena.one;
            let hi1 = arena.add(&[hi, one]);
            let top_arg = add_rat(arena, hi1, &shift);
            let bot_arg = add_rat(arena, lo, &shift);
            let top = arena.gamma(top_arg);
            let bot = arena.gamma(bot_arg);
            factors.push(pow_rat(arena, top, &e));
            factors.push(pow_rat(arena, bot, &(-e)));
        }
    }
    Some(arena.mul(&factors))
}

/// Cancel integer-shifted Gamma functions in a product:
/// `Γ(X + d) = Γ(X)·X(X+1)…(X+d−1)`, grouping arguments that differ by an
/// integer.  Anything that is not a Gamma factor is left untouched.
pub(crate) fn gamma_ratio_normalize(arena: &mut Arena, expr: ExprId) -> ExprId {
    let expr = eval::eval(arena, expr);
    let factors = mul_factors(arena, expr);
    // (arg, exponent) for Gamma factors; others pass through.
    let mut gammas: Vec<(ExprId, Rat)> = Vec::new();
    let mut others: Vec<ExprId> = Vec::new();
    for f in factors {
        match arena.node(f).clone() {
            ExprNode::Gamma(arg) => gammas.push((arg, Rat::one())),
            ExprNode::Pow(base, exp) => {
                if let (ExprNode::Gamma(arg), Some(e)) =
                    (arena.node(base).clone(), as_rat(arena, exp))
                {
                    gammas.push((arg, e));
                } else {
                    others.push(f);
                }
            }
            _ => others.push(f),
        }
    }
    if gammas.len() < 2 {
        return expr;
    }
    // Group by integer difference of arguments.
    let mut groups: Vec<Vec<(ExprId, Rat, i64)>> = Vec::new(); // (arg, exp, offset from group ref)
    'outer: for (arg, e) in gammas {
        for g in groups.iter_mut() {
            let (ref_arg, _, _) = g[0];
            let diff = arena.sub(arg, ref_arg);
            if let Some(d) = eval_rat(arena, diff)
                && d.is_integer()
                && let Some(di) = d.to_integer().to_i64()
                && di.abs() <= MAX_GAMMA_SHIFT
            {
                g.push((arg, e, di));
                continue 'outer;
            }
        }
        groups.push(vec![(arg, e, 0)]);
    }
    let mut out = others;
    for g in groups {
        if g.len() == 1 {
            let (arg, e, _) = g[0].clone();
            let ga = arena.gamma(arg);
            out.push(pow_rat(arena, ga, &e));
            continue;
        }
        let min_off = g.iter().map(|(_, _, d)| *d).min().unwrap_or(0);
        let (ref_arg, _, _) = g[0];
        // X = ref_arg + min_off
        let x = add_rat(arena, ref_arg, &rat_i(min_off));
        let x = eval::eval(arena, x);
        let mut total_e = Rat::zero();
        for (_, e, d) in &g {
            total_e += e;
            let steps = d - min_off;
            for j in 0..steps {
                let lin = add_rat(arena, x, &rat_i(j));
                out.push(pow_rat(arena, lin, e));
            }
        }
        if !total_e.is_zero() {
            let gx = arena.gamma(x);
            out.push(pow_rat(arena, gx, &total_e));
        }
    }
    let r = mul_all(arena, &out);
    eval::eval(arena, r)
}

/// Rewrite `Γ(n + c)` with a positive integer constant part `c` as `(n + c − 1)!`.
fn gamma_to_factorial(arena: &mut Arena, expr: ExprId) -> ExprId {
    let factors = mul_factors(arena, expr);
    let mut out = Vec::with_capacity(factors.len());
    let mut changed = false;
    for f in factors {
        let (base, exp) = arena.as_base_exp(f);
        if let ExprNode::Gamma(arg) = arena.node(base).clone()
            && let Some(fa) = gamma_arg_to_factorial(arena, arg)
        {
            changed = true;
            let p = if exp == arena.one {
                fa
            } else {
                arena.pow(fa, exp)
            };
            out.push(p);
        } else {
            out.push(f);
        }
    }
    if changed {
        let r = mul_all(arena, &out);
        eval::eval(arena, r)
    } else {
        expr
    }
}

fn gamma_arg_to_factorial(arena: &mut Arena, arg: ExprId) -> Option<ExprId> {
    if let Some(r) = as_rat(arena, arg) {
        if r.is_integer() && r.is_positive() {
            let m1 = rat_expr(arena, r - Rat::one());
            return Some(arena.factorial(m1));
        }
        return None;
    }
    let terms = add_terms(arena, arg);
    let mut const_part = Rat::zero();
    let mut rest = Vec::new();
    for t in terms {
        if let Some(r) = as_rat(arena, t) {
            const_part += r;
        } else {
            rest.push(t);
        }
    }
    if rest.is_empty() {
        return None;
    }
    let inner = add_all(arena, &rest);
    if const_part.is_integer() && const_part.is_positive() {
        let shifted = add_rat(arena, inner, &(const_part - Rat::one()));
        return Some(arena.factorial(shifted));
    }
    // Legendre duplication: Γ(m + 1/2) = (2m)! √π / (4^m m!)  for m = inner + (c − 1/2).
    let half = Rat::new(BigInt::one(), BigInt::from(2));
    let m_shift = &const_part - &half;
    if m_shift.is_integer() && !m_shift.is_negative() {
        let m = add_rat(arena, inner, &m_shift);
        let two = arena.int(2);
        let two_m = arena.mul(&[two, m]);
        let num = arena.factorial(two_m);
        let pi = arena.pi;
        let sp = arena.sqrt(pi);
        // Write 4^m as 2^(2m) so it merges with other powers of two.
        let two_m_e = arena.mul(&[two, m]);
        let four_m = arena.pow(two, two_m_e);
        let mf = arena.factorial(m);
        let inv_four_m = arena.pow(four_m, arena_neg_one(arena));
        let inv_mf = arena.pow(mf, arena_neg_one(arena));
        return Some(arena.mul(&[num, inv_four_m, inv_mf, sp]));
    }
    None
}

fn infinite_product(arena: &mut Arena, body: ExprId, var: ExprId, lo: ExprId) -> SumOutcome {
    if !depends_on(arena, body, var) {
        let v = eval::eval(arena, body);
        if let Some(r) = as_rat(arena, v) {
            if r.is_one() {
                return SumOutcome::Closed(v);
            }
            if r.is_zero() || r.abs() < Rat::one() {
                return SumOutcome::Closed(arena.zero);
            }
            if r > Rat::one() {
                return SumOutcome::Divergent(Some(arena.infinity));
            }
            return SumOutcome::Divergent(None);
        }
        return SumOutcome::Unevaluated;
    }
    let n = arena.symbol("_N");
    let Some(pn) = product_closed(arena, body, var, lo, n) else {
        return SumOutcome::Unevaluated;
    };
    let pn = gamma_ratio_normalize(arena, pn);
    // Exact limits only: rational functions of N.
    match rational_limit_at_infinity(arena, pn, n) {
        Some(Some(r)) => SumOutcome::Closed(rat_expr(arena, r)),
        Some(None) => {
            let sign = eventual_sign(arena, pn, n);
            SumOutcome::Divergent(infinity_of_sign(arena, sign))
        }
        None => {
            // P(N)·r^N with |r|<1 → 0
            if let Some(l) = limit_at_infinity(arena, pn, n) {
                return SumOutcome::Closed(l);
            }
            SumOutcome::Unevaluated
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn closed(o: SumOutcome) -> ExprId {
        match o {
            SumOutcome::Closed(id) => id,
            other => panic!("expected closed form, got {other:?}"),
        }
    }

    fn eval_at(arena: &mut Arena, e: ExprId, n: ExprId, v: i64) -> Rat {
        let ve = arena.int(v);
        let s = subs::subs(arena, e, n, ve);
        let s = eval::eval(arena, s);
        as_rat(arena, s).unwrap_or_else(|| panic!("not rational: {}", arena.display(s)))
    }

    fn brute(arena: &mut Arena, body: ExprId, k: ExprId, lo: i64, hi: i64) -> Rat {
        let mut acc = Rat::zero();
        for i in lo..=hi {
            let ie = arena.int(i);
            let t = subs::subs(arena, body, k, ie);
            let t = eval::eval(arena, t);
            acc += as_rat(arena, t)
                .unwrap_or_else(|| panic!("term not rational: {}", arena.display(t)));
        }
        acc
    }

    #[test]
    fn faulhaber_matches_enumeration_p_0_to_8() {
        let mut arena = Arena::new();
        let k = arena.symbol("k");
        let n = arena.symbol("n");
        let one = arena.one;
        for p in 0..=8usize {
            let body = if p == 0 {
                arena.one
            } else if p == 1 {
                k
            } else {
                let pe = arena.int(p as i64);
                arena.pow(k, pe)
            };
            let s = closed(summation(&mut arena, body, k, one, n));
            for nv in [0i64, 1, 2, 5, 10, 17] {
                let expected = brute(&mut arena, body, k, 1, nv);
                let got = eval_at(&mut arena, s, n, nv);
                assert_eq!(got, expected, "p={p}, n={nv}");
            }
        }
    }

    #[test]
    fn faulhaber_general_lower_bound() {
        let mut arena = Arena::new();
        let k = arena.symbol("k");
        let n = arena.symbol("n");
        let three = arena.int(3);
        let e = arena.int(3);
        let body = arena.pow(k, e);
        let s = closed(summation(&mut arena, body, k, three, n));
        for nv in [3i64, 4, 9, 20] {
            let expected = brute(&mut arena, body, k, 3, nv);
            assert_eq!(eval_at(&mut arena, s, n, nv), expected);
        }
    }

    #[test]
    fn faulhaber_b1_sign_convention() {
        // Σ_{k=1}^{n} k = n²/2 + n/2  ⇒ coefficient of n is +1/2.
        let p = faulhaber_coefficients(1);
        assert_eq!(p.coeff(1), Rat::new(BigInt::from(1), BigInt::from(2)));
        assert_eq!(p.coeff(2), Rat::new(BigInt::from(1), BigInt::from(2)));
    }

    #[test]
    fn telescoping_rational() {
        let mut arena = Arena::new();
        let k = arena.symbol("k");
        let n = arena.symbol("n");
        let one = arena.one;
        let k1 = arena.add(&[k, one]);
        let den = arena.mul(&[k, k1]);
        let body = arena.div(one, den);
        let s = closed(summation(&mut arena, body, k, one, n));
        for nv in [1i64, 2, 7, 30] {
            let expected = brute(&mut arena, body, k, 1, nv);
            assert_eq!(eval_at(&mut arena, s, n, nv), expected, "n={nv}");
        }
        // Infinite: 1
        let inf = arena.infinity;
        let s = closed(summation(&mut arena, body, k, one, inf));
        assert_eq!(as_rat(&arena, s), Some(Rat::one()));
    }

    #[test]
    fn telescoping_shift_two() {
        let mut arena = Arena::new();
        let k = arena.symbol("k");
        let n = arena.symbol("n");
        let one = arena.one;
        let two = arena.int(2);
        let k2 = arena.add(&[k, two]);
        let den = arena.mul(&[k, k2]);
        let body = arena.div(one, den);
        let s = closed(summation(&mut arena, body, k, one, n));
        for nv in [1i64, 2, 5, 12] {
            let expected = brute(&mut arena, body, k, 1, nv);
            assert_eq!(eval_at(&mut arena, s, n, nv), expected, "n={nv}");
        }
        let inf = arena.infinity;
        let s = closed(summation(&mut arena, body, k, one, inf));
        assert_eq!(
            as_rat(&arena, s),
            Some(Rat::new(BigInt::from(3), BigInt::from(4)))
        );
    }

    #[test]
    fn odd_reciprocals_product() {
        // Σ 1/((2k−1)(2k+1)) from 1 to n = n/(2n+1)
        let mut arena = Arena::new();
        let k = arena.symbol("k");
        let n = arena.symbol("n");
        let one = arena.one;
        let two = arena.int(2);
        let m1 = arena.neg_one;
        let twok = arena.mul(&[two, k]);
        let a = arena.add(&[twok, m1]);
        let b = arena.add(&[twok, one]);
        let den = arena.mul(&[a, b]);
        let body = arena.div(one, den);
        let s = closed(summation(&mut arena, body, k, one, n));
        for nv in [1i64, 3, 8] {
            let expected = brute(&mut arena, body, k, 1, nv);
            assert_eq!(eval_at(&mut arena, s, n, nv), expected, "n={nv}");
        }
        let inf = arena.infinity;
        let s = closed(summation(&mut arena, body, k, one, inf));
        assert_eq!(
            as_rat(&arena, s),
            Some(Rat::new(BigInt::from(1), BigInt::from(2)))
        );
    }

    #[test]
    fn harmonic_finite() {
        let mut arena = Arena::new();
        let k = arena.symbol("k");
        let n = arena.symbol("n");
        let one = arena.one;
        let body = arena.div(one, k);
        let s = closed(summation(&mut arena, body, k, one, n));
        assert_eq!(arena.display(s).to_string(), "harmonic(n)");
        assert_eq!(
            eval_at(&mut arena, s, n, 4),
            Rat::new(BigInt::from(25), BigInt::from(12))
        );
    }

    #[test]
    fn harmonic_infinite_diverges() {
        let mut arena = Arena::new();
        let k = arena.symbol("k");
        let one = arena.one;
        let inf = arena.infinity;
        let body = arena.div(one, k);
        assert_eq!(
            summation(&mut arena, body, k, one, inf),
            SumOutcome::Divergent(Some(arena.infinity))
        );
    }

    #[test]
    fn basel_and_friends() {
        let mut arena = Arena::new();
        let k = arena.symbol("k");
        let one = arena.one;
        let inf = arena.infinity;
        let pi = arena.pi;
        // ζ(2) = π²/6
        let m2 = arena.int(-2);
        let body = arena.pow(k, m2);
        let s = closed(summation(&mut arena, body, k, one, inf));
        let two = arena.int(2);
        let pi2 = arena.pow(pi, two);
        let six = arena.int(6);
        let expected = arena.div(pi2, six);
        assert_eq!(s, expected, "got {}", arena.display(s));
        // ζ(4) = π⁴/90
        let m4 = arena.int(-4);
        let body = arena.pow(k, m4);
        let s = closed(summation(&mut arena, body, k, one, inf));
        let four = arena.int(4);
        let pi4 = arena.pow(pi, four);
        let ninety = arena.int(90);
        let expected = arena.div(pi4, ninety);
        assert_eq!(s, expected, "got {}", arena.display(s));
        // ζ(3) → Zeta(3) node (Apéry's constant has no elementary form)
        let m3 = arena.int(-3);
        let body = arena.pow(k, m3);
        let s = closed(summation(&mut arena, body, k, one, inf));
        assert_eq!(arena.display(s).to_string(), "zeta(3)");
        // Σ (−1)^k/(2k+1)² = Catalan
        let two = arena.int(2);
        let two_k = arena.mul(&[two, k]);
        let odd = arena.add(&[two_k, one]);
        let m2 = arena.int(-2);
        let inv_sq = arena.pow(odd, m2);
        let neg_one = arena.neg_one;
        let alt = arena.pow(neg_one, k);
        let body = arena.mul(&[alt, inv_sq]);
        let zero = arena.zero;
        let s = closed(summation(&mut arena, body, k, zero, inf));
        assert_eq!(s, arena.catalan);
    }

    #[test]
    fn zeta_even_rationals() {
        assert_eq!(
            zeta_even_rational(1),
            Rat::new(BigInt::from(1), BigInt::from(6))
        );
        assert_eq!(
            zeta_even_rational(2),
            Rat::new(BigInt::from(1), BigInt::from(90))
        );
        assert_eq!(
            zeta_even_rational(3),
            Rat::new(BigInt::from(1), BigInt::from(945))
        );
        assert_eq!(
            zeta_even_rational(4),
            Rat::new(BigInt::from(1), BigInt::from(9450))
        );
    }

    #[test]
    fn euler_numbers() {
        assert_eq!(euler_number(0), BigInt::from(1));
        assert_eq!(euler_number(1), BigInt::from(-1));
        assert_eq!(euler_number(2), BigInt::from(5));
        assert_eq!(euler_number(3), BigInt::from(-61));
        assert_eq!(euler_number(4), BigInt::from(1385));
    }

    #[test]
    fn alternating_constants() {
        let mut arena = Arena::new();
        let k = arena.symbol("k");
        let one = arena.one;
        let zero = arena.zero;
        let inf = arena.infinity;
        let m1 = arena.neg_one;
        // Σ (−1)^(k+1)/k = ln 2
        let k1 = arena.add(&[k, one]);
        let sgn = arena.pow(m1, k1);
        let body = arena.div(sgn, k);
        let s = closed(summation(&mut arena, body, k, one, inf));
        assert_eq!(arena.display(s).to_string(), "ln(2)");
        // Σ (−1)^k/(2k+1) = π/4
        let two = arena.int(2);
        let twok1 = arena.mul(&[two, k]);
        let twok1 = arena.add(&[twok1, one]);
        let sgn = arena.pow(m1, k);
        let body = arena.div(sgn, twok1);
        let s = closed(summation(&mut arena, body, k, zero, inf));
        let pi = arena.pi;
        let four = arena.int(4);
        let expected = arena.div(pi, four);
        assert_eq!(s, expected, "got {}", arena.display(s));
        // Σ (−1)^(k+1)/k² = π²/12
        let k1 = arena.add(&[k, one]);
        let sgn = arena.pow(m1, k1);
        let m2 = arena.int(-2);
        let k2 = arena.pow(k, m2);
        let body = arena.mul(&[sgn, k2]);
        let s = closed(summation(&mut arena, body, k, one, inf));
        let pi2 = arena.pow(pi, two);
        let twelve = arena.int(12);
        let expected = arena.div(pi2, twelve);
        assert_eq!(s, expected, "got {}", arena.display(s));
        // Σ_{k≥0} 1/(2k+1)² = π²/8
        let m2 = arena.int(-2);
        let body = arena.pow(twok1, m2);
        let s = closed(summation(&mut arena, body, k, zero, inf));
        let eight = arena.int(8);
        let expected = arena.div(pi2, eight);
        assert_eq!(s, expected, "got {}", arena.display(s));
    }

    #[test]
    fn geometric_finite_and_infinite() {
        let mut arena = Arena::new();
        let k = arena.symbol("k");
        let n = arena.symbol("n");
        let zero = arena.zero;
        let inf = arena.infinity;
        let half = arena.rational(1, 2);
        let body = arena.pow(half, k);
        let s = closed(summation(&mut arena, body, k, zero, n));
        for nv in [0i64, 1, 5, 9] {
            let expected = brute(&mut arena, body, k, 0, nv);
            assert_eq!(eval_at(&mut arena, s, n, nv), expected);
        }
        let s = closed(summation(&mut arena, body, k, zero, inf));
        assert_eq!(as_rat(&arena, s), Some(rat_i(2)));
        // Σ k/2^k = 2
        let body2 = arena.mul(&[k, body]);
        let s = closed(summation(&mut arena, body2, k, zero, inf));
        assert_eq!(
            as_rat(&arena, s),
            Some(rat_i(2)),
            "got {}",
            arena.display(s)
        );
        // Σ k²/2^k = 6
        let two = arena.int(2);
        let k2 = arena.pow(k, two);
        let body3 = arena.mul(&[k2, body]);
        let s = closed(summation(&mut arena, body3, k, zero, inf));
        assert_eq!(
            as_rat(&arena, s),
            Some(rat_i(6)),
            "got {}",
            arena.display(s)
        );
        // Σ 2^k diverges
        let body4 = arena.pow(two, k);
        assert_eq!(
            summation(&mut arena, body4, k, zero, inf),
            SumOutcome::Divergent(Some(arena.infinity))
        );
    }

    #[test]
    fn arithmetico_geometric_symbolic_ratio() {
        let mut arena = Arena::new();
        let k = arena.symbol("k");
        let n = arena.symbol("n");
        let r = arena.symbol("r");
        let zero = arena.zero;
        let rk = arena.pow(r, k);
        let body = arena.mul(&[k, rk]);
        let s = closed(summation(&mut arena, body, k, zero, n));
        assert!(
            matches!(arena.node(s), ExprNode::Piecewise(_)),
            "{}",
            arena.display(s)
        );
        // Evaluate at r = 3, n = 4: Σ k 3^k = 3 + 18 + 81 + 324 = 426
        let three = arena.int(3);
        let four = arena.int(4);
        let v = subs::subs(&mut arena, s, r, three);
        let v = subs::subs(&mut arena, v, n, four);
        let v = eval::eval(&mut arena, v);
        assert_eq!(
            as_rat(&arena, v),
            Some(rat_i(426)),
            "got {}",
            arena.display(v)
        );
        // r = 1 branch: Σ k = 10
        let one = arena.one;
        let v = subs::subs(&mut arena, s, r, one);
        let v = subs::subs(&mut arena, v, n, four);
        let v = eval::eval(&mut arena, v);
        assert_eq!(
            as_rat(&arena, v),
            Some(rat_i(10)),
            "got {}",
            arena.display(v)
        );
    }

    #[test]
    fn binomial_identities() {
        let mut arena = Arena::new();
        let k = arena.symbol("k");
        let n = arena.symbol("n");
        let zero = arena.zero;
        let bin = arena.binomial(n, k);
        let s = closed(summation(&mut arena, bin, k, zero, n));
        let two = arena.int(2);
        assert_eq!(s, arena.pow(two, n), "got {}", arena.display(s));
        let body = arena.mul(&[k, bin]);
        let s = closed(summation(&mut arena, body, k, zero, n));
        for nv in [1i64, 2, 5, 8] {
            let nve = arena.int(nv);
            let body_n = subs::subs(&mut arena, body, n, nve);
            let expected = brute(&mut arena, body_n, k, 0, nv);
            assert_eq!(eval_at(&mut arena, s, n, nv), expected);
        }
        let body = arena.pow(bin, two);
        let s = closed(summation(&mut arena, body, k, zero, n));
        for nv in [0i64, 1, 3, 6] {
            let nve = arena.int(nv);
            let body_n = subs::subs(&mut arena, body, n, nve);
            let expected = brute(&mut arena, body_n, k, 0, nv);
            assert_eq!(eval_at(&mut arena, s, n, nv), expected);
        }
        let x = arena.symbol("x");
        let xk = arena.pow(x, k);
        let body = arena.mul(&[bin, xk]);
        let s = closed(summation(&mut arena, body, k, zero, n));
        let one = arena.one;
        let opx = arena.add(&[one, x]);
        assert_eq!(s, arena.pow(opx, n), "got {}", arena.display(s));
        let m1 = arena.neg_one;
        let sgn = arena.pow(m1, k);
        let body = arena.mul(&[bin, sgn]);
        let s = closed(summation(&mut arena, body, k, zero, n));
        assert_eq!(eval_at(&mut arena, s, n, 0), Rat::one());
        assert_eq!(eval_at(&mut arena, s, n, 5), Rat::zero());
    }

    #[test]
    fn power_series_table() {
        let mut arena = Arena::new();
        let k = arena.symbol("k");
        let x = arena.symbol("x");
        let zero = arena.zero;
        let one = arena.one;
        let inf = arena.infinity;
        let kf = arena.factorial(k);
        let xk = arena.pow(x, k);
        let body = arena.div(xk, kf);
        let s = closed(summation(&mut arena, body, k, zero, inf));
        assert_eq!(s, arena.exp(x), "got {}", arena.display(s));
        // sin
        let two = arena.int(2);
        let m1 = arena.neg_one;
        let twok1 = arena.mul(&[two, k]);
        let twok1 = arena.add(&[twok1, one]);
        let sgn = arena.pow(m1, k);
        let xp = arena.pow(x, twok1);
        let f = arena.factorial(twok1);
        let num = arena.mul(&[sgn, xp]);
        let body = arena.div(num, f);
        let s = closed(summation(&mut arena, body, k, zero, inf));
        assert_eq!(s, arena.sin(x), "got {}", arena.display(s));
        // cosh
        let twok = arena.mul(&[two, k]);
        let xp = arena.pow(x, twok);
        let f = arena.factorial(twok);
        let body = arena.div(xp, f);
        let s = closed(summation(&mut arena, body, k, zero, inf));
        assert_eq!(s, arena.cosh(x), "got {}", arena.display(s));
        // Σ (1/2)^k / k = ln 2
        let half = arena.rational(1, 2);
        let hk = arena.pow(half, k);
        let body = arena.div(hk, k);
        let s = closed(summation(&mut arena, body, k, one, inf));
        assert_eq!(arena.display(s).to_string(), "ln(2)");
        // Σ x^k/k with symbolic x → unevaluated (convergence unknown)
        let body = arena.div(xk, k);
        assert_eq!(
            summation(&mut arena, body, k, one, inf),
            SumOutcome::Unevaluated
        );
        // Σ 1/k! = e
        let body = arena.div(one, kf);
        let s = closed(summation(&mut arena, body, k, zero, inf));
        assert_eq!(s, arena.e_const, "got {}", arena.display(s));
        // Σ k x^k/k! = x e^x
        let body = arena.mul(&[k, xk]);
        let body = arena.div(body, kf);
        let s = closed(summation(&mut arena, body, k, zero, inf));
        let ex = arena.exp(x);
        let expected = arena.mul(&[x, ex]);
        assert_eq!(s, expected, "got {}", arena.display(s));
        // Σ_{k≥1} x^k/k! = e^x − 1
        let body = arena.div(xk, kf);
        let s = closed(summation(&mut arena, body, k, one, inf));
        let expected = arena.sub(ex, one);
        assert_eq!(s, expected, "got {}", arena.display(s));
    }

    #[test]
    fn negative_binomial_series_and_binomial_theorem() {
        let mut arena = Arena::new();
        let k = arena.symbol("k");
        let c = arena.symbol("c");
        let zero = arena.zero;
        let inf = arena.infinity;
        let third = arena.rational(1, 3);
        let xk = arena.pow(third, k);
        // Σ C(k+3,k)(1/3)^k = 81/16, Σ k·C(k+3,k)(1/3)^k = 81/8 (SymPy).
        let three = arena.int(3);
        let k3 = arena.add(&[k, three]);
        let bin = arena.binomial(k3, k);
        let body = arena.mul(&[bin, xk]);
        let s = closed(summation(&mut arena, body, k, zero, inf));
        assert_eq!(as_rat(&arena, s), Some(Rat::new(81.into(), 16.into())));
        let body_k = arena.mul(&[k, bin, xk]);
        let s = closed(summation(&mut arena, body_k, k, zero, inf));
        assert_eq!(as_rat(&arena, s), Some(Rat::new(81.into(), 8.into())));
        // Symbolic c with a numeric ratio closes to (2/3)^(-c-1); at c = 2 → 27/8.
        let kc = arena.add(&[k, c]);
        let binc = arena.binomial(kc, k);
        let body = arena.mul(&[binc, xk]);
        let s = closed(summation(&mut arena, body, k, zero, inf));
        assert!(!walk::has_unevaluated(&arena, s), "{}", arena.display(s));
        assert_eq!(eval_at(&mut arena, s, c, 2), Rat::new(27.into(), 8.into()));
        // Symbolic ratio: no |x| < 1 assumption → unevaluated.
        let x = arena.symbol("x");
        let xs = arena.pow(x, k);
        let body = arena.mul(&[bin, xs]);
        assert_eq!(
            summation(&mut arena, body, k, zero, inf),
            SumOutcome::Unevaluated
        );
        // Binomial theorem with a symbolic exponent offset: Σ C(n,k) p^k (1−p)^(n−k) = 1.
        let n = arena.symbol("n");
        let p = arena.symbol("p");
        let one = arena.one;
        let q = arena.sub(one, p);
        let nmk = arena.sub(n, k);
        let qnk = arena.pow(q, nmk);
        let pk = arena.pow(p, k);
        let bn = arena.binomial(n, k);
        let body = arena.mul(&[bn, pk, qnk]);
        let s = closed(summation(&mut arena, body, k, zero, n));
        assert_eq!(s, one, "got {}", arena.display(s));
        let body = arena.mul(&[k, bn, pk, qnk]);
        let s = closed(summation(&mut arena, body, k, zero, n));
        let np = arena.mul(&[n, p]);
        assert_eq!(s, np, "got {}", arena.display(s));
    }

    #[test]
    fn gosper_fallback_k_factorial() {
        let mut arena = Arena::new();
        let k = arena.symbol("k");
        let n = arena.symbol("n");
        let zero = arena.zero;
        let kf = arena.factorial(k);
        let body = arena.mul(&[k, kf]);
        let s = closed(summation(&mut arena, body, k, zero, n));
        for nv in [0i64, 1, 3, 5] {
            let expected = brute(&mut arena, body, k, 0, nv);
            assert_eq!(eval_at(&mut arena, s, n, nv), expected);
        }
    }

    #[test]
    fn linearity_partial_keeps_unevaluated_sum() {
        let mut arena = Arena::new();
        let k = arena.symbol("k");
        let n = arena.symbol("n");
        let one = arena.one;
        let sk = arena.sin(k);
        let body = arena.add(&[k, sk]);
        let s = closed(summation(&mut arena, body, k, one, n));
        assert!(walk::has_unevaluated(&arena, s));
        assert!(arena.display(s).to_string().contains("Sum"));
    }

    #[test]
    fn products_basic() {
        let mut arena = Arena::new();
        let k = arena.symbol("k");
        let n = arena.symbol("n");
        let one = arena.one;
        let two = arena.int(2);
        // Π k = n!
        let p = closed(product(&mut arena, k, k, one, n));
        assert_eq!(p, arena.factorial(n), "got {}", arena.display(p));
        // Π 2k = 2^n n!
        let body = arena.mul(&[two, k]);
        let p = closed(product(&mut arena, body, k, one, n));
        for nv in [1i64, 2, 4, 6] {
            let expected: Rat = (1..=nv).map(|i| rat_i(2 * i)).product();
            assert_eq!(eval_at(&mut arena, p, n, nv), expected);
        }
        // Π (1 + 1/k) = n + 1
        let inv = arena.div(one, k);
        let body = arena.add(&[one, inv]);
        let p = closed(product(&mut arena, body, k, one, n));
        let expected = arena.add(&[n, one]);
        assert_eq!(p, expected, "got {}", arena.display(p));
        // Π_{k=2}^{n} (1 − 1/k²) = (n+1)/(2n)
        let m2 = arena.int(-2);
        let k2 = arena.pow(k, m2);
        let body = arena.sub(one, k2);
        let p = closed(product(&mut arena, body, k, two, n));
        for nv in [2i64, 3, 5, 9] {
            let expected = Rat::new(BigInt::from(nv + 1), BigInt::from(2 * nv));
            assert_eq!(
                eval_at(&mut arena, p, n, nv),
                expected,
                "got {}",
                arena.display(p)
            );
        }
        let inf = arena.infinity;
        let p = closed(product(&mut arena, body, k, two, inf));
        assert_eq!(
            as_rat(&arena, p),
            Some(Rat::new(BigInt::from(1), BigInt::from(2)))
        );
        // Π (2k−1) = (2n)!/(2^n n!)
        let m1 = arena.neg_one;
        let twok = arena.mul(&[two, k]);
        let body = arena.add(&[twok, m1]);
        let p = closed(product(&mut arena, body, k, one, n));
        for nv in [1i64, 2, 3, 5] {
            let expected: Rat = (1..=nv).map(|i| rat_i(2 * i - 1)).product();
            assert_eq!(
                eval_at(&mut arena, p, n, nv),
                expected,
                "got {}",
                arena.display(p)
            );
        }
        // Π a^k = a^(n(n+1)/2)
        let a = arena.symbol("a");
        let body = arena.pow(a, k);
        let p = closed(product(&mut arena, body, k, one, n));
        let three = arena.int(3);
        let v = subs::subs(&mut arena, p, n, three);
        let v = eval::eval(&mut arena, v);
        let six = arena.int(6);
        assert_eq!(v, arena.pow(a, six), "got {}", arena.display(v));
    }
}

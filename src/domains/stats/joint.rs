//! Several random variables at once, and quantities derived from one:
//! joint expectations, variance / covariance / correlation of expressions
//! in several variables, closed-family sums, probabilities of rectangles
//! and orderings, conditional expectation and probability, entropy.
//!
//! # Independence
//!
//! Every function here that takes several variables treats them as
//! **independent**: the joint density is the product of the marginals.
//! That is the only joint model in this module, and it is the model SymPy's
//! `stats` uses for variables declared separately (`X = Normal('X', 0, 1)`,
//! `Y = Normal('Y', 0, 1)`).  Listing the same variable twice is rejected;
//! a variable that is listed but does not occur in the expression is
//! harmless.
//!
//! Symbols in an expression that are *not* among the listed variables are
//! parameters (constants): `E[aX] = a·E[X]`.
//!
//! ```
//! use symplex::prelude::*;
//! use symplex::stats::{self, Distribution, RandomVariable};
//!
//! let ctx = Context::new();
//! let x = RandomVariable::new(&ctx, "X", Distribution::normal(ctx.int(0), ctx.int(1)));
//! let y = RandomVariable::new(&ctx, "Y", Distribution::normal(ctx.int(0), ctx.int(1)));
//! let (xs, ys) = (x.symbol(), y.symbol());
//!
//! assert_eq!(stats::expectation(&[&x, &y], &(xs * ys))?, ctx.int(0));
//! assert_eq!(stats::variance(&[&x, &y], &(xs + ys))?, ctx.int(2));
//! assert_eq!(stats::probability(&[&x, &y], &xs.lt(ys))?, ctx.rational(1, 2));
//! # Ok::<(), SymplexError>(())
//! ```

use crate::api::context::Context;
use crate::api::expr::{BoolEx, Ex, Expr, Sort};
use crate::api::poly_ex::Poly;
use crate::base::errors::SymplexError;

use super::continuous::{ChiSquared, Exponential, Gamma, Normal};
use super::discrete::{Bernoulli, Binomial, NegativeBinomial, Poisson};
use super::events::{Rel, flatten_relations};
use super::family::Distribution;
use super::rv::RandomVariable;
use super::support::{Kind, Support, is_neg_inf};

fn invalid(operation: &'static str, reason: impl Into<String>) -> SymplexError {
    SymplexError::invalid_argument(operation, reason)
}

/// The variable list of a joint query: non-empty, one context, pairwise
/// distinct symbols (the same variable twice is not independent of itself).
fn check_vars(operation: &'static str, vars: &[&RandomVariable]) -> Result<Context, SymplexError> {
    let Some(first) = vars.first() else {
        return Err(invalid(
            operation,
            "at least one random variable is required",
        ));
    };
    let ctx = first.context();
    for (i, v) in vars.iter().enumerate() {
        if v.context().id != ctx.id {
            return Err(invalid(
                operation,
                "the random variables must live in the same context",
            ));
        }
        if vars[..i].iter().any(|u| u.symbol() == v.symbol()) {
            return Err(invalid(
                operation,
                format!(
                    "`{}` is listed twice; the variables must be distinct (and independent)",
                    v.symbol()
                ),
            ));
        }
    }
    Ok(ctx)
}

/// An expression handed to a joint query must live in the variables'
/// context (an `Err` here instead of the cross-context panic).
fn check_context<S: Sort>(
    operation: &'static str,
    ctx: &Context,
    e: &Expr<S>,
) -> Result<(), SymplexError> {
    if e.ctx_id == ctx.id {
        Ok(())
    } else {
        Err(invalid(
            operation,
            "the expression and the random variables live in different contexts",
        ))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Joint expectations
// ═══════════════════════════════════════════════════════════════════════════

/// `E[g(X₁, …, Xₖ)]` for **independent** random variables.  SymPy:
/// `E(X*Y)`, `E(X + Y)`, `E(expr)` with several variables in `expr`.
///
/// Three routes, tried in order:
///
/// 1. `g` is a polynomial in the variables' symbols (other symbols are
///    coefficients): each monomial `c·ΠXᵢ^{nᵢ}` contributes `c·ΠE[Xᵢ^{nᵢ}]`
///    through the marginals' raw [`moment`](RandomVariable::moment)s —
///    exact and cheap.
/// 2. Otherwise the variables are integrated out one at a time
///    (`E[g] = E_{X_k}[⋯E_{X_1}[g]]`), each step being that marginal's own
///    [`expectation`](RandomVariable::expectation); variables in which `g`
///    is polynomial go first so that the closed-form moments do as much of
///    the work as possible.  For a non-polynomial `g` in exactly one
///    variable this is simply that variable's `expectation`.
///
/// The result may contain an unevaluated `Integral` or `Sum` when the
/// integrator cannot close a step (check with
/// [`has_unevaluated`](Ex::has_unevaluated)); such a result still
/// evaluates numerically with [`eval_f64`](Ex::eval_f64).
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if `vars` is empty, lists a variable
/// twice, or mixes contexts.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::{self, Distribution, RandomVariable};
///
/// let ctx = Context::new();
/// let x = RandomVariable::new(&ctx, "X", Distribution::normal(ctx.int(0), ctx.int(1)));
/// let y = RandomVariable::new(&ctx, "Y", Distribution::normal(ctx.int(0), ctx.int(1)));
/// let (xs, ys) = (x.symbol(), y.symbol());
///
/// // Independent standard normals: E[XY] = 0, E[(X + Y)²] = 2, E[X²Y²] = 1.
/// assert_eq!(stats::expectation(&[&x, &y], &(xs * ys))?, ctx.int(0));
/// assert_eq!(stats::expectation(&[&x, &y], &(xs + ys).powi(2))?, ctx.int(2));
/// assert_eq!(stats::expectation(&[&x, &y], &(xs.powi(2) * ys.powi(2)))?, ctx.int(1));
/// # Ok::<(), SymplexError>(())
/// ```
pub fn expectation(vars: &[&RandomVariable], g: &Ex) -> Result<Ex, SymplexError> {
    const OP: &str = "stats::expectation";
    let ctx = check_vars(OP, vars)?;
    check_context(OP, &ctx, g)?;
    let present: Vec<&RandomVariable> = vars
        .iter()
        .copied()
        .filter(|v| g.contains(v.symbol()))
        .collect();
    if present.is_empty() {
        return Ok(g.clone());
    }
    let expanded = g.expand();
    // Route 1: a polynomial in all the present symbols — product of raw
    // moments per monomial (independence).
    let symbols: Vec<&Ex> = present.iter().map(|v| v.symbol()).collect();
    if let Ok(poly) = Poly::try_new(&expanded, &symbols) {
        let mut acc = ctx.zero();
        for (exps, coeff) in poly.terms() {
            let mut term = coeff;
            for (v, &n) in present.iter().zip(&exps) {
                if n > 0 {
                    term *= v.moment(n);
                }
            }
            acc += term;
        }
        return Ok(acc.simplify());
    }
    // Route 2: integrate / sum out one variable at a time.  Variables in
    // which `g` is polynomial first (closed by moments), the rest after.
    let (poly_vars, other_vars): (Vec<&RandomVariable>, Vec<&RandomVariable>) = present
        .iter()
        .partition(|v| Poly::try_new(&expanded, &[v.symbol()]).is_ok());
    let mut acc = g.clone();
    for v in poly_vars.iter().chain(&other_vars) {
        acc = v.expectation(&acc);
    }
    Ok(acc.simplify())
}

/// `Var[g] = E[g²] − E[g]²` for an expression `g` in independent variables.
/// SymPy: `variance(X + Y)`, `variance(2*X + 1)`.
///
/// # Errors
///
/// As [`expectation`].
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::{self, Distribution, RandomVariable};
///
/// let ctx = Context::new();
/// let x = RandomVariable::new(&ctx, "X", Distribution::normal(ctx.int(0), ctx.int(1)));
/// let y = RandomVariable::new(&ctx, "Y", Distribution::normal(ctx.int(0), ctx.int(1)));
/// let (xs, ys) = (x.symbol(), y.symbol());
///
/// assert_eq!(stats::variance(&[&x, &y], &(xs + ys))?, ctx.int(2));
/// assert_eq!(stats::variance(&[&x], &(2 * xs + 1))?, ctx.int(4));
/// # Ok::<(), SymplexError>(())
/// ```
pub fn variance(vars: &[&RandomVariable], g: &Ex) -> Result<Ex, SymplexError> {
    let mean = expectation(vars, g)?;
    let second = expectation(vars, &g.powi(2))?;
    Ok((second - mean.powi(2)).simplify())
}

/// `Cov[g, h] = E[gh] − E[g]E[h]` for expressions in independent
/// variables.  SymPy: `covariance(X, 2*X)` (= 2 for a standard normal `X`).
///
/// # Errors
///
/// As [`expectation`].
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::{self, Distribution, RandomVariable};
///
/// let ctx = Context::new();
/// let x = RandomVariable::new(&ctx, "X", Distribution::normal(ctx.int(0), ctx.int(1)));
/// let y = RandomVariable::new(&ctx, "Y", Distribution::normal(ctx.int(0), ctx.int(1)));
/// let (xs, ys) = (x.symbol(), y.symbol());
///
/// assert_eq!(stats::covariance(&[&x], xs, &(2 * xs))?, ctx.int(2));
/// // Independent variables are uncorrelated.
/// assert_eq!(stats::covariance(&[&x, &y], xs, ys)?, ctx.int(0));
/// # Ok::<(), SymplexError>(())
/// ```
pub fn covariance(vars: &[&RandomVariable], g: &Ex, h: &Ex) -> Result<Ex, SymplexError> {
    let joint = expectation(vars, &(g * h))?;
    let eg = expectation(vars, g)?;
    let eh = expectation(vars, h)?;
    Ok((joint - eg * eh).simplify())
}

/// Pearson correlation `Cov[g, h] / √(Var[g]·Var[h])`.  SymPy:
/// `correlation(X, 2*X + 1)` (= 1).
///
/// # Errors
///
/// As [`expectation`]; additionally [`SymplexError::InvalidArgument`] when
/// `g` or `h` is degenerate (its variance is known to be zero).
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::{self, Distribution, RandomVariable};
///
/// let ctx = Context::new();
/// let x = RandomVariable::new(&ctx, "X", Distribution::normal(ctx.int(0), ctx.int(1)));
/// let xs = x.symbol();
///
/// assert_eq!(stats::correlation(&[&x], xs, &(2 * xs + 1))?, ctx.int(1));
/// assert!(stats::correlation(&[&x], xs, &ctx.int(3)).is_err());
/// # Ok::<(), SymplexError>(())
/// ```
pub fn correlation(vars: &[&RandomVariable], g: &Ex, h: &Ex) -> Result<Ex, SymplexError> {
    const OP: &str = "stats::correlation";
    let cov = covariance(vars, g, h)?;
    let vg = variance(vars, g)?;
    let vh = variance(vars, h)?;
    if vg.is_zero() == Some(true) || vh.is_zero() == Some(true) {
        return Err(invalid(
            OP,
            "the correlation of a degenerate (zero-variance) quantity is undefined",
        ));
    }
    Ok((cov / (vg * vh).sqrt()).simplify())
}

// ═══════════════════════════════════════════════════════════════════════════
// Sums of independent variables
// ═══════════════════════════════════════════════════════════════════════════

/// The distribution of `A + B` for independent `a`, `b` when it belongs to
/// a closed family; `None` otherwise.  The closure results implemented
/// (parameters that must agree — `p`, `λ` — are compared structurally):
///
/// * `Normal(μ₁, σ₁) + Normal(μ₂, σ₂) = Normal(μ₁ + μ₂, √(σ₁² + σ₂²))`
/// * `Binomial(n, p) + Binomial(m, p) = Binomial(n + m, p)`, with
///   `Bernoulli(p)` counting as `Binomial(1, p)`
/// * `Poisson(λ₁) + Poisson(λ₂) = Poisson(λ₁ + λ₂)`
/// * `NegativeBinomial(r₁, p) + NegativeBinomial(r₂, p) = NegativeBinomial(r₁ + r₂, p)`
///
/// SymPy has no direct call for this; `density(X + Y)` computes the
/// convolution.  Two handles to the *same* variable are not independent
/// (`X + X = 2X`) and give `None`.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::{self, Distribution, RandomVariable};
///
/// let ctx = Context::new();
/// let x = RandomVariable::new(&ctx, "X", Distribution::normal(ctx.int(0), ctx.int(1)));
/// let y = RandomVariable::new(&ctx, "Y", Distribution::normal(ctx.int(1), ctx.int(2)));
/// assert_eq!(
///     stats::sum_distribution(&x, &y),
///     Some(Distribution::normal(ctx.int(1), ctx.int(5).sqrt()))
/// );
///
/// let b = RandomVariable::new(&ctx, "B", Distribution::binomial(ctx.int(3), ctx.rational(1, 2)));
/// let c = RandomVariable::new(&ctx, "C", Distribution::binomial(ctx.int(2), ctx.rational(1, 2)));
/// assert_eq!(
///     stats::sum_distribution(&b, &c),
///     Some(Distribution::binomial(ctx.int(5), ctx.rational(1, 2)))
/// );
/// assert_eq!(stats::sum_distribution(&x, &b), None);
///
/// let p = RandomVariable::new(&ctx, "P", Distribution::poisson(ctx.int(2)));
/// let q = RandomVariable::new(&ctx, "Q", Distribution::poisson(ctx.int(3)));
/// assert_eq!(stats::sum_distribution(&p, &q), Some(Distribution::poisson(ctx.int(5))));
/// ```
pub fn sum_distribution(a: &RandomVariable, b: &RandomVariable) -> Option<Distribution> {
    if a.symbol() == b.symbol() || a.context().id != b.context().id {
        return None;
    }
    let ctx = a.context();
    let (da, db) = (a.distribution(), b.distribution());
    // `Bernoulli(p)` is `Binomial(1, p)`: n + m trials with the same
    // success probability.
    if let (Some((n, p)), Some((m, q))) = (as_binomial(da), as_binomial(db))
        && p == q
    {
        return Some(Distribution::binomial((n + m).simplify(), p));
    }
    // Sum of independent normals: means add, variances add.
    if let (Some(x), Some(y)) = (da.downcast_ref::<Normal>(), db.downcast_ref::<Normal>()) {
        return Some(Distribution::normal(
            (&x.mean + &y.mean).simplify(),
            (x.std.powi(2) + y.std.powi(2)).sqrt().simplify(),
        ));
    }
    // Superposition of Poisson processes: rates add.
    if let (Some(x), Some(y)) = (da.downcast_ref::<Poisson>(), db.downcast_ref::<Poisson>()) {
        return Some(Distribution::poisson((&x.rate + &y.rate).simplify()));
    }
    // Failures before the r₁-th success, then before r₂ more.
    if let (Some(x), Some(y)) = (
        da.downcast_ref::<NegativeBinomial>(),
        db.downcast_ref::<NegativeBinomial>(),
    ) && x.p == y.p
    {
        return Some(Distribution::negative_binomial(
            (&x.r + &y.r).simplify(),
            x.p.clone(),
        ));
    }
    // Gamma with a common scale: shapes add (Exponential(λ) is
    // Gamma(1, 1/λ); ChiSquared(k) is Gamma(k/2, 2)).
    let (k1, t1) = as_gamma(da)?;
    let (k2, t2) = as_gamma(db)?;
    if t1 != t2 {
        return None;
    }
    let shape = (k1 + k2).simplify();
    // Two chi-squareds stay a chi-squared.
    if da.downcast_ref::<ChiSquared>().is_some() && db.downcast_ref::<ChiSquared>().is_some() {
        return Some(Distribution::chi_squared((shape * ctx.int(2)).simplify()));
    }
    Some(Distribution::gamma(shape, t1))
}

/// A distribution as `Gamma(shape, scale)`, when it is one.
fn as_gamma(d: &Distribution) -> Option<(Ex, Ex)> {
    if let Some(g) = d.downcast_ref::<Gamma>() {
        return Some((g.shape.clone(), g.scale.clone()));
    }
    if let Some(e) = d.downcast_ref::<Exponential>() {
        let ctx = e.rate.context();
        return Some((ctx.one(), (ctx.one() / &e.rate).simplify()));
    }
    if let Some(c) = d.downcast_ref::<ChiSquared>() {
        let ctx = c.dof.context();
        return Some(((&c.dof / ctx.int(2)).simplify(), ctx.int(2)));
    }
    None
}

/// `(n, p)` for a `Binomial(n, p)` or a `Bernoulli(p)` (`n = 1`).
fn as_binomial(d: &Distribution) -> Option<(Ex, Ex)> {
    if let Some(b) = d.downcast_ref::<Binomial>() {
        return Some((b.n.clone(), b.p.clone()));
    }
    if let Some(b) = d.downcast_ref::<Bernoulli>() {
        return Some((b.p.context().one(), b.p.clone()));
    }
    None
}

// ═══════════════════════════════════════════════════════════════════════════
// Joint probabilities
// ═══════════════════════════════════════════════════════════════════════════

/// `P(event)` for an event in several **independent** variables.  SymPy:
/// `P(And(X > 0, Y > 0))`, `P(X < Y)`.
///
/// Two shapes are supported:
///
/// * a conjunction of relations each mentioning exactly one of the
///   variables (a rectangle: `X > 0 ∧ Y ≤ 2 ∧ 1 < X`) — the product of the
///   marginal [`probabilities`](RandomVariable::probability), so every
///   shape that method accepts per variable (`X < a`, `a ≤ X`, `X = a`,
///   …) is accepted here;
/// * a single ordering `X < Y`, `X ≤ Y`, `X > Y`, `X ≥ Y` (or `X = Y`) of
///   two continuous variables: for two normals exactly, through
///   `X − Y ~ Normal(μ_X − μ_Y, √(σ_X² + σ_Y²))`; otherwise as
///   `∫ f_X(t)·(1 − F_Y(t)) dt` over the support, which needs closed-form
///   distribution functions for both and may stay an unevaluated
///   `Integral` (numerically evaluable with [`eval_f64`](Ex::eval_f64)).
///   `P(X = Y) = 0` for continuous variables.
///
/// # Errors
///
/// [`SymplexError::NotImplemented`] for any other shape (disjunctions,
/// relations mixing variables such as `X + Y > 0`, an ordering combined
/// with further relations, an ordering of discrete variables);
/// [`SymplexError::InvalidArgument`] as [`expectation`].
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::{self, Distribution, RandomVariable};
///
/// let ctx = Context::new();
/// let x = RandomVariable::new(&ctx, "X", Distribution::normal(ctx.int(0), ctx.int(1)));
/// let y = RandomVariable::new(&ctx, "Y", Distribution::normal(ctx.int(0), ctx.int(1)));
/// let (xs, ys) = (x.symbol(), y.symbol());
/// let zero = ctx.int(0);
///
/// let quadrant = xs.gt(&zero).and(&ys.gt(&zero));
/// assert_eq!(stats::probability(&[&x, &y], &quadrant)?, ctx.rational(1, 4));
/// assert_eq!(stats::probability(&[&x, &y], &xs.lt(ys))?, ctx.rational(1, 2));
/// assert!(stats::probability(&[&x, &y], &(xs + ys).gt(&zero)).is_err());
/// # Ok::<(), SymplexError>(())
/// ```
pub fn probability(vars: &[&RandomVariable], event: &BoolEx) -> Result<Ex, SymplexError> {
    const OP: &str = "stats::probability";
    let ctx = check_vars(OP, vars)?;
    check_context(OP, &ctx, event)?;
    let unsupported = || {
        SymplexError::NotImplemented(format!(
            "probability of the joint event `{event}`: supported are conjunctions of relations \
             each in a single listed variable (`X > a`, `a ≤ X`, `X = a`, …), and a single \
             ordering `X < Y` / `X ≤ Y` of two continuous variables"
        ))
    };
    let rels = flatten_relations(event).ok_or_else(unsupported)?;
    // Sort the relations: per-variable ones by variable, orderings apart.
    let mut per_var: Vec<Vec<BoolEx>> = vec![Vec::new(); vars.len()];
    let mut orderings: Vec<(Rel, usize, usize)> = Vec::new();
    for rel in rels {
        let mentioned: Vec<usize> = (0..vars.len())
            .filter(|&i| {
                let s = vars[i].symbol();
                rel.lhs.contains(s) || rel.rhs.contains(s)
            })
            .collect();
        match mentioned.as_slice() {
            [i] => per_var[*i].push(rel.node),
            [i, j] => {
                // Only the bare ordering of the two symbols.
                let (si, sj) = (vars[*i].symbol(), vars[*j].symbol());
                if rel.lhs == *si && rel.rhs == *sj {
                    orderings.push((rel.kind, *i, *j));
                } else if rel.lhs == *sj && rel.rhs == *si {
                    orderings.push((rel.kind, *j, *i));
                } else {
                    return Err(unsupported());
                }
            }
            _ => return Err(unsupported()),
        }
    }
    if !orderings.is_empty() {
        // Exactly one ordering and nothing else.
        let [(kind, l, r)] = orderings.as_slice() else {
            return Err(unsupported());
        };
        if per_var.iter().any(|parts| !parts.is_empty()) {
            return Err(unsupported());
        }
        let (left, right) = (vars[*l], vars[*r]);
        return match kind {
            // `X_l > X_r` / `X_l ≥ X_r`, i.e. `X_r < X_l`.
            Rel::Gt | Rel::Ge => probability_less(right, left),
            Rel::Eq => {
                if left.distribution().is_continuous() && right.distribution().is_continuous() {
                    Ok(ctx.zero())
                } else {
                    Err(unsupported())
                }
            }
        };
    }
    // A rectangle: independence makes the probability a product.
    let mut total = ctx.one();
    for (v, parts) in vars.iter().zip(per_var) {
        let mut parts = parts.into_iter();
        let Some(first) = parts.next() else {
            continue;
        };
        let conjunction = parts.fold(first, |acc, b| acc.and(&b));
        total *= v.probability(&conjunction)?;
    }
    Ok(total.simplify())
}

/// `P(L < U)` for independent continuous `lower`, `upper`.
fn probability_less(lower: &RandomVariable, upper: &RandomVariable) -> Result<Ex, SymplexError> {
    let ctx = lower.context();
    let not_implemented = || {
        SymplexError::NotImplemented(format!(
            "P({} < {}): the ordering of two variables is supported for two continuous \
             variables whose distribution functions have closed forms",
            lower.symbol(),
            upper.symbol()
        ))
    };
    // Two normals: L − U ~ Normal(μ_L − μ_U, √(σ_L² + σ_U²)), so
    // P(L < U) = P(L − U < 0) — exact, no erf integral left over.
    if let (Some(x), Some(y)) = (
        lower.distribution().downcast_ref::<Normal>(),
        upper.distribution().downcast_ref::<Normal>(),
    ) {
        let diff = Distribution::normal(
            (&x.mean - &y.mean).simplify(),
            (x.std.powi(2) + y.std.powi(2)).sqrt().simplify(),
        );
        return diff.probability_of(&Support::from_pieces(
            Kind::Continuous,
            vec![super::support::Piece::Interval(
                crate::base::interval::Interval::open(ctx.neg_infinity(), ctx.zero()),
            )],
        ));
    }
    let (sl, su) = (lower.support(), upper.support());
    let (Some(iv_l), Some(iv_u)) = (sl.as_interval(), su.as_interval()) else {
        return Err(not_implemented());
    };
    let (lo_l, hi_l) = (&iv_l.lower, &iv_l.upper);
    let (lo_u, hi_u) = (&iv_u.lower, &iv_u.upper);
    if sl.kind() != Kind::Continuous || su.kind() != Kind::Continuous {
        return Err(not_implemented());
    }
    let t = lower.symbol();
    let (Some(_), Some(cdf_upper)) = (
        lower.distribution().family().cdf(t),
        upper.distribution().family().cdf(t),
    ) else {
        return Err(not_implemented());
    };
    // P(L < U) = ∫ f_L(t)·(1 − F_U(t)) dt.  The closed form of F_U is valid
    // on U's support only: below it `1 − F_U = 1`, which contributes
    // P(L < lo_U); above it the factor is 0.
    let mut total = ctx.zero();
    if !is_neg_inf(lo_u) {
        total += lower.probability(&t.lt(lo_u))?;
    }
    let both = Support::interval(lo_l.clone(), hi_l.clone())
        .intersect(&Support::interval(lo_u.clone(), hi_u.clone()))
        .ok_or_else(not_implemented)?;
    let integrand = lower.density(t) * (ctx.one() - cdf_upper);
    total += lower.distribution().integrate_over(&integrand, t, &both);
    Ok(total.simplify())
}

// ═══════════════════════════════════════════════════════════════════════════
// Conditioning on an event of the same variable
// ═══════════════════════════════════════════════════════════════════════════

/// `E[g(X) | event] = E[g(X)·1_event] / P(event)` for an event in the same
/// variable: `g · density` integrated (summed) over the part of the
/// support where the event holds, divided by its probability.  SymPy:
/// `E(X, X > 0)` (= `√(2/π)` for a standard normal), `E(X**2, X > 0)`.
///
/// For a discrete variable and the event `X = a` (positive probability)
/// the conditional law is the point mass at `a`, so the result is `g(a)`.
/// The result may contain an unevaluated `Integral`/`Sum` when the
/// integrator cannot close the form.
///
/// # Errors
///
/// [`SymplexError::NotImplemented`] for events of another shape than
/// `RandomVariable::probability` accepts (relations `X < a`, `X ≤ a`,
/// `X > a`, `X ≥ a`, `X = a` and their conjunctions);
/// [`SymplexError::InvalidArgument`] when the event has probability zero
/// (`X = a` for a continuous variable) or the expressions live in another
/// context.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::{self, Distribution, RandomVariable};
///
/// let ctx = Context::new();
/// let x = RandomVariable::new(&ctx, "X", Distribution::normal(ctx.int(0), ctx.int(1)));
/// let xs = x.symbol();
/// let positive = xs.gt(&ctx.int(0));
///
/// // E[X | X > 0] = √(2/π) ≈ 0.7979.
/// let half_normal_mean = stats::conditional_expectation(&x, xs, &positive)?;
/// assert!((half_normal_mean.eval_f64()? - (2.0 / std::f64::consts::PI).sqrt()).abs() < 1e-12);
/// // E[X² | X > 0] = 1.
/// assert_eq!(stats::conditional_expectation(&x, &xs.powi(2), &positive)?, ctx.int(1));
/// # Ok::<(), SymplexError>(())
/// ```
pub fn conditional_expectation(
    x: &RandomVariable,
    g: &Ex,
    event: &BoolEx,
) -> Result<Ex, SymplexError> {
    const OP: &str = "stats::conditional_expectation";
    let ctx = x.context();
    check_context(OP, &ctx, g)?;
    check_context(OP, &ctx, event)?;
    let region = x.event_region(event)?;
    let p = x.probability(event)?;
    if p.is_zero() == Some(true) {
        return Err(invalid(
            OP,
            format!("the event `{event}` has probability zero"),
        ));
    }
    let numerator = x.distribution().expectation_over(g, x.symbol(), &region)?;
    Ok((numerator / p).simplify())
}

/// `P(event | given) = P(event ∧ given) / P(given)` for two events in the
/// same variable.  SymPy: `P(X > 1, X > 0)` (= `2·P(X > 1)` for a
/// standard normal).
///
/// # Errors
///
/// [`SymplexError::NotImplemented`] for events of another shape than
/// `RandomVariable::probability` accepts; [`SymplexError::InvalidArgument`]
/// when `given` has probability zero or the events live in another
/// context.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::{self, Distribution, RandomVariable};
///
/// let ctx = Context::new();
/// let x = RandomVariable::new(&ctx, "X", Distribution::normal(ctx.int(0), ctx.int(1)));
/// let xs = x.symbol();
///
/// let p = stats::conditional_probability(&x, &xs.gt(&ctx.int(1)), &xs.gt(&ctx.int(0)))?;
/// let twice = 2 * x.probability(&xs.gt(&ctx.int(1)))?;
/// assert!((p.eval_f64()? - twice.eval_f64()?).abs() < 1e-12);
/// # Ok::<(), SymplexError>(())
/// ```
pub fn conditional_probability(
    x: &RandomVariable,
    event: &BoolEx,
    given: &BoolEx,
) -> Result<Ex, SymplexError> {
    const OP: &str = "stats::conditional_probability";
    let ctx = x.context();
    check_context(OP, &ctx, event)?;
    check_context(OP, &ctx, given)?;
    let p_given = x.probability(given)?;
    if p_given.is_zero() == Some(true) {
        return Err(invalid(
            OP,
            format!("the conditioning event `{given}` has probability zero"),
        ));
    }
    let p_both = x.probability(&event.and(given))?;
    Ok((p_both / p_given).simplify())
}

// ═══════════════════════════════════════════════════════════════════════════
// Derived descriptive quantities
// ═══════════════════════════════════════════════════════════════════════════

/// The (differential) entropy `H[X] = −E[ln f(X)]` in nats — for a
/// discrete variable the Shannon entropy `−Σ p(k) ln p(k)`.  SymPy:
/// `entropy(X)` (`log(2)/2 + 1/2 + log(pi)/2` for a standard normal).
///
/// The logarithm of the density is expanded (`ln(ab) → ln a + ln b`,
/// `ln eᵘ → u`; the density is positive on the support and its parameters
/// are real, which justifies the expansion) and handed to
/// [`expectation`](RandomVariable::expectation), so families whose
/// log-density is a polynomial in `X` come out in closed form through the
/// moments: `Normal(μ, σ)` gives `½ ln(2π) + ln σ + ½`, i.e. `½ ln(2πeσ²)`.
/// Otherwise the result may contain an unevaluated `Integral`/`Sum`.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::{self, Distribution, RandomVariable};
///
/// let ctx = Context::new();
/// let x = RandomVariable::new(&ctx, "X", Distribution::normal(ctx.int(0), ctx.int(1)));
/// let h = stats::entropy(&x);
/// // ½ ln(2πe) ≈ 1.4189
/// let expected = (2 * ctx.pi() * ctx.e()).ln() / 2;
/// assert_eq!((h - expected).expand_log().simplify(), ctx.int(0));
/// ```
pub fn entropy(x: &RandomVariable) -> Ex {
    x.entropy()
}

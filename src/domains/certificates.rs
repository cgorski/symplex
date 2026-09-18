//! Exact, machine-checkable certificates that a polynomial is non-negative
//! on a box.
//!
//! [`prove_nonnegative_on_box`] searches for a **Handelman certificate**: a
//! representation
//!
//! ```text
//! goal(x) = Σₖ λₖ · Πᵢ (xᵢ − lᵢ)^{aₖᵢ} · (uᵢ − xᵢ)^{bₖᵢ},      λₖ ≥ 0
//! ```
//!
//! over the box `lᵢ ≤ xᵢ ≤ uᵢ`, with all products of total degree at most
//! `degree`.  Every factor is non-negative on the box, so the identity is a
//! proof of `goal ≥ 0` there.  Handelman's theorem guarantees such a
//! certificate exists for some degree whenever `goal` is strictly positive
//! on the (compact) box; when `goal` touches zero in the interior none may
//! exist, and the search reports that honestly.
//!
//! The search is an exact rational linear program ([`linprog`](crate::linprog)):
//! the products are expanded with [`Poly`] arithmetic, their coefficient
//! vectors form the columns of a matrix, and `nonneg_combination` finds
//! `λ ≥ 0` exactly or returns a Farkas vector proving that no certificate of
//! that degree exists.  Before a certificate is returned it is
//! **re-verified** by recomputing `Σ λₖ·productₖ` with exact polynomial
//! arithmetic and checking structural equality with `goal` — so a
//! [`Certificate`] can be trusted without trusting the LP solver.
//!
//! [`Certificate::to_lean`] renders the result as a Lean 4 / Mathlib
//! theorem whose proof is `nlinarith` fed with exactly the products that
//! appear with non-zero weight (so it is a linear-arithmetic check, not a
//! search).
//!
//! # Example
//!
//! ```
//! use symplex::prelude::*;
//! use symplex::certificates::{prove_nonnegative_on_box, BoxOutcome};
//!
//! let ctx = Context::new();
//! let (r, f) = (ctx.symbol("r"), ctx.symbol("f"));
//! // 1/4 − (r − f/2)² ≥ 0 on [0, 1/2] × [0, 1]
//! let goal = ctx.rational(1, 4) - (&r - &f / 2).powi(2);
//! let bounds = [
//!     (r.clone(), ctx.int(0), ctx.rational(1, 2)),
//!     (f.clone(), ctx.int(0), ctx.int(1)),
//! ];
//! match prove_nonnegative_on_box(&goal, &bounds, 2).unwrap() {
//!     BoxOutcome::Proved(cert) => {
//!         assert!(cert.verify());
//!         assert!(cert.to_lean("quarter_bound").unwrap().contains("nlinarith"));
//!     }
//!     other => panic!("expected a certificate, got {other:?}"),
//! }
//! ```

use std::fmt;

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::Zero;

use crate::api::context::Context;
use crate::api::expr::Ex;
use crate::api::poly_ex::Poly;
use crate::base::errors::SymplexError;
use crate::domains::linprog::{Feasibility, LpProblem, LpStatus, nonneg_combination};
use crate::output::lean::{LeanOpts, lean_ident};

type Q = Ratio<BigInt>;

fn invalid(reason: impl Into<String>) -> SymplexError {
    SymplexError::InvalidArgument {
        operation: "prove_nonnegative_on_box",
        reason: reason.into(),
    }
}

/// One variable of the box: `lo ≤ var ≤ hi` with exact rational endpoints.
#[derive(Clone, Debug)]
pub struct BoxBound {
    /// The variable.
    pub var: Ex,
    /// Lower endpoint (a rational literal).
    pub lo: Ex,
    /// Upper endpoint (a rational literal, `> lo`).
    pub hi: Ex,
}

/// One product `Πᵢ (xᵢ − lᵢ)^{aᵢ} (uᵢ − xᵢ)^{bᵢ}` of a Handelman certificate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HandelmanTerm {
    /// Exponents `aᵢ` of the lower-bound factors `xᵢ − lᵢ`, one per variable.
    pub lower_powers: Vec<u32>,
    /// Exponents `bᵢ` of the upper-bound factors `uᵢ − xᵢ`, one per variable.
    pub upper_powers: Vec<u32>,
    /// The weight `λₖ > 0`.
    pub weight: Q,
}

impl HandelmanTerm {
    /// Total degree of the product.
    pub fn degree(&self) -> u32 {
        self.lower_powers.iter().sum::<u32>() + self.upper_powers.iter().sum::<u32>()
    }
}

/// A verified Handelman certificate: `goal = Σ weightₖ · productₖ` on the
/// box, every `weightₖ > 0`.
#[derive(Clone, Debug)]
pub struct Certificate {
    goal: Poly,
    bounds: Vec<BoxBound>,
    terms: Vec<HandelmanTerm>,
}

impl Certificate {
    /// The polynomial that was proved non-negative.
    pub fn goal(&self) -> &Poly {
        &self.goal
    }

    /// The box.
    pub fn bounds(&self) -> &[BoxBound] {
        &self.bounds
    }

    /// The weighted products, in the order the search enumerated them.
    pub fn terms(&self) -> &[HandelmanTerm] {
        &self.terms
    }

    /// Largest total degree among the products.
    pub fn degree(&self) -> u32 {
        self.terms
            .iter()
            .map(HandelmanTerm::degree)
            .max()
            .unwrap_or(0)
    }

    /// The product `Πᵢ (xᵢ − lᵢ)^{aᵢ} (uᵢ − xᵢ)^{bᵢ}` of one term as a `Poly`.
    pub fn product(&self, term: &HandelmanTerm) -> Poly {
        let ctx = self.goal.context();
        let gens: Vec<&Ex> = self.goal.gens().iter().collect();
        let mut acc = Poly::one(&ctx, &gens).unwrap_or_else(|_| self.goal.clone());
        for (i, b) in self.bounds.iter().enumerate() {
            let lower = &b.var - &b.lo;
            let upper = &b.hi - &b.var;
            for _ in 0..term.lower_powers.get(i).copied().unwrap_or(0) {
                if let Some(p) = Poly::new(&lower, &gens)
                    && let Ok(m) = acc.mul(&p)
                {
                    acc = m;
                }
            }
            for _ in 0..term.upper_powers.get(i).copied().unwrap_or(0) {
                if let Some(p) = Poly::new(&upper, &gens)
                    && let Ok(m) = acc.mul(&p)
                {
                    acc = m;
                }
            }
        }
        acc
    }

    /// Recompute `Σ weightₖ · productₖ` with exact polynomial arithmetic and
    /// compare it structurally with the goal.  `prove_nonnegative_on_box`
    /// only returns certificates for which this holds; call it again when
    /// the certificate has crossed a trust boundary (serialisation, another
    /// process).
    pub fn verify(&self) -> bool {
        let ctx = self.goal.context();
        let gens: Vec<&Ex> = self.goal.gens().iter().collect();
        let Ok(mut acc) = Poly::zero(&ctx, &gens) else {
            return false;
        };
        for t in &self.terms {
            if t.weight <= Q::zero() {
                return false;
            }
            let w = ctx.from_ratio(t.weight.clone());
            let Ok(scaled) = self.product(t).scale(&w) else {
                return false;
            };
            let Ok(sum) = acc.add(&scaled) else {
                return false;
            };
            acc = sum;
        }
        acc.equals(&self.goal)
    }

    /// The product of one term in factored form, `(x₁ − l₁)^a₁ · (u₁ − x₁)^b₁ · …`
    /// (not expanded; see [`product`](Self::product) for the `Poly`).
    pub fn product_expr(&self, term: &HandelmanTerm) -> Ex {
        let ctx = self.goal.context();
        let mut acc = ctx.one();
        for (i, b) in self.bounds.iter().enumerate() {
            let a = i64::from(term.lower_powers.get(i).copied().unwrap_or(0));
            let e = i64::from(term.upper_powers.get(i).copied().unwrap_or(0));
            if a > 0 {
                acc *= (&b.var - &b.lo).powi(a);
            }
            if e > 0 {
                acc *= (&b.hi - &b.var).powi(e);
            }
        }
        acc
    }

    /// The certificate identity `goal = Σ λₖ·productₖ` as an expression pair
    /// `(lhs, rhs)`, with the products kept in factored form.
    pub fn identity(&self) -> (Ex, Ex) {
        let ctx = self.goal.context();
        let mut rhs = ctx.zero();
        for t in &self.terms {
            rhs += ctx.from_ratio(t.weight.clone()) * self.product_expr(t);
        }
        (self.goal.to_ex(), rhs)
    }

    /// A Lean 4 / Mathlib theorem proving `0 ≤ goal` on the box.
    ///
    /// The statement takes each variable as a real and each bound as a
    /// hypothesis (`h_r_lo : 0 ≤ r`, `h_r_hi : r ≤ 1 / 2`, …); the proof is
    /// `nlinarith` supplied with every product of the certificate as a
    /// `mul_nonneg` hint, so the only search Lean performs is linear
    /// arithmetic over the exact identity.  Degree-1 certificates (no
    /// products) use `linarith`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::NotImplemented`] if the goal cannot be rendered (see
    /// [`Ex::to_lean`]).
    pub fn to_lean(&self, theorem_name: &str) -> Result<String, SymplexError> {
        self.to_lean_with(theorem_name, &LeanOpts::default())
    }

    /// [`to_lean`](Self::to_lean) with explicit rendering options.
    pub fn to_lean_with(
        &self,
        theorem_name: &str,
        opts: &LeanOpts,
    ) -> Result<String, SymplexError> {
        let real = &opts.real_type;
        let vars: Vec<String> = self
            .bounds
            .iter()
            .map(|b| lean_ident(&b.var.to_string()))
            .collect();
        // Which bounds the certificate actually uses (others are named with a
        // leading underscore so Mathlib's unused-variable linter stays quiet).
        let n = self.bounds.len();
        let mut uses_lo = vec![false; n];
        let mut uses_hi = vec![false; n];
        for t in &self.terms {
            for i in 0..n {
                uses_lo[i] |= t.lower_powers.get(i).copied().unwrap_or(0) > 0;
                uses_hi[i] |= t.upper_powers.get(i).copied().unwrap_or(0) > 0;
            }
        }
        // Hypotheses.
        let mut hyps: Vec<String> = Vec::new();
        let mut lo_names: Vec<String> = Vec::new();
        let mut hi_names: Vec<String> = Vec::new();
        for (i, (b, v)) in self.bounds.iter().zip(&vars).enumerate() {
            let lo = b.lo.to_lean_with(opts)?;
            let hi = b.hi.to_lean_with(opts)?;
            let base = v.trim_matches(['«', '»']);
            let lo_name = format!("{}h_{base}_lo", if uses_lo[i] { "" } else { "_" });
            let hi_name = format!("{}h_{base}_hi", if uses_hi[i] { "" } else { "_" });
            hyps.push(format!("({lo_name} : {lo} ≤ {v})"));
            hyps.push(format!("({hi_name} : {v} ≤ {hi})"));
            lo_names.push(lo_name);
            hi_names.push(hi_name);
        }
        let goal = self.goal.to_ex().to_lean_with(opts)?;

        // Non-negativity of every product with non-zero weight, built from
        // the bound hypotheses with `sub_nonneg.mpr` and `mul_nonneg`.
        let mut hints: Vec<String> = Vec::new();
        let mut max_factors = 0usize;
        for t in &self.terms {
            let mut factors: Vec<String> = Vec::new();
            for i in 0..n {
                for _ in 0..t.lower_powers.get(i).copied().unwrap_or(0) {
                    factors.push(format!("sub_nonneg.mpr {}", lo_names[i]));
                }
                for _ in 0..t.upper_powers.get(i).copied().unwrap_or(0) {
                    factors.push(format!("sub_nonneg.mpr {}", hi_names[i]));
                }
            }
            let Some((first, rest)) = factors.split_first() else {
                continue; // the constant term needs no hint
            };
            max_factors = max_factors.max(factors.len());
            let mut acc = first.clone();
            for f in rest {
                acc = format!("mul_nonneg ({acc}) ({f})");
            }
            if !hints.contains(&acc) {
                hints.push(acc);
            }
        }

        let sig = format!(
            "theorem {} ({} : {real}) {} :\n    0 ≤ {goal} := by\n",
            lean_ident(theorem_name),
            vars.join(" "),
            hyps.join(" ")
        );
        // With every product supplied as a hint, the remaining step is linear
        // arithmetic over the exact identity; `nlinarith` is only needed to
        // move the products' atoms into `linarith`'s view.
        let tactic = if hints.is_empty() {
            "  linarith".to_string()
        } else if max_factors <= 1 {
            format!("  linarith [{}]", hints.join(", "))
        } else {
            format!("  nlinarith [{}]", hints.join(", "))
        };
        Ok(format!("{sig}{tactic}\n"))
    }
}

impl fmt::Display for Certificate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (lhs, rhs) = self.identity();
        write!(f, "{lhs} = {rhs}")?;
        for b in &self.bounds {
            write!(f, ", {} ≤ {} ≤ {}", b.lo, b.var, b.hi)?;
        }
        Ok(())
    }
}

/// Result of [`prove_nonnegative_on_box`].
#[derive(Clone, Debug)]
pub enum BoxOutcome {
    /// A verified certificate: the goal is non-negative on the box.
    Proved(Certificate),
    /// The goal is negative at this point of the box (exact rational
    /// coordinates, in variable order) — the inequality is false.
    Refuted {
        /// A point of the box where the goal is negative.
        point: Vec<Q>,
        /// The (negative) value of the goal there.
        value: Q,
    },
    /// No certificate of the requested degree exists and no counterexample
    /// was found on the sampled grid.  Try a higher `degree`; if the goal
    /// has a zero in the interior of the box, Handelman certificates may
    /// not exist at any degree.
    Unknown {
        /// The Farkas vector (one entry per monomial of the coefficient
        /// system) proving that no certificate of this degree exists.
        farkas: Option<Vec<Q>>,
        /// The degree that was searched.
        degree: u32,
    },
}

impl BoxOutcome {
    /// The certificate, if proved.
    pub fn certificate(&self) -> Option<&Certificate> {
        match self {
            BoxOutcome::Proved(c) => Some(c),
            _ => None,
        }
    }

    /// `true` for [`BoxOutcome::Proved`].
    pub fn is_proved(&self) -> bool {
        matches!(self, BoxOutcome::Proved(_))
    }
}

/// Enumerate exponent vectors `(a₁..aₙ, b₁..bₙ)` with total degree ≤ `degree`.
fn products_up_to(n: usize, degree: u32) -> Vec<(Vec<u32>, Vec<u32>)> {
    let slots = 2 * n;
    let mut out = Vec::new();
    let mut current = vec![0u32; slots];
    // Iterative odometer over compositions with bounded sum.
    fn rec(
        slot: usize,
        remaining: u32,
        current: &mut Vec<u32>,
        n: usize,
        out: &mut Vec<(Vec<u32>, Vec<u32>)>,
    ) {
        if slot == current.len() {
            out.push((current[..n].to_vec(), current[n..].to_vec()));
            return;
        }
        for e in 0..=remaining {
            current[slot] = e;
            rec(slot + 1, remaining - e, current, n, out);
        }
        current[slot] = 0;
    }
    rec(0, degree, &mut current, n, &mut out);
    // Products containing both (x − l) and (u − x) for the same variable
    // are kept: they are legitimate Handelman terms.
    out
}

/// Exact rational value of `goal` at a point.
fn value_at(goal: &Poly, point: &[Q]) -> Option<Q> {
    let ctx = goal.context();
    let vals: Vec<Ex> = point.iter().map(|q| ctx.from_ratio(q.clone())).collect();
    let refs: Vec<&Ex> = vals.iter().collect();
    goal.eval(&refs).ok()?.as_rational()
}

/// Search a grid of `steps + 1` points per axis (endpoints included) for a
/// point where the goal is negative.
fn find_counterexample(goal: &Poly, bounds: &[(Q, Q)], steps: u32) -> Option<(Vec<Q>, Q)> {
    let n = bounds.len();
    let total = (u64::from(steps) + 1).checked_pow(n as u32)?;
    if total > 200_000 {
        return None;
    }
    let mut idx = vec![0u32; n];
    loop {
        let point: Vec<Q> = idx
            .iter()
            .zip(bounds)
            .map(|(&k, (lo, hi))| lo + (hi - lo) * Q::new(BigInt::from(k), BigInt::from(steps)))
            .collect();
        if let Some(v) = value_at(goal, &point)
            && v < Q::zero()
        {
            return Some((point, v));
        }
        // Increment the odometer.
        let mut pos = 0;
        loop {
            if pos == n {
                return None;
            }
            if idx[pos] < steps {
                idx[pos] += 1;
                break;
            }
            idx[pos] = 0;
            pos += 1;
        }
    }
}

/// `nonneg_combination` with the objective `min Σ (1 + degₖ)·λₖ`, so the
/// certificate uses as few and as low-degree products as the exact LP can
/// find.  Falls back to plain feasibility if the weighted problem is
/// (numerically impossible here, but defensively) not `Optimal`.
fn sparse_nonneg_combination(
    columns: &[Vec<Q>],
    target: &[Q],
    exponents: &[(Vec<u32>, Vec<u32>)],
) -> Result<Feasibility, SymplexError> {
    let m = target.len();
    let cost: Vec<Q> = exponents
        .iter()
        .map(|(a, b)| {
            let deg: u32 = a.iter().sum::<u32>() + b.iter().sum::<u32>();
            Q::from_integer(BigInt::from(1 + u64::from(deg)))
        })
        .collect();
    let mut lp = LpProblem::minimize(cost);
    for i in 0..m {
        let row: Vec<Q> = columns.iter().map(|c| c[i].clone()).collect();
        lp = lp.eq(row, target[i].clone());
    }
    let sol = lp.solve()?;
    match sol.status {
        LpStatus::Optimal => Ok(Feasibility::Feasible(sol.x)),
        LpStatus::Infeasible => Ok(Feasibility::Infeasible { farkas: sol.farkas }),
        LpStatus::Unbounded => nonneg_combination(columns, target),
    }
}

/// Prove `goal ≥ 0` on the box `bounds` (each `(var, lo, hi)` with rational
/// literal endpoints) by a Handelman certificate of total degree at most
/// `degree`, or refute it with an exact counterexample.
///
/// `goal` must be a polynomial with rational coefficients in exactly the
/// box variables.  The search cost grows with `C(2n + degree, degree)`
/// products; `degree ≤ 4` with a handful of variables is instantaneous,
/// `degree = 6` in three variables is a few hundred columns.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if `bounds` is empty, an endpoint is
/// not a rational literal, `lo ≥ hi`, a variable repeats, or `goal` is not a
/// rational-coefficient polynomial in the box variables.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::certificates::{prove_nonnegative_on_box, BoxOutcome};
///
/// let ctx = Context::new();
/// let x = ctx.symbol("x");
/// let bounds = [(x.clone(), ctx.int(0), ctx.int(1))];
/// // x(1 − x) ≥ 0 on [0, 1]: the certificate is the single product itself.
/// let out = prove_nonnegative_on_box(&(&x * (1 - &x)), &bounds, 2).unwrap();
/// assert!(out.is_proved());
/// // x − 1/2 is negative near 0: refuted with an exact witness.
/// match prove_nonnegative_on_box(&(&x - ctx.rational(1, 2)), &bounds, 2).unwrap() {
///     BoxOutcome::Refuted { value, .. } => assert!(value < num_rational::Ratio::from_integer(0.into())),
///     other => panic!("{other:?}"),
/// }
/// ```
pub fn prove_nonnegative_on_box(
    goal: &Ex,
    bounds: &[(Ex, Ex, Ex)],
    degree: u32,
) -> Result<BoxOutcome, SymplexError> {
    if bounds.is_empty() {
        return Err(invalid("at least one bounded variable is required"));
    }
    let ctx: Context = goal.context();
    let mut vars: Vec<&Ex> = Vec::with_capacity(bounds.len());
    let mut q_bounds: Vec<(Q, Q)> = Vec::with_capacity(bounds.len());
    let mut box_bounds: Vec<BoxBound> = Vec::with_capacity(bounds.len());
    for (var, lo, hi) in bounds {
        if vars.contains(&var) {
            return Err(invalid(format!("variable `{var}` is bounded twice")));
        }
        let (Some(l), Some(h)) = (lo.eval().as_rational(), hi.eval().as_rational()) else {
            return Err(invalid(format!(
                "bounds of `{var}` must be rational literals, got [{lo}, {hi}]"
            )));
        };
        if l >= h {
            return Err(invalid(format!(
                "bounds of `{var}` must satisfy lo < hi, got [{lo}, {hi}]"
            )));
        }
        vars.push(var);
        q_bounds.push((l, h));
        box_bounds.push(BoxBound {
            var: var.clone(),
            lo: lo.eval(),
            hi: hi.eval(),
        });
    }
    let goal_poly = Poly::new(goal, &vars).ok_or_else(|| {
        invalid("goal must be a polynomial in the box variables (other symbols or non-polynomial operations found)")
    })?;
    if !goal_poly.has_rational_coeffs() {
        return Err(invalid(
            "goal must have rational coefficients (parameters are not supported)",
        ));
    }

    // 1. Cheap exact refutation on a grid.
    if let Some((point, value)) = find_counterexample(&goal_poly, &q_bounds, 8) {
        return Ok(BoxOutcome::Refuted { point, value });
    }

    // 2. Build the products and their coefficient vectors.
    let n = vars.len();
    let lower: Vec<Poly> = box_bounds
        .iter()
        .map(|b| {
            Poly::new(&(&b.var - &b.lo), &vars).ok_or_else(|| invalid("internal: bound factor"))
        })
        .collect::<Result<_, _>>()?;
    let upper: Vec<Poly> = box_bounds
        .iter()
        .map(|b| {
            Poly::new(&(&b.hi - &b.var), &vars).ok_or_else(|| invalid("internal: bound factor"))
        })
        .collect::<Result<_, _>>()?;
    let exponents = products_up_to(n, degree);
    let one = Poly::one(&ctx, &vars)?;
    let mut products: Vec<Poly> = Vec::with_capacity(exponents.len());
    for (a, b) in &exponents {
        let mut acc = one.clone();
        for i in 0..n {
            for _ in 0..a[i] {
                acc = acc.mul(&lower[i])?;
            }
            for _ in 0..b[i] {
                acc = acc.mul(&upper[i])?;
            }
        }
        products.push(acc);
    }
    let mut all: Vec<&Poly> = products.iter().collect();
    all.push(&goal_poly);
    let monos = Poly::monomial_basis(&all)?;
    let coeff_vec = |p: &Poly| -> Result<Vec<Q>, SymplexError> {
        monos
            .iter()
            .map(|m| {
                p.coeff_monomial(m)?
                    .as_rational()
                    .ok_or_else(|| invalid("internal: non-rational coefficient"))
            })
            .collect()
    };
    let columns: Vec<Vec<Q>> = products.iter().map(coeff_vec).collect::<Result<_, _>>()?;
    let target = coeff_vec(&goal_poly)?;

    // 3. Exact LP: goal = Σ λₖ productₖ, λ ≥ 0.  Minimising Σ (1 + degₖ)·λₖ
    // prefers sparse, low-degree certificates (shorter Lean proofs) over an
    // arbitrary feasible vertex.
    match sparse_nonneg_combination(&columns, &target, &exponents)? {
        Feasibility::Feasible(lambda) => {
            let terms: Vec<HandelmanTerm> = exponents
                .iter()
                .zip(&lambda)
                .filter(|(_, w)| **w > Q::zero())
                .map(|((a, b), w)| HandelmanTerm {
                    lower_powers: a.clone(),
                    upper_powers: b.clone(),
                    weight: w.clone(),
                })
                .collect();
            let cert = Certificate {
                goal: goal_poly,
                bounds: box_bounds,
                terms,
            };
            if !cert.verify() {
                return Err(SymplexError::ComputationFailed {
                    operation: "prove_nonnegative_on_box",
                    reason:
                        "the LP solution did not reproduce the goal under exact re-verification"
                            .into(),
                });
            }
            Ok(BoxOutcome::Proved(cert))
        }
        Feasibility::Infeasible { farkas } => {
            // A finer grid before giving up.
            if let Some((point, value)) = find_counterexample(&goal_poly, &q_bounds, 32) {
                return Ok(BoxOutcome::Refuted { point, value });
            }
            Ok(BoxOutcome::Unknown { farkas, degree })
        }
    }
}

/// Convenience: is `goal ≥ 0` on the box?  `Some(true)` with a verified
/// certificate up to `max_degree`, `Some(false)` with an exact
/// counterexample, `None` when undecided.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::certificates::is_nonnegative_on_box;
///
/// let ctx = Context::new();
/// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
/// let bounds = [(x.clone(), ctx.int(0), ctx.int(1)), (y.clone(), ctx.int(0), ctx.int(1))];
/// assert_eq!(is_nonnegative_on_box(&(1 - &x * &y), &bounds, 4), Some(true));
/// assert_eq!(is_nonnegative_on_box(&(&x * &y - 1), &bounds, 4), Some(false));
/// ```
pub fn is_nonnegative_on_box(goal: &Ex, bounds: &[(Ex, Ex, Ex)], max_degree: u32) -> Option<bool> {
    for d in 1..=max_degree.max(1) {
        match prove_nonnegative_on_box(goal, bounds, d) {
            Ok(BoxOutcome::Proved(_)) => return Some(true),
            Ok(BoxOutcome::Refuted { .. }) => return Some(false),
            Ok(BoxOutcome::Unknown { .. }) => continue,
            Err(_) => return None,
        }
    }
    None
}

impl Poly {
    /// Express this polynomial as a non-negative combination `Σ λⱼ bⱼ` of
    /// `basis` (same generators), exactly: [`Feasibility::Feasible`] with the
    /// weights, or [`Feasibility::Infeasible`] with a Farkas vector over the
    /// monomials.  Coefficients must be rational.
    ///
    /// This is the linear-algebra core of a Positivstellensatz / Handelman
    /// search; [`prove_nonnegative_on_box`] builds the basis for you.
    ///
    /// # Errors
    ///
    /// `InvalidArgument` if `basis` is empty, generators differ, or a
    /// coefficient is not rational.
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::linprog::Feasibility;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let goal = (&x + 1).powi(2).as_poly(&[&x]).unwrap();
    /// let b1 = (&x + 1).as_poly(&[&x]).unwrap();
    /// let b2 = (&x.powi(2) - 1).as_poly(&[&x]).unwrap();
    /// // (x + 1)² = 2(x + 1) + 1·(x² − 1)
    /// match goal.express_as_nonneg_combination(&[&b1, &b2]).unwrap() {
    ///     Feasibility::Feasible(w) => assert_eq!(w.iter().map(ToString::to_string).collect::<Vec<_>>(), ["2", "1"]),
    ///     other => panic!("{other:?}"),
    /// }
    /// ```
    pub fn express_as_nonneg_combination(
        &self,
        basis: &[&Poly],
    ) -> Result<Feasibility, SymplexError> {
        const OP: &str = "Poly::express_as_nonneg_combination";
        let bad = |reason: &str| SymplexError::InvalidArgument {
            operation: OP,
            reason: reason.into(),
        };
        if basis.is_empty() {
            return Err(bad("basis must not be empty"));
        }
        let mut all: Vec<&Poly> = basis.to_vec();
        all.push(self);
        let monos = Poly::monomial_basis(&all).map_err(|_| bad("generators differ"))?;
        let coeff_vec = |p: &Poly| -> Result<Vec<Q>, SymplexError> {
            monos
                .iter()
                .map(|m| {
                    p.coeff_monomial(m)?
                        .as_rational()
                        .ok_or_else(|| bad("coefficients must be rational"))
                })
                .collect()
        };
        let columns: Vec<Vec<Q>> = basis
            .iter()
            .map(|p| coeff_vec(p))
            .collect::<Result<_, _>>()?;
        let target = coeff_vec(self)?;
        nonneg_combination(&columns, &target)
    }
}

impl Ex {
    /// [`prove_nonnegative_on_box`] as a method.
    pub fn prove_nonnegative_on_box(
        &self,
        bounds: &[(Ex, Ex, Ex)],
        degree: u32,
    ) -> Result<BoxOutcome, SymplexError> {
        prove_nonnegative_on_box(self, bounds, degree)
    }
}

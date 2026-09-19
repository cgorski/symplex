//! Exact, machine-checkable certificates that a polynomial is non-negative
//! on a box, on a half-line, or on a polyhedron whose facets depend on a
//! parameter — each exportable as a Lean 4 / Mathlib proof.
//!
//! # Parametric polyhedra
//!
//! [`prove_nonnegative_on_polyhedron`] takes hypotheses `h₁, …, hₘ ∈
//! ℚ[j, x]`, a goal `g ∈ ℚ[j, x]` and a parameter bound `j ≥ j₀`, and
//! searches for the identity
//!
//! ```text
//! λ(j) · g  =  Σₖ Σ_{a,b} μ_{k,a,b} · jᵃ (j − j₀)ᵇ · hₖ
//!            + Σ_{k≤l} Σ_{a+b≤1} μ_{k,l,a,b} · jᵃ (j − j₀)ᵇ · hₖ hₗ      (optional)
//!            + Σ_{a+b≥1} μ_{a,b} · jᵃ (j − j₀)ᵇ  +  μ₀ ,
//! λ(j) = 1 + Σ_{a≥1} νₐ jᵃ ,        all μ, ν ≥ 0 .
//! ```
//!
//! Every term on the right is non-negative wherever the hypotheses hold
//! and `j ≥ j₀` (powers of `j` itself are only used when `j₀ ≥ 0`), and
//! `λ(j) > 0`, so the identity proves `g ≥ 0` for **every** admissible
//! `j`.  The polynomial multiplier `λ` on the goal is what makes the
//! parametric case work: the Farkas multipliers of `j`-dependent facets
//! are rational functions of `j`, and clearing their denominators puts a
//! polynomial in front of `g`.  With the goal `−1`
//! ([`prove_polyhedron_empty`]) the same identity proves the polyhedron
//! **empty** for every `j ≥ j₀`.  The search is a sequence of exact LPs
//! staged from the smallest basis upwards ([`PolyhedronOpts`]); every
//! [`PolyhedronCertificate`] is re-verified with exact polynomial
//! arithmetic, exports as a theorem ([`PolyhedronCertificate::to_lean`])
//! or as bare proof steps for an existing skeleton
//! ([`PolyhedronCertificate::lean_steps`]).
//!
//! # Boxes
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
//! # Half-lines
//!
//! [`prove_nonnegative_on_halfline`] and [`prove_nonnegative_on_reals`]
//! handle univariate goals on `x ≥ a` / `x ≤ a` / all of ℝ with a shift,
//! a Pólya multiplier `(1 + k)ᴺ` and square factors for interior double
//! zeros ([`HalfLineCertificate`], [`RealLineCertificate`]).
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
use num_traits::{One, Zero};

use crate::api::context::Context;
use crate::api::expr::Ex;
use crate::api::poly_ex::Poly;
use crate::base::errors::SymplexError;
use crate::domains::linprog::{Feasibility, LpProblem, LpStatus, nonneg_combination};
use crate::output::lean::{LeanOpts, MATHLIB_LINE_WIDTH, lean_ident, wrap_lean};

mod polyhedron;
pub use polyhedron::{
    PolyhedronCertificate, PolyhedronCertificateData, PolyhedronLeanNames, PolyhedronLeanSteps,
    PolyhedronOpts, PolyhedronOutcome, PolyhedronTerm, prove_nonnegative_on_polyhedron,
    prove_polyhedron_empty,
};

/// Exact rationals as `"p/q"` strings for the serialisable certificate forms.
pub(crate) mod serial {
    use super::{BigInt, Q, SymplexError};

    pub(crate) fn q_to_str(q: &Q) -> String {
        format!("{}/{}", q.numer(), q.denom())
    }

    pub(crate) fn q_from_str(s: &str, operation: &'static str) -> Result<Q, SymplexError> {
        let bad = || SymplexError::InvalidArgument {
            operation,
            reason: format!("malformed rational `{s}` (expected `p/q`)"),
        };
        let (n, d) = s.split_once('/').unwrap_or((s, "1"));
        let n: BigInt = n.trim().parse().map_err(|_| bad())?;
        let d: BigInt = d.trim().parse().map_err(|_| bad())?;
        if d == BigInt::from(0) {
            return Err(bad());
        }
        Ok(Q::new(n, d))
    }
}

/// Serialisable form of a box [`Certificate`] (see [`Certificate::to_json`]).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CertificateData {
    /// The goal (a polynomial in the box variables).
    pub goal: crate::output::tree::ExprTree,
    /// The box as `(variable, lo, hi)` trees.
    pub bounds: Vec<(
        crate::output::tree::ExprTree,
        crate::output::tree::ExprTree,
        crate::output::tree::ExprTree,
    )>,
    /// Terms as `(lower powers, upper powers, weight "p/q")`.
    pub terms: Vec<(Vec<u32>, Vec<u32>, String)>,
    /// The square factor `g`, if any.
    pub square: Option<crate::output::tree::ExprTree>,
}

/// Serialisable form of a [`HalfLineCertificate`] (see
/// [`HalfLineCertificate::to_json`]).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HalfLineCertificateData {
    /// The goal.
    pub goal: crate::output::tree::ExprTree,
    /// The variable.
    pub var: crate::output::tree::ExprTree,
    /// The endpoint `a`.
    pub endpoint: crate::output::tree::ExprTree,
    /// `"at_least"` (`x ≥ a`) or `"at_most"` (`x ≤ a`).
    pub ray: String,
    /// The Pólya power `N`.
    pub polya_power: u32,
    /// Coefficients of `(1 + k)ᴺ·goal / g²` in ascending powers of `k`, as `"p/q"`.
    pub coefficients: Vec<String>,
    /// The square factor `g`, if any.
    pub square: Option<crate::output::tree::ExprTree>,
}

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

/// A verified Handelman certificate: `goal = square² · Σ weightₖ · productₖ`
/// on the box, every `weightₖ > 0`.  The square factor is `1` unless the
/// goal had even-multiplicity zeros inside the box (see
/// [`prove_nonnegative_on_box`]).
#[derive(Clone, Debug)]
pub struct Certificate {
    goal: Poly,
    bounds: Vec<BoxBound>,
    terms: Vec<HandelmanTerm>,
    /// `g` with `goal = g² · Σ λₖ productₖ`; `None` when `g = 1`.
    square: Option<Poly>,
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

    /// The square factor `g` in `goal = g² · Σ λₖ productₖ`, if any.  It
    /// collects the even-multiplicity factors of the goal (`(x − 1)²·h`
    /// gives `g = x − 1`), which is what lets a goal with interior zeros be
    /// certified: `h` is strictly positive and gets the Handelman part.
    pub fn square(&self) -> Option<&Poly> {
        self.square.as_ref()
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
        if let Some(g) = &self.square {
            let Ok(g2) = g.mul(g) else {
                return false;
            };
            let Ok(prod) = acc.mul(&g2) else {
                return false;
            };
            acc = prod;
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
        if let Some(g) = &self.square {
            rhs = g.to_ex().powi(2) * rhs;
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
        let square_hint = match &self.square {
            Some(g) => Some(format!("sq_nonneg ({})", g.to_ex().to_lean_with(opts)?)),
            None => None,
        };

        // Non-negativity of every product with non-zero weight, built from
        // the bound hypotheses with `sub_nonneg.mpr` and `mul_nonneg`; with
        // a square factor each hint becomes `mul_nonneg (sq_nonneg g) (…)`.
        let mut hints: Vec<String> = Vec::new();
        let mut max_factors = 0usize;
        if let Some(sq) = &square_hint
            && self.terms.iter().any(|t| t.degree() == 0)
        {
            // The constant term of Σ λₖ Pₖ contributes λ₀·g².
            hints.push(sq.clone());
            max_factors = max_factors.max(2);
        }
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
            let mut acc = first.clone();
            for f in rest {
                acc = format!("mul_nonneg ({acc}) ({f})");
            }
            if let Some(sq) = &square_hint {
                acc = format!("mul_nonneg ({sq}) ({acc})");
                max_factors = max_factors.max(factors.len() + 2);
            } else {
                max_factors = max_factors.max(factors.len());
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
        Ok(wrap_lean(&format!("{sig}{tactic}\n"), MATHLIB_LINE_WIDTH))
    }
}

impl Certificate {
    /// The certificate as plain data for serialisation with serde.
    pub fn to_data(&self) -> CertificateData {
        CertificateData {
            goal: self.goal.to_ex().to_tree(),
            bounds: self
                .bounds
                .iter()
                .map(|b| (b.var.to_tree(), b.lo.to_tree(), b.hi.to_tree()))
                .collect(),
            terms: self
                .terms
                .iter()
                .map(|t| {
                    (
                        t.lower_powers.clone(),
                        t.upper_powers.clone(),
                        serial::q_to_str(&t.weight),
                    )
                })
                .collect(),
            square: self.square.as_ref().map(|g| g.to_ex().to_tree()),
        }
    }

    /// Rebuild from data in `ctx` and **re-verify**; data that does not
    /// verify is rejected.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] for malformed data or a failed
    /// verification.
    pub fn from_data(ctx: &Context, data: &CertificateData) -> Result<Self, SymplexError> {
        const OP: &str = "Certificate::from_data";
        let bad = |reason: String| SymplexError::InvalidArgument {
            operation: OP,
            reason,
        };
        let bounds: Vec<BoxBound> = data
            .bounds
            .iter()
            .map(|(v, lo, hi)| BoxBound {
                var: ctx.from_tree(v),
                lo: ctx.from_tree(lo),
                hi: ctx.from_tree(hi),
            })
            .collect();
        if bounds.is_empty() {
            return Err(bad(
                "a certificate needs at least one bounded variable".into()
            ));
        }
        let gens: Vec<&Ex> = bounds.iter().map(|b| &b.var).collect();
        let goal_ex = ctx.from_tree(&data.goal);
        let goal = Poly::new(&goal_ex, &gens).ok_or_else(|| {
            bad(format!(
                "goal `{goal_ex}` is not a polynomial in the box variables"
            ))
        })?;
        let square = match &data.square {
            Some(t) => {
                let e = ctx.from_tree(t);
                Some(
                    Poly::new(&e, &gens)
                        .ok_or_else(|| bad(format!("square factor `{e}` is not a polynomial")))?,
                )
            }
            None => None,
        };
        let mut terms = Vec::with_capacity(data.terms.len());
        for (lo, hi, w) in &data.terms {
            if lo.len() != bounds.len() || hi.len() != bounds.len() {
                return Err(bad(
                    "a term's power vectors must have one entry per variable".into(),
                ));
            }
            terms.push(HandelmanTerm {
                lower_powers: lo.clone(),
                upper_powers: hi.clone(),
                weight: serial::q_from_str(w, OP)?,
            });
        }
        let cert = Certificate {
            goal,
            bounds,
            terms,
            square,
        };
        if !cert.verify() {
            return Err(bad("the certificate data does not verify".into()));
        }
        Ok(cert)
    }

    /// JSON form of [`to_data`](Self::to_data), for crossing a trust
    /// boundary: [`from_json`](Self::from_json) re-verifies before
    /// accepting.
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::certificates::{Certificate, prove_nonnegative_on_box};
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let out = prove_nonnegative_on_box(&(&x * (1 - &x)), &[(x.clone(), ctx.int(0), ctx.int(1))], 2).unwrap();
    /// let json = out.certificate().unwrap().to_json().unwrap();
    /// let back = Certificate::from_json(&Context::new(), &json).unwrap();
    /// assert!(back.verify());
    /// assert_eq!(back.to_string(), out.certificate().unwrap().to_string());
    /// ```
    ///
    /// # Errors
    ///
    /// [`SymplexError::ComputationFailed`] if serialisation fails.
    pub fn to_json(&self) -> Result<String, SymplexError> {
        serde_json::to_string(&self.to_data()).map_err(|e| SymplexError::ComputationFailed {
            operation: "Certificate::to_json",
            reason: e.to_string(),
        })
    }

    /// Parse [`to_json`](Self::to_json) output in `ctx` and re-verify it.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] for malformed JSON or data that
    /// does not verify.
    pub fn from_json(ctx: &Context, json: &str) -> Result<Self, SymplexError> {
        let data: CertificateData =
            serde_json::from_str(json).map_err(|e| SymplexError::InvalidArgument {
                operation: "Certificate::from_json",
                reason: format!("malformed JSON: {e}"),
            })?;
        Self::from_data(ctx, &data)
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

    // 2. Plain Handelman search; on failure, split off the even-multiplicity
    //    factors (`goal = g²·h`) and certify `h`, which has no interior
    //    zeros of even order left.
    match handelman_search(&goal_poly, &vars, &box_bounds, degree, None)? {
        Ok(cert) => Ok(BoxOutcome::Proved(cert)),
        Err(farkas) => {
            if let Some((g, h)) = split_square_factor(&goal_poly, &vars)
                && let Ok(cert) = handelman_search(&h, &vars, &box_bounds, degree, Some(&g))?
            {
                let cert = Certificate {
                    goal: goal_poly.clone(),
                    ..cert
                };
                if cert.verify() {
                    return Ok(BoxOutcome::Proved(cert));
                }
            }
            // A finer grid before giving up.
            if let Some((point, value)) = find_counterexample(&goal_poly, &q_bounds, 32) {
                return Ok(BoxOutcome::Refuted { point, value });
            }
            Ok(BoxOutcome::Unknown { farkas, degree })
        }
    }
}

/// `goal = g² · h` with `g` the product of the even-multiplicity factors
/// (`f^(m div 2)` for each factor `f^m`) and `h` the remaining part
/// (`content · Π f^(m mod 2)`), when the goal has at least one repeated
/// factor.  Uses exact factoring over ℤ (univariate) or the multivariate
/// factoring of `factor_list_all`.
fn split_square_factor(goal: &Poly, vars: &[&Ex]) -> Option<(Poly, Poly)> {
    let e = goal.to_ex();
    let (content, factors) = if vars.len() == 1 {
        e.factor_list(vars[0])
    } else {
        e.factor_list_all()
    };
    if factors.iter().all(|(_, m)| *m < 2) {
        return None;
    }
    let ctx = goal.context();
    let mut g = ctx.one();
    let mut h = content;
    for (f, m) in &factors {
        if *m >= 2 {
            g *= f.powi(i64::from(*m / 2));
        }
        if *m % 2 == 1 {
            h *= f;
        }
    }
    let g = Poly::new(&g, vars)?;
    let h = Poly::new(&h, vars)?;
    // Sanity: g²·h must reproduce the goal exactly.
    let back = g.mul(&g).ok()?.mul(&h).ok()?;
    if !back.equals(goal) {
        return None;
    }
    Some((g, h))
}

/// The Handelman LP for `goal = Σ λₖ Pₖ` over `bounds` up to `degree`.
/// `Ok(Ok(cert))` with a verified certificate (carrying `square`),
/// `Ok(Err(farkas))` when no certificate of that degree exists.
fn handelman_search(
    goal_poly: &Poly,
    vars: &[&Ex],
    box_bounds: &[BoxBound],
    degree: u32,
    square: Option<&Poly>,
) -> Result<Result<Certificate, Option<Vec<Q>>>, SymplexError> {
    let ctx = goal_poly.context();
    let n = vars.len();
    let lower: Vec<Poly> = box_bounds
        .iter()
        .map(|b| {
            Poly::new(&(&b.var - &b.lo), vars).ok_or_else(|| invalid("internal: bound factor"))
        })
        .collect::<Result<_, _>>()?;
    let upper: Vec<Poly> = box_bounds
        .iter()
        .map(|b| {
            Poly::new(&(&b.hi - &b.var), vars).ok_or_else(|| invalid("internal: bound factor"))
        })
        .collect::<Result<_, _>>()?;
    let exponents = products_up_to(n, degree);
    let one = Poly::one(&ctx, vars)?;
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
    all.push(goal_poly);
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
    let target = coeff_vec(goal_poly)?;

    // Minimising Σ (1 + degₖ)·λₖ prefers sparse, low-degree certificates
    // (shorter Lean proofs) over an arbitrary feasible vertex.
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
            // With a square factor the certified polynomial is `square²·goal_poly`;
            // the caller substitutes the original goal.
            let certified = match square {
                Some(g) => g.mul(g)?.mul(goal_poly)?,
                None => goal_poly.clone(),
            };
            let cert = Certificate {
                goal: certified,
                bounds: box_bounds.to_vec(),
                terms,
                square: square.cloned(),
            };
            if !cert.verify() {
                return Err(SymplexError::ComputationFailed {
                    operation: "prove_nonnegative_on_box",
                    reason:
                        "the LP solution did not reproduce the goal under exact re-verification"
                            .into(),
                });
            }
            Ok(Ok(cert))
        }
        Feasibility::Infeasible { farkas } => Ok(Err(farkas)),
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

// ═══════════════════════════════════════════════════════════════════════════
// Half-lines and the real line (univariate)
// ═══════════════════════════════════════════════════════════════════════════

/// Which unbounded domain a [`HalfLineCertificate`] covers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ray {
    /// `x ≥ a`.
    AtLeast,
    /// `x ≤ a`.
    AtMost,
}

/// A verified certificate that a univariate polynomial is non-negative on
/// a half-line.
///
/// With `k = x − a` (or `k = a − x` for [`Ray::AtMost`]) the identity is
///
/// ```text
/// (1 + k)^N · goal(x) = square(x)² · Σᵢ cᵢ kⁱ,      cᵢ ≥ 0,
/// ```
///
/// which proves `goal ≥ 0` for `k ≥ 0`.  `N = 0` is the plain
/// *shift-and-read-off-the-coefficients* certificate (the same sufficient
/// condition `linarith` re-derives in Lean); `N > 0` is a Pólya multiplier,
/// which by Pólya's theorem always exists when `goal` is strictly positive
/// on the closed half-line and has positive leading coefficient.  The
/// square factor collects even-multiplicity zeros, exactly as for the box
/// certificates.
#[derive(Clone, Debug)]
pub struct HalfLineCertificate {
    goal: Poly,
    var: Ex,
    endpoint: Ex,
    ray: Ray,
    polya_power: u32,
    coefficients: Vec<Q>,
    square: Option<Poly>,
}

impl HalfLineCertificate {
    /// The polynomial that was proved non-negative.
    pub fn goal(&self) -> &Poly {
        &self.goal
    }

    /// The variable.
    pub fn var(&self) -> &Ex {
        &self.var
    }

    /// The finite endpoint `a`.
    pub fn endpoint(&self) -> &Ex {
        &self.endpoint
    }

    /// Whether the domain is `x ≥ a` or `x ≤ a`.
    pub fn ray(&self) -> &Ray {
        &self.ray
    }

    /// The Pólya exponent `N` (`0` for a pure shift certificate).
    pub fn polya_power(&self) -> u32 {
        self.polya_power
    }

    /// The non-negative coefficients `cᵢ` of `(1 + k)^N · goal / square²` in
    /// powers of `k`, ascending.
    pub fn coefficients(&self) -> &[Q] {
        &self.coefficients
    }

    /// The square factor `g`, if any.
    pub fn square(&self) -> Option<&Poly> {
        self.square.as_ref()
    }

    /// `k` as an expression: `x − a` or `a − x`.
    pub fn shift_expr(&self) -> Ex {
        match self.ray {
            Ray::AtLeast => &self.var - &self.endpoint,
            Ray::AtMost => &self.endpoint - &self.var,
        }
    }

    /// The certificate as plain data for serialisation with serde.
    pub fn to_data(&self) -> HalfLineCertificateData {
        HalfLineCertificateData {
            goal: self.goal.to_ex().to_tree(),
            var: self.var.to_tree(),
            endpoint: self.endpoint.to_tree(),
            ray: match self.ray {
                Ray::AtLeast => "at_least".to_string(),
                Ray::AtMost => "at_most".to_string(),
            },
            polya_power: self.polya_power,
            coefficients: self.coefficients.iter().map(serial::q_to_str).collect(),
            square: self.square.as_ref().map(|g| g.to_ex().to_tree()),
        }
    }

    /// Rebuild from data in `ctx` and **re-verify**; data that does not
    /// verify is rejected.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] for malformed data or a failed
    /// verification.
    pub fn from_data(ctx: &Context, data: &HalfLineCertificateData) -> Result<Self, SymplexError> {
        const OP: &str = "HalfLineCertificate::from_data";
        let bad = |reason: String| SymplexError::InvalidArgument {
            operation: OP,
            reason,
        };
        let var = ctx.from_tree(&data.var);
        let goal_ex = ctx.from_tree(&data.goal);
        let goal = Poly::new(&goal_ex, &[&var])
            .ok_or_else(|| bad(format!("goal `{goal_ex}` is not a polynomial in `{var}`")))?;
        let square = match &data.square {
            Some(t) => {
                let e = ctx.from_tree(t);
                Some(
                    Poly::new(&e, &[&var])
                        .ok_or_else(|| bad(format!("square factor `{e}` is not a polynomial")))?,
                )
            }
            None => None,
        };
        let ray = match data.ray.as_str() {
            "at_least" => Ray::AtLeast,
            "at_most" => Ray::AtMost,
            other => {
                return Err(bad(format!(
                    "unknown ray `{other}` (expected `at_least` or `at_most`)"
                )));
            }
        };
        let cert = HalfLineCertificate {
            goal,
            var,
            endpoint: ctx.from_tree(&data.endpoint).eval(),
            ray,
            polya_power: data.polya_power,
            coefficients: data
                .coefficients
                .iter()
                .map(|s| serial::q_from_str(s, OP))
                .collect::<Result<_, _>>()?,
            square,
        };
        if !cert.verify() {
            return Err(bad("the certificate data does not verify".into()));
        }
        Ok(cert)
    }

    /// JSON form of [`to_data`](Self::to_data); [`from_json`](Self::from_json)
    /// re-verifies before accepting.
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::certificates::{HalfLineCertificate, Ray, prove_nonnegative_on_halfline};
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let out = prove_nonnegative_on_halfline(&(&x.powi(2) - &x + 1), &x, &ctx.int(0), Ray::AtLeast, 4).unwrap();
    /// let json = out.certificate().unwrap().to_json().unwrap();
    /// let back = HalfLineCertificate::from_json(&Context::new(), &json).unwrap();
    /// assert_eq!(back.polya_power(), out.certificate().unwrap().polya_power());
    /// assert!(back.verify());
    /// ```
    ///
    /// # Errors
    ///
    /// [`SymplexError::ComputationFailed`] if serialisation fails.
    pub fn to_json(&self) -> Result<String, SymplexError> {
        serde_json::to_string(&self.to_data()).map_err(|e| SymplexError::ComputationFailed {
            operation: "HalfLineCertificate::to_json",
            reason: e.to_string(),
        })
    }

    /// Parse [`to_json`](Self::to_json) output in `ctx` and re-verify it.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] for malformed JSON or data that
    /// does not verify.
    pub fn from_json(ctx: &Context, json: &str) -> Result<Self, SymplexError> {
        let data: HalfLineCertificateData =
            serde_json::from_str(json).map_err(|e| SymplexError::InvalidArgument {
                operation: "HalfLineCertificate::from_json",
                reason: format!("malformed JSON: {e}"),
            })?;
        Self::from_data(ctx, &data)
    }

    /// Recompute `(1 + k)^N · goal` and `square² · Σ cᵢ kⁱ` exactly and
    /// compare them; also checks every `cᵢ ≥ 0`.
    pub fn verify(&self) -> bool {
        if self.coefficients.iter().any(|c| *c < Q::zero()) {
            return false;
        }
        let ctx = self.goal.context();
        let k = self.shift_expr();
        let mut rhs = ctx.zero();
        for (i, c) in self.coefficients.iter().enumerate() {
            rhs += ctx.from_ratio(c.clone()) * k.powi(i as i64);
        }
        if let Some(g) = &self.square {
            rhs = g.to_ex().powi(2) * rhs;
        }
        let lhs = (1 + &k).powi(i64::from(self.polya_power)) * self.goal.to_ex();
        let vars = [&self.var];
        match (Poly::new(&lhs, &vars), Poly::new(&rhs, &vars)) {
            (Some(l), Some(r)) => l.equals(&r),
            _ => false,
        }
    }

    /// The identity as `(lhs, rhs)` expressions:
    /// `((1 + k)^N · goal, square² · Σ cᵢ kⁱ)`.
    pub fn identity(&self) -> (Ex, Ex) {
        let ctx = self.goal.context();
        let k = self.shift_expr();
        let mut rhs = ctx.zero();
        for (i, c) in self.coefficients.iter().enumerate() {
            rhs += ctx.from_ratio(c.clone()) * k.powi(i as i64);
        }
        if let Some(g) = &self.square {
            rhs = g.to_ex().powi(2) * rhs;
        }
        let lhs = if self.polya_power == 0 {
            self.goal.to_ex()
        } else {
            (1 + &k).powi(i64::from(self.polya_power)) * self.goal.to_ex()
        };
        (lhs, rhs)
    }

    /// The hint terms of the Lean proof, for an existing proof skeleton:
    /// one entry per power of `k = x − a` (or `a − x`) with a positive
    /// coefficient, given the caller's name `hk` of the hypothesis
    /// `0 ≤ k` — `hk` itself for the first power, `pow_nonneg hk n` above,
    /// each wrapped in `mul_nonneg (sq_nonneg g) (…)` when the certificate
    /// has a square factor.  The constant term needs no hint.
    ///
    /// With a Pólya power `N > 0` the identity proves
    /// `0 ≤ (1 + k) ^ N * goal`; the caller then divides by
    /// `pow_pos (by linarith) N` as [`to_lean`](Self::to_lean) does.
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::certificates::{prove_nonnegative_on_halfline, Ray};
    /// use symplex::lean::LeanOpts;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// // x² − 2x + 3 = (x − 1)² + 2 on x ≥ 1: coefficients [2, 0, 1] in k = x − 1.
    /// let out = prove_nonnegative_on_halfline(&(&x.powi(2) - &x * 2 + 3), &x, &ctx.int(1), Ray::AtLeast, 0).unwrap();
    /// let cert = out.certificate().unwrap();
    /// assert_eq!(cert.lean_hints("hk", &LeanOpts::default()).unwrap(), vec!["pow_nonneg hk 2"]);
    /// ```
    ///
    /// # Errors
    ///
    /// [`SymplexError::NotImplemented`] if the square factor cannot be
    /// rendered.
    pub fn lean_hints(&self, hk: &str, opts: &LeanOpts) -> Result<Vec<String>, SymplexError> {
        let square_hint = match &self.square {
            Some(g) => Some(format!("sq_nonneg ({})", g.to_ex().to_lean_with(opts)?)),
            None => None,
        };
        let mut hints: Vec<String> = Vec::new();
        for (i, c) in self.coefficients.iter().enumerate() {
            if *c <= Q::zero() {
                continue;
            }
            let h = match i {
                0 => None,
                1 => Some(hk.to_string()),
                _ => Some(format!("pow_nonneg {hk} {i}")),
            };
            let h = match (&square_hint, h) {
                (Some(sq), Some(h)) => format!("mul_nonneg ({sq}) ({h})"),
                (Some(sq), None) => sq.clone(),
                (None, Some(h)) => h,
                (None, None) => continue,
            };
            hints.push(h);
        }
        Ok(hints)
    }

    /// A Lean 4 / Mathlib theorem `0 ≤ goal` for `a ≤ x` (or `x ≤ a`).
    ///
    /// Shape for `N = 0`:
    /// ```text
    /// theorem name (x : ℝ) (h_x_lo : a ≤ x) : 0 ≤ goal := by
    ///   have hk : 0 ≤ x - a := sub_nonneg.mpr h_x_lo
    ///   nlinarith [pow_nonneg hk 2, pow_nonneg hk 3]
    /// ```
    /// and for a Pólya multiplier the product `(1 + (x - a)) ^ N * goal` is
    /// shown non-negative the same way and divided out with
    /// `nonneg_of_mul_nonneg_right`.  A square factor turns every hint into
    /// `mul_nonneg (sq_nonneg g) (…)`.
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
        let v = lean_ident(&self.var.to_string());
        let base = v.trim_matches(['«', '»']);
        let a = self.endpoint.to_lean_with(opts)?;
        let goal = self.goal.to_ex().to_lean_with(opts)?;
        let (hyp_name, hyp, k) = match self.ray {
            Ray::AtLeast => (
                format!("h_{base}_lo"),
                format!("{a} ≤ {v}"),
                format!("{v} - {a}"),
            ),
            Ray::AtMost => (
                format!("h_{base}_hi"),
                format!("{v} ≤ {a}"),
                format!("{a} - {v}"),
            ),
        };
        let square_hint = self.square.is_some();
        // One hint per power of k that carries a positive coefficient.
        let hints = self.lean_hints("hk", opts)?;
        let tactic_name = if square_hint || self.coefficients.len() > 2 {
            "nlinarith"
        } else {
            "linarith"
        };
        let hint_list = if hints.is_empty() {
            String::new()
        } else {
            format!(" [{}]", hints.join(", "))
        };
        let mut out = format!(
            "theorem {} ({v} : {real}) ({hyp_name} : {hyp}) :\n    0 ≤ {goal} := by\n  have hk : 0 ≤ {k} := sub_nonneg.mpr {hyp_name}\n",
            lean_ident(theorem_name),
        );
        if self.polya_power == 0 {
            out.push_str(&format!("  {tactic_name}{hint_list}\n"));
        } else {
            let n = self.polya_power;
            out.push_str(&format!(
                "  have hpos : 0 < (1 + ({k})) ^ {n} := pow_pos (by linarith) {n}\n  have hprod : 0 ≤ (1 + ({k})) ^ {n} * ({goal}) := by nlinarith{hint_list}\n  exact nonneg_of_mul_nonneg_right hprod hpos\n"
            ));
        }
        Ok(wrap_lean(&out, MATHLIB_LINE_WIDTH))
    }
}

impl fmt::Display for HalfLineCertificate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (lhs, rhs) = self.identity();
        write!(f, "{lhs} = {rhs}")?;
        match self.ray {
            Ray::AtLeast => write!(f, ", {} ≥ {}", self.var, self.endpoint),
            Ray::AtMost => write!(f, ", {} ≤ {}", self.var, self.endpoint),
        }
    }
}

/// Result of [`prove_nonnegative_on_halfline`].
#[derive(Clone, Debug)]
pub enum HalfLineOutcome {
    /// A verified certificate.
    Proved(HalfLineCertificate),
    /// The goal is negative at `point` (on the half-line).
    Refuted {
        /// A point of the half-line where the goal is negative.
        point: Q,
        /// The (negative) value there.
        value: Q,
    },
    /// The goal is non-negative on the half-line (decided exactly by Sturm's
    /// theorem) but no certificate was found within the Pólya budget — this
    /// happens when the goal has an interior zero that is not an
    /// even-multiplicity factor over ℚ (e.g. an irreducible SOS such as
    /// `x⁴ − 2x² + 2`… with an irrational double root).
    Unknown {
        /// The largest Pólya exponent tried.
        max_polya_power: u32,
    },
}

impl HalfLineOutcome {
    /// The certificate, if proved.
    pub fn certificate(&self) -> Option<&HalfLineCertificate> {
        match self {
            HalfLineOutcome::Proved(c) => Some(c),
            _ => None,
        }
    }

    /// `true` for [`HalfLineOutcome::Proved`].
    pub fn is_proved(&self) -> bool {
        matches!(self, HalfLineOutcome::Proved(_))
    }
}

/// Coefficients (ascending in `k`) of `p(x)` rewritten in `k` where
/// `x = a + k` (`AtLeast`) or `x = a − k` (`AtMost`).
fn shifted_coefficients(p: &Poly, var: &Ex, a: &Ex, ray: &Ray) -> Option<Vec<Q>> {
    let ctx = p.context();
    let k = ctx.symbol("__k");
    let x_of_k = match ray {
        Ray::AtLeast => a + &k,
        Ray::AtMost => a - &k,
    };
    let q = p.to_ex().subs(var, &x_of_k);
    let qp = Poly::new(&q, &[&k])?;
    let coeffs = qp.all_coeffs()?; // highest first
    let mut asc: Vec<Q> = coeffs
        .iter()
        .rev()
        .map(|c| c.as_rational())
        .collect::<Option<_>>()?;
    while asc.len() > 1 && asc.last().is_some_and(Zero::is_zero) {
        asc.pop();
    }
    Some(asc)
}

/// Multiply the coefficient list by `(1 + k)`.
fn times_one_plus_k(c: &[Q]) -> Vec<Q> {
    let mut out = vec![Q::zero(); c.len() + 1];
    for (i, ci) in c.iter().enumerate() {
        out[i] += ci;
        out[i + 1] += ci;
    }
    out
}

/// Find a point of the half-line where `p < 0`, using the isolating
/// intervals of the real roots and the Cauchy bound.
fn halfline_counterexample(p: &Poly, var: &Ex, a: &Q, ray: &Ray) -> Option<(Q, Q)> {
    let ctx = p.context();
    let e = p.to_ex();
    let eval =
        |x: &Q| -> Option<Q> { e.subs(var, &ctx.from_ratio(x.clone())).eval().as_rational() };
    let one = Q::one();
    let inside = |x: &Q| match ray {
        Ray::AtLeast => x >= a,
        Ray::AtMost => x <= a,
    };
    // Candidate points: the endpoint, midpoints and outer points of the root
    // isolating intervals, and a point beyond all roots.
    let mut candidates: Vec<Q> = vec![a.clone()];
    let iv = e.real_roots_isolate(var);
    for (lo, hi) in &iv {
        if let (Some(l), Some(h)) = (lo.as_rational(), hi.as_rational()) {
            candidates.push((&l + &h) / Q::from_integer(BigInt::from(2)));
            candidates.push(&l - &one);
            candidates.push(&h + &one);
            candidates.push((&l + a) / Q::from_integer(BigInt::from(2)));
        }
    }
    let far = match ray {
        Ray::AtLeast => a + Q::from_integer(BigInt::from(1000)),
        Ray::AtMost => a - Q::from_integer(BigInt::from(1000)),
    };
    candidates.push(far);
    for x in candidates {
        if inside(&x)
            && let Some(v) = eval(&x)
            && v < Q::zero()
        {
            return Some((x, v));
        }
    }
    None
}

/// Prove `goal ≥ 0` for `var ≥ a` ([`Ray::AtLeast`]) or `var ≤ a`
/// ([`Ray::AtMost`]) with a [`HalfLineCertificate`], or refute it exactly.
///
/// The search: shift to `k ≥ 0`; if every coefficient of the shifted
/// polynomial is non-negative that is the certificate (`N = 0`); otherwise
/// multiply by `(1 + k)` up to `max_polya_power` times; if that fails, split
/// off even-multiplicity factors (`goal = g²·h`) and retry on `h`.  A goal
/// that is negative somewhere on the half-line is refuted with an exact
/// point; a goal that is non-negative (by Sturm's theorem) but has no
/// certificate of this form is reported as [`HalfLineOutcome::Unknown`].
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if `a` is not a rational literal, or
/// `goal` is not a univariate polynomial in `var` with rational
/// coefficients.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::certificates::{prove_nonnegative_on_halfline, HalfLineOutcome, Ray};
///
/// let ctx = Context::new();
/// let j = ctx.symbol("j");
/// // (j − 1)(j − 3) ≥ 0 for j ≥ 3:  p(3 + k) = k² + 2k, all coefficients ≥ 0.
/// let p = (&j - 1) * (&j - 3);
/// let out = prove_nonnegative_on_halfline(&p, &j, &ctx.int(3), Ray::AtLeast, 10).unwrap();
/// let cert = out.certificate().unwrap();
/// assert_eq!(cert.polya_power(), 0);
/// assert!(cert.verify());
/// // For j ≥ 2 the claim is false (p(5/2) < 0).
/// assert!(matches!(
///     prove_nonnegative_on_halfline(&p, &j, &ctx.int(2), Ray::AtLeast, 10).unwrap(),
///     HalfLineOutcome::Refuted { .. }
/// ));
/// ```
pub fn prove_nonnegative_on_halfline(
    goal: &Ex,
    var: &Ex,
    a: &Ex,
    ray: Ray,
    max_polya_power: u32,
) -> Result<HalfLineOutcome, SymplexError> {
    let bad = |reason: &str| SymplexError::InvalidArgument {
        operation: "prove_nonnegative_on_halfline",
        reason: reason.into(),
    };
    let a_ex = a.eval();
    let a_q = a_ex
        .as_rational()
        .ok_or_else(|| bad("the endpoint must be a rational literal"))?;
    let goal_poly =
        Poly::new(goal, &[var]).ok_or_else(|| bad("goal must be a polynomial in the variable"))?;
    if !goal_poly.has_rational_coeffs() {
        return Err(bad("goal must have rational coefficients"));
    }
    let ctx: Context = goal.context();

    // Exact decision first, so a false claim is refuted with a point and a
    // true one never gets a spurious Unknown from a failed search.
    let (lo, hi) = match ray {
        Ray::AtLeast => (a_ex.clone(), ctx.infinity()),
        Ray::AtMost => (ctx.neg_infinity(), a_ex.clone()),
    };
    if goal_poly.is_nonnegative_on(&lo, &hi) == Some(false)
        && let Some((point, value)) = halfline_counterexample(&goal_poly, var, &a_q, &ray)
    {
        return Ok(HalfLineOutcome::Refuted { point, value });
    }

    let try_certify = |p: &Poly, square: Option<&Poly>| -> Option<HalfLineCertificate> {
        let mut coeffs = shifted_coefficients(p, var, &a_ex, &ray)?;
        for n in 0..=max_polya_power {
            if coeffs.iter().all(|c| *c >= Q::zero()) {
                let cert = HalfLineCertificate {
                    goal: goal_poly.clone(),
                    var: var.clone(),
                    endpoint: a_ex.clone(),
                    ray: ray.clone(),
                    polya_power: n,
                    coefficients: coeffs,
                    square: square.cloned(),
                };
                return cert.verify().then_some(cert);
            }
            coeffs = times_one_plus_k(&coeffs);
        }
        None
    };

    if let Some(c) = try_certify(&goal_poly, None) {
        return Ok(HalfLineOutcome::Proved(c));
    }
    if let Some((g, h)) = split_square_factor(&goal_poly, &[var])
        && let Some(c) = try_certify(&h, Some(&g))
    {
        return Ok(HalfLineOutcome::Proved(c));
    }
    if let Some((point, value)) = halfline_counterexample(&goal_poly, var, &a_q, &ray) {
        return Ok(HalfLineOutcome::Refuted { point, value });
    }
    Ok(HalfLineOutcome::Unknown { max_polya_power })
}

/// A verified proof that a univariate polynomial is non-negative on all of
/// ℝ: a pair of half-line certificates meeting at `split`.
#[derive(Clone, Debug)]
pub struct RealLineCertificate {
    /// Certificate for `x ≥ split`.
    pub upper: HalfLineCertificate,
    /// Certificate for `x ≤ split`.
    pub lower: HalfLineCertificate,
}

impl RealLineCertificate {
    /// Both halves re-verified.
    pub fn verify(&self) -> bool {
        self.upper.verify() && self.lower.verify() && self.upper.endpoint == self.lower.endpoint
    }

    /// A Lean theorem with no hypotheses, by cases on `le_total split x`.
    pub fn to_lean(&self, theorem_name: &str) -> Result<String, SymplexError> {
        self.to_lean_with(theorem_name, &LeanOpts::default())
    }

    /// [`to_lean`](Self::to_lean) with explicit rendering options.
    pub fn to_lean_with(
        &self,
        theorem_name: &str,
        opts: &LeanOpts,
    ) -> Result<String, SymplexError> {
        let up = self.upper.to_lean_with("_", opts)?;
        let lo = self.lower.to_lean_with("_", opts)?;
        // Take the tactic blocks (everything after the `:= by` line).
        let body = |text: &str| -> String {
            text.split_once(":= by\n")
                .map(|(_, b)| b.to_string())
                .unwrap_or_default()
        };
        let v = lean_ident(&self.upper.var.to_string());
        let base = v.trim_matches(['«', '»']).to_string();
        let a = self.upper.endpoint.to_lean_with(opts)?;
        let goal = self.upper.goal.to_ex().to_lean_with(opts)?;
        let indent = |b: String| -> String {
            b.lines()
                .map(|l| format!("  {l}"))
                .collect::<Vec<_>>()
                .join("\n")
        };
        let text = format!(
            "theorem {} ({v} : {}) : 0 ≤ {goal} := by\n  rcases le_total {a} {v} with h_{base}_lo | h_{base}_hi\n  · -- {a} ≤ {v}\n{}\n  · -- {v} ≤ {a}\n{}\n",
            lean_ident(theorem_name),
            opts.real_type,
            indent(body(&up)),
            indent(body(&lo)),
        );
        Ok(wrap_lean(&text, MATHLIB_LINE_WIDTH))
    }
}

/// Prove `goal ≥ 0` on all of ℝ (univariate) by splitting at `split` into
/// two half-line certificates.
///
/// Returns `Ok(None)` when one side could not be certified (the goal is
/// negative somewhere, or has a zero that is not an even-multiplicity
/// rational factor).
///
/// ```
/// use symplex::prelude::*;
/// use symplex::certificates::prove_nonnegative_on_reals;
///
/// let ctx = Context::new();
/// let x = ctx.symbol("x");
/// let cert = prove_nonnegative_on_reals(&(&x.powi(2) - &x + 1), &x, &ctx.int(0), 10).unwrap().unwrap();
/// assert!(cert.verify());
/// assert!(cert.to_lean("pos_quadratic").unwrap().contains("rcases le_total"));
/// ```
pub fn prove_nonnegative_on_reals(
    goal: &Ex,
    var: &Ex,
    split: &Ex,
    max_polya_power: u32,
) -> Result<Option<RealLineCertificate>, SymplexError> {
    let upper = prove_nonnegative_on_halfline(goal, var, split, Ray::AtLeast, max_polya_power)?;
    let lower = prove_nonnegative_on_halfline(goal, var, split, Ray::AtMost, max_polya_power)?;
    match (upper, lower) {
        (HalfLineOutcome::Proved(upper), HalfLineOutcome::Proved(lower)) => {
            Ok(Some(RealLineCertificate { upper, lower }))
        }
        _ => Ok(None),
    }
}

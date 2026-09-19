//! Certificates of non-negativity (and emptiness) on a polyhedron whose
//! facets depend polynomially on one real parameter `j ≥ j₀`.
//!
//! Given hypotheses `h₁, …, hₘ ∈ ℚ[j, x]` and a goal `g ∈ ℚ[j, x]`, the
//! question is whether `g ≥ 0` wherever every `hₖ ≥ 0`, for every real
//! `j ≥ j₀`.  The certificate is the identity
//!
//! ```text
//! λ(j) · g  =  Σₖ Σ_{a,b} μ_{k,a,b} · jᵃ (j − j₀)ᵇ · hₖ
//!            + Σ_{k≤l} Σ_{a+b≤1} μ_{k,l,a,b} · jᵃ (j − j₀)ᵇ · hₖ hₗ      (optional)
//!            + Σ_{a+b≥1} μ_{a,b} · jᵃ (j − j₀)ᵇ  +  μ₀ ,
//! λ(j) = 1 + Σ_{a≥1} νₐ jᵃ ,        all μ, ν ≥ 0 .
//! ```
//!
//! Every term on the right is non-negative on the set (`j ≥ j₀ ≥ 0` makes
//! the parameter powers non-negative), and `λ(j) > 0`, so the identity
//! proves `g ≥ 0`.  The **polynomial multiplier `λ` on the goal** is what
//! makes the parametric case work: the Farkas multipliers of
//! `j`-dependent facets are rational functions of `j`, and clearing their
//! denominators puts a polynomial in front of `g`.  With the goal `−1` the
//! same identity proves that the polyhedron is **empty** for every
//! `j ≥ j₀`.
//!
//! The search is a sequence of exact LPs ([`linprog`](crate::linprog)),
//! staged from the smallest basis (degree-1 multipliers, `λ = 1`) upwards,
//! because most facets certify cheaply and the small certificates give the
//! shortest Lean hint lists.  Every certificate is re-verified with exact
//! polynomial arithmetic before it is returned.
//!
//! [`PolyhedronCertificate::to_lean`] renders a complete theorem;
//! [`PolyhedronCertificate::lean_steps`] renders only the `have` lines and
//! the closing `linarith only […]` so the proof can be dropped into an
//! existing skeleton with the caller's hypothesis names.

use std::fmt;

use num_bigint::BigInt;
use num_traits::{One, Signed, Zero};

use crate::api::context::Context;
use crate::api::expr::Ex;
use crate::api::poly_ex::Poly;
use crate::base::errors::SymplexError;
use crate::domains::certificates::serial::{q_from_str, q_to_str};
use crate::domains::certificates::{Certificate, Outcome};
use crate::domains::linprog::{LpProblem, LpStatus, Q};
use crate::output::lean::{
    Block, LeanOpts, MATHLIB_LINE_WIDTH, Proof, Tactic, lean_ident, wrap_lean,
};
use crate::output::tree::ExprTree;
use crate::poly::multipoly::{GrevLex, MultiPoly};

const OP: &str = "prove_nonnegative_on_polyhedron";

fn invalid(reason: impl Into<String>) -> SymplexError {
    SymplexError::InvalidArgument {
        operation: OP,
        reason: reason.into(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Options and outcome
// ═══════════════════════════════════════════════════════════════════════════

/// Search limits for [`prove_nonnegative_on_polyhedron`].
///
/// A *stage* is one exact LP with multipliers `jᵃ (j − j₀)ᵇ` of total
/// degree `≤ degree` on the hypotheses, a goal multiplier `λ(j)` of degree
/// `≤ lambda_degree`, and optionally pairwise products of hypotheses.  With
/// `staged = true` the stages run from the smallest basis upwards and the
/// first success is returned; with `staged = false` a single LP at the
/// maxima is solved.
///
/// `#[non_exhaustive]`: build it with [`Default`] and the `with_*`
/// builders (or assign fields on a `mut` default), so that a future option
/// is not a breaking change.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct PolyhedronOpts {
    /// Largest total degree `a + b` of the parameter multipliers on
    /// hypothesis terms (pure parameter terms go one degree higher).
    /// Ignored without a parameter.
    pub max_degree: u32,
    /// Largest degree of `λ(j)`; `0` forces `λ = 1`.  Ignored without a
    /// parameter.
    pub max_lambda_degree: u32,
    /// Also try products `hₖ hₗ` of two hypotheses (with multipliers of
    /// degree `≤ 1`) as the final stage.
    pub pairwise: bool,
    /// Try small bases first and escalate on failure.
    pub staged: bool,
}

impl Default for PolyhedronOpts {
    /// `max_degree = 3`, `max_lambda_degree = 3`, `pairwise = true`,
    /// `staged = true`.
    fn default() -> Self {
        PolyhedronOpts {
            max_degree: 3,
            max_lambda_degree: 3,
            pairwise: true,
            staged: true,
        }
    }
}

impl PolyhedronOpts {
    /// One stage only: degree-`degree` multipliers, `λ` of degree
    /// `lambda_degree`, no pairwise products, not staged.
    pub fn single(degree: u32, lambda_degree: u32) -> Self {
        PolyhedronOpts {
            max_degree: degree,
            max_lambda_degree: lambda_degree,
            pairwise: false,
            staged: false,
        }
    }

    /// Set [`max_degree`](Self::max_degree).
    #[must_use]
    pub fn with_max_degree(mut self, max_degree: u32) -> Self {
        self.max_degree = max_degree;
        self
    }

    /// Set [`max_lambda_degree`](Self::max_lambda_degree).
    #[must_use]
    pub fn with_max_lambda_degree(mut self, max_lambda_degree: u32) -> Self {
        self.max_lambda_degree = max_lambda_degree;
        self
    }

    /// Set [`pairwise`](Self::pairwise).
    #[must_use]
    pub fn with_pairwise(mut self, pairwise: bool) -> Self {
        self.pairwise = pairwise;
        self
    }

    /// Set [`staged`](Self::staged).
    #[must_use]
    pub fn with_staged(mut self, staged: bool) -> Self {
        self.staged = staged;
        self
    }

    /// The `(degree, lambda_degree, pairwise)` stages this configuration
    /// runs, in order.
    fn stages(&self) -> Vec<(u32, u32, bool)> {
        if !self.staged {
            return vec![(self.max_degree, self.max_lambda_degree, self.pairwise)];
        }
        let mut stages: Vec<(u32, u32, bool)> = Vec::new();
        for d in 1..=self.max_degree.max(1) {
            for l in [d.saturating_sub(1), d] {
                let l = l.min(self.max_lambda_degree);
                if !stages.contains(&(d, l, false)) {
                    stages.push((d, l, false));
                }
            }
        }
        if self.pairwise {
            let d = self.max_degree.clamp(1, 2);
            stages.push((d, d.min(self.max_lambda_degree), true));
        }
        stages
    }
}

/// One weighted product of a [`PolyhedronCertificate`]:
/// `weight · jᵃ (j − j₀)ᵇ · Π hₖ` over the listed hypotheses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolyhedronTerm {
    /// Indices (into the certificate's hypotheses) of the hypothesis
    /// factors: empty for a pure parameter term or the constant, one entry
    /// for `hₖ`, two for a pairwise product `hₖ hₗ`.
    pub hyps: Vec<usize>,
    /// Exponent `a` of the parameter `j` itself.
    pub var_power: u32,
    /// Exponent `b` of the shifted parameter `j − j₀`.
    pub shift_power: u32,
    /// The weight `μ > 0`.
    pub weight: Q,
}

impl PolyhedronTerm {
    /// Total degree of the parameter multiplier, `a + b`.
    pub fn multiplier_degree(&self) -> u32 {
        self.var_power + self.shift_power
    }
}

/// Result of [`prove_nonnegative_on_polyhedron`]: an [`Outcome`] with a
/// [`PolyhedronCertificate`] or a [`PolyhedronUnknown`].  A refutation's
/// `point` lists the free variables then the parameter, and `param_value`
/// is the sampled parameter value; for the emptiness question the point
/// lies *in* the polyhedron.
pub type PolyhedronOutcome = Outcome<PolyhedronCertificate, PolyhedronUnknown>;

/// Why [`prove_nonnegative_on_polyhedron`] could not decide: no
/// certificate up to these limits, and no counterexample on the sampled
/// parameter values.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct PolyhedronUnknown {
    /// The largest multiplier degree that was tried.
    pub degree: u32,
    /// The largest `λ` degree that was tried.
    pub lambda_degree: u32,
    /// Whether pairwise products were tried.
    pub pairwise: bool,
}

impl fmt::Display for PolyhedronUnknown {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "no certificate up to multiplier degree {}, λ degree {}{}",
            self.degree,
            self.lambda_degree,
            if self.pairwise {
                ", with pairwise products"
            } else {
                ""
            }
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Certificate
// ═══════════════════════════════════════════════════════════════════════════

/// The parameter of a certificate: `var ≥ lo`.
#[derive(Clone, Debug)]
struct Param {
    var: Ex,
    lo: Ex,
    /// `true` when `lo ≥ 0`, so that `var` itself is a non-negative atom
    /// and `varᵃ` may appear in multipliers and in `λ`.
    var_nonneg: bool,
    /// `true` when `lo = 0`: the shifted atom `var − lo` coincides with
    /// `var`, so only powers of `var` are used.
    lo_zero: bool,
}

impl Param {
    /// Is the shifted atom `var − lo` a distinct generator of the cone?
    fn uses_shift(&self) -> bool {
        !self.lo_zero
    }
}

/// A verified certificate `λ(j)·goal = Σ weight · jᵃ (j − j₀)ᵇ · Π hₖ`.
#[derive(Clone, Debug)]
pub struct PolyhedronCertificate {
    goal: Poly,
    hyps: Vec<Poly>,
    param: Option<Param>,
    /// Coefficients of `λ` in ascending powers of the parameter atom
    /// (`j` when `j₀ ≥ 0`, else `j − j₀`); `lambda[0] = 1`.
    lambda: Vec<Q>,
    terms: Vec<PolyhedronTerm>,
}

/// Hypothesis names for [`PolyhedronCertificate::lean_steps`].
#[derive(Clone, Debug)]
pub struct PolyhedronLeanNames<'a> {
    /// One Lean hypothesis name per certificate hypothesis, each proving
    /// `0 ≤ hₖ`.
    pub hyps: &'a [&'a str],
    /// Name of the hypothesis `0 ≤ j` (the parameter itself); only used
    /// when `j₀ ≥ 0`.
    pub param_nonneg: &'a str,
    /// Name of the hypothesis `0 ≤ j − j₀`.
    pub shift_nonneg: &'a str,
}

/// The tactic lines of a polyhedron proof, without indentation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolyhedronLeanSteps {
    /// `have name := mul_nonneg …` lines building every product that the
    /// certificate uses, in dependency order (each product is introduced
    /// once).
    pub haves: Vec<String>,
    /// The names to pass to `linarith only […]` for the identity
    /// `λ·goal = Σ …`: hypotheses, products and parameter powers.
    pub hints: Vec<String>,
    /// The names proving `0 < λ(j)` (`0 ≤ j` and its powers); empty when
    /// `λ = 1`.
    pub lambda_hints: Vec<String>,
    /// `λ(j)` rendered in Lean, or `None` when `λ = 1`.
    pub lambda: Option<String>,
    /// The closing lines: `linarith only […]` when `λ = 1`, otherwise the
    /// `have hg … / have hg' := nonneg_of_mul_nonneg_right … / linarith
    /// only [hg']` block.  Lines that continue a tactic are indented by two
    /// spaces relative to the block.
    pub closing: Vec<String>,
}

impl PolyhedronLeanSteps {
    /// All lines (`haves` then `closing`) with `indent` prepended and
    /// re-flowed to Mathlib's line width ([`MATHLIB_LINE_WIDTH`], indent
    /// included; `:= by` stays on its `have` line), as one string ending in
    /// a newline.  This is [`block`](Self::block) rendered at `indent`.
    pub fn to_block(&self, indent: &str) -> String {
        self.to_block_width(indent, MATHLIB_LINE_WIDTH)
    }

    /// [`to_block`](Self::to_block) with an explicit line width.
    pub fn to_block_width(&self, indent: &str, width: usize) -> String {
        self.block().render_width(indent, width)
    }

    /// The proof as a structured [`Block`]: one raw `have` tactic per
    /// product, then the closing tactics — `linarith only […]`, or the
    /// `have hg : … := by` block (whose `linarith` sits inside the `by`)
    /// followed by `have hg' := …` and `linarith only [hg']`.  Splice it
    /// into a larger [`Block`] (a bullet of a `refine … ?_` call, say) and
    /// let the renderer place every line.
    pub fn block(&self) -> Block {
        let mut tactics: Vec<Tactic> = self.haves.iter().map(Tactic::raw).collect();
        if self.lambda.is_some() && !self.proves_emptiness_shape() {
            // `have hg : … := by` / `  linarith only […]` / `have hg' := …` / `linarith only [hg']`.
            let mut lines = self.closing.iter();
            if let (Some(hg), Some(inner), Some(hg2), Some(last)) =
                (lines.next(), lines.next(), lines.next(), lines.next())
            {
                let ty = hg
                    .strip_prefix("have hg : ")
                    .and_then(|s| s.strip_suffix(" := by"))
                    .unwrap_or(hg);
                tactics.push(Tactic::have(
                    "hg",
                    Some(ty),
                    Proof::by(Block::new(vec![Tactic::raw(inner.trim_start())])),
                ));
                tactics.push(Tactic::raw(hg2));
                tactics.push(Tactic::raw(last));
                for extra in lines {
                    tactics.push(Tactic::raw(extra));
                }
                return Block::new(tactics);
            }
        }
        for l in &self.closing {
            tactics.push(Tactic::raw(l));
        }
        Block::new(tactics)
    }

    /// The closing block of an emptiness certificate is a single
    /// `linarith only […]` even when `λ ≠ 1`.
    fn proves_emptiness_shape(&self) -> bool {
        self.closing.len() == 1
    }
}

/// Serialisable form of a [`PolyhedronCertificate`] (see
/// [`PolyhedronCertificate::to_json`]).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PolyhedronCertificateData {
    /// The free variables then the parameter, as expression trees.
    pub gens: Vec<ExprTree>,
    /// The goal.
    pub goal: ExprTree,
    /// The hypotheses `hₖ ≥ 0`.
    pub hyps: Vec<ExprTree>,
    /// `(parameter, lower bound)` if any.
    pub param: Option<(ExprTree, ExprTree)>,
    /// `λ` coefficients as exact rationals `"p/q"`, ascending.
    pub lambda: Vec<String>,
    /// The terms as `(hypothesis indices, j power, shift power, weight "p/q")`.
    pub terms: Vec<(Vec<usize>, u32, u32, String)>,
}

impl PolyhedronCertificate {
    /// The certificate as plain data (expression trees and rationals as
    /// strings), for serialisation with serde.
    pub fn to_data(&self) -> PolyhedronCertificateData {
        PolyhedronCertificateData {
            gens: self.goal.gens().iter().map(Ex::to_tree).collect(),
            goal: self.goal.to_ex().to_tree(),
            hyps: self.hyps.iter().map(|h| h.to_ex().to_tree()).collect(),
            param: self
                .param
                .as_ref()
                .map(|p| (p.var.to_tree(), p.lo.to_tree())),
            lambda: self.lambda.iter().map(q_to_str).collect(),
            terms: self
                .terms
                .iter()
                .map(|t| {
                    (
                        t.hyps.clone(),
                        t.var_power,
                        t.shift_power,
                        q_to_str(&t.weight),
                    )
                })
                .collect(),
        }
    }

    /// Rebuild a certificate from its data in `ctx` and **re-verify** it;
    /// data that does not verify is rejected.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if a rational is malformed, a
    /// hypothesis index is out of range, an expression is not a polynomial
    /// in the generators, or the identity does not hold.
    pub fn from_data(
        ctx: &Context,
        data: &PolyhedronCertificateData,
    ) -> Result<Self, SymplexError> {
        let gens: Vec<Ex> = data.gens.iter().map(|t| ctx.from_tree(t)).collect();
        let gen_refs: Vec<&Ex> = gens.iter().collect();
        let poly = |t: &ExprTree, what: &str| -> Result<Poly, SymplexError> {
            let e = ctx.from_tree(t);
            Poly::new(&e, &gen_refs).ok_or_else(|| {
                invalid(format!(
                    "{what} `{e}` is not a polynomial in the generators"
                ))
            })
        };
        let goal = poly(&data.goal, "goal")?;
        let hyps: Vec<Poly> = data
            .hyps
            .iter()
            .map(|h| poly(h, "hypothesis"))
            .collect::<Result<_, _>>()?;
        let param = match &data.param {
            Some((v, lo)) => {
                let lo = ctx.from_tree(lo).eval();
                let lo_q = lo
                    .as_rational()
                    .ok_or_else(|| invalid("the parameter bound must be a rational literal"))?;
                Some(Param {
                    var: ctx.from_tree(v),
                    lo,
                    var_nonneg: !lo_q.is_negative(),
                    lo_zero: lo_q.is_zero(),
                })
            }
            None => None,
        };
        let lambda: Vec<Q> = data
            .lambda
            .iter()
            .map(|s| q_from_str(s, OP))
            .collect::<Result<_, _>>()?;
        let mut terms = Vec::with_capacity(data.terms.len());
        for (hs, a, b, w) in &data.terms {
            if hs.iter().any(|&k| k >= hyps.len()) {
                return Err(invalid(format!("hypothesis index out of range in {hs:?}")));
            }
            terms.push(PolyhedronTerm {
                hyps: hs.clone(),
                var_power: *a,
                shift_power: *b,
                weight: q_from_str(w, OP)?,
            });
        }
        let cert = PolyhedronCertificate {
            goal,
            hyps,
            param,
            lambda,
            terms,
        };
        if !cert.verify() {
            return Err(invalid("the certificate data does not verify"));
        }
        Ok(cert)
    }

    /// JSON form of [`to_data`](Self::to_data).
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::certificates::{PolyhedronCertificate, PolyhedronOpts, prove_nonnegative_on_polyhedron};
    ///
    /// let ctx = Context::new();
    /// let (j, r, t) = (ctx.symbol("j"), ctx.symbol("r"), ctx.symbol("t"));
    /// let hyps = [&t - &r, &t + &j * &r - &j - 1];
    /// let out = prove_nonnegative_on_polyhedron(&(&t - 1), &hyps, Some((&j, &ctx.int(0))), &PolyhedronOpts::default()).unwrap();
    /// let json = out.certificate().unwrap().to_json().unwrap();
    /// // … across a process boundary …
    /// let other = Context::new();
    /// let back = PolyhedronCertificate::from_json(&other, &json).unwrap();   // re-verified
    /// assert_eq!(back.to_string(), out.certificate().unwrap().to_string());
    /// assert!(PolyhedronCertificate::from_json(&other, &json.replace("\"1/1\"", "\"2/1\"")).is_err());
    /// ```
    ///
    /// # Errors
    ///
    /// [`SymplexError::ComputationFailed`] if serialisation fails.
    pub fn to_json(&self) -> Result<String, SymplexError> {
        serde_json::to_string(&self.to_data()).map_err(|e| SymplexError::ComputationFailed {
            operation: OP,
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
        let data: PolyhedronCertificateData =
            serde_json::from_str(json).map_err(|e| invalid(format!("malformed JSON: {e}")))?;
        Self::from_data(ctx, &data)
    }

    /// The goal `g`.
    pub fn goal(&self) -> &Poly {
        &self.goal
    }

    /// The hypotheses `hₖ ≥ 0`, in input order.
    pub fn hyps(&self) -> &[Poly] {
        &self.hyps
    }

    /// The parameter `(var, lo)` meaning `var ≥ lo`, if any.
    pub fn parameter(&self) -> Option<(&Ex, &Ex)> {
        self.param.as_ref().map(|p| (&p.var, &p.lo))
    }

    /// The weighted products, in the order the LP enumerated them.
    pub fn terms(&self) -> &[PolyhedronTerm] {
        &self.terms
    }

    /// Coefficients of `λ` in ascending powers of the parameter atom
    /// (`j`, or `j − j₀` when `j₀ < 0`); `[1]` when `λ = 1`.
    pub fn lambda_coeffs(&self) -> &[Q] {
        &self.lambda
    }

    /// `true` when the goal multiplier is the constant `1`.
    pub fn lambda_is_one(&self) -> bool {
        self.lambda.len() == 1
    }

    /// The parameter atom raised to `n`: `jⁿ`, or `(j − j₀)ⁿ` when `j₀ < 0`.
    fn lambda_atom(&self) -> Option<Ex> {
        let p = self.param.as_ref()?;
        Some(if p.var_nonneg {
            p.var.clone()
        } else {
            &p.var - &p.lo
        })
    }

    /// Is the goal a negative constant?  Then the certificate proves that
    /// the set is empty, and the Lean statement concludes `False`.
    pub fn proves_emptiness(&self) -> bool {
        self.goal
            .is_ground()
            .then(|| self.goal.to_ex().as_rational())
            .flatten()
            .is_some_and(|c| c.is_negative())
    }

    /// `λ(j)` as an expression (`1` when there is no parameter).
    pub fn lambda(&self) -> Ex {
        let ctx = self.goal.context();
        let mut acc = ctx.one();
        if let Some(atom) = self.lambda_atom() {
            for (a, c) in self.lambda.iter().enumerate().skip(1) {
                if !c.is_zero() {
                    acc += ctx.from_ratio(c.clone()) * atom.powi(a as i64);
                }
            }
        }
        acc
    }

    /// Largest multiplier degree among the terms.
    pub fn degree(&self) -> u32 {
        self.terms
            .iter()
            .map(PolyhedronTerm::multiplier_degree)
            .max()
            .unwrap_or(0)
    }

    /// `true` if some term is a product of two hypotheses.
    pub fn uses_pairwise(&self) -> bool {
        self.terms.iter().any(|t| t.hyps.len() == 2)
    }

    /// Indices (into [`hyps`](Self::hyps)) of the hypotheses that occur in
    /// some term, ascending and without repetition — the hypotheses a Lean
    /// proof of this certificate actually needs, so that a generated lemma
    /// can list exactly those in its signature.  Hypotheses of the
    /// polyhedron that the identity does not use are absent.
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::certificates::{prove_nonnegative_on_polyhedron, PolyhedronOpts, PolyhedronOutcome};
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// // x ≥ 0, y ≥ 0, 1 − x ≥ 0, 1 − y ≥ 0 ⊢ x + 2 ≥ 0 uses only the first.
    /// let hyps = [x.clone(), y.clone(), 1 - &x, 1 - &y];
    /// let PolyhedronOutcome::Proved(c) =
    ///     prove_nonnegative_on_polyhedron(&(&x + 2), &hyps, None, &PolyhedronOpts::default())?
    /// else { panic!() };
    /// assert_eq!(c.used_hyps(), vec![0]);
    /// # Ok::<(), SymplexError>(())
    /// ```
    pub fn used_hyps(&self) -> Vec<usize> {
        let mut used = vec![false; self.hyps.len()];
        for t in &self.terms {
            for &k in &t.hyps {
                if let Some(u) = used.get_mut(k) {
                    *u = true;
                }
            }
        }
        used.iter()
            .enumerate()
            .filter_map(|(k, &u)| u.then_some(k))
            .collect()
    }

    /// The product `jᵃ (j − j₀)ᵇ · Π hₖ` of one term, in factored form.
    pub fn product_expr(&self, term: &PolyhedronTerm) -> Ex {
        let ctx = self.goal.context();
        let mut acc = ctx.one();
        if let Some(p) = &self.param {
            if term.var_power > 0 {
                acc *= p.var.powi(i64::from(term.var_power));
            }
            if term.shift_power > 0 {
                acc *= (&p.var - &p.lo).powi(i64::from(term.shift_power));
            }
        }
        for &k in &term.hyps {
            if let Some(h) = self.hyps.get(k) {
                acc *= h.to_ex();
            }
        }
        acc
    }

    /// The product of one term as a `Poly` (expanded).
    fn product_poly(&self, term: &PolyhedronTerm) -> Result<Poly, SymplexError> {
        let gens: Vec<&Ex> = self.goal.gens().iter().collect();
        Poly::new(&self.product_expr(term), &gens).ok_or_else(|| invalid("internal: product"))
    }

    /// The identity `λ·goal = Σ weight·product` as `(lhs, rhs)`, with the
    /// products kept in factored form.
    pub fn identity(&self) -> (Ex, Ex) {
        let ctx = self.goal.context();
        let mut rhs = ctx.zero();
        for t in &self.terms {
            rhs += ctx.from_ratio(t.weight.clone()) * self.product_expr(t);
        }
        (self.lambda() * self.goal.to_ex(), rhs)
    }

    /// Recompute `Σ weight·product − λ·goal` with exact polynomial
    /// arithmetic and check that it is identically zero, that every weight
    /// is positive and every `λ` coefficient non-negative with `λ(0) = 1`.
    pub fn verify(&self) -> bool {
        if self.lambda.first().is_none_or(|c| !c.is_one())
            || self.lambda.iter().any(Signed::is_negative)
            || self.terms.iter().any(|t| !t.weight.is_positive())
        {
            return false;
        }
        let gens: Vec<&Ex> = self.goal.gens().iter().collect();
        let ctx = self.goal.context();
        let Ok(mut acc) = Poly::zero(&ctx, &gens) else {
            return false;
        };
        for t in &self.terms {
            let Ok(p) = self.product_poly(t) else {
                return false;
            };
            let Ok(scaled) = p.scale(&ctx.from_ratio(t.weight.clone())) else {
                return false;
            };
            let Ok(sum) = acc.add(&scaled) else {
                return false;
            };
            acc = sum;
        }
        let Some(lambda) = Poly::new(&self.lambda(), &gens) else {
            return false;
        };
        let Ok(lg) = lambda.mul(&self.goal) else {
            return false;
        };
        acc.equals(&lg)
    }

    // ── Lean ──────────────────────────────────────────────────────────────

    /// The `have` lines, hint names and closing tactic of a Lean proof of
    /// `0 ≤ goal` from the named hypotheses, for embedding in an existing
    /// proof.  `names.hyps` must have one entry per hypothesis.
    ///
    /// Products are named by appending `J` per power of `j` and `K` per
    /// power of `j − j₀` to the hypothesis name (`h0K`, `h0JK`), pairwise
    /// products as `h0xh1`, and pure parameter powers as `pJK`, `pJJ`, …
    /// (`pJ`/`pK` are the parameter hypotheses themselves).
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if the number of names does not
    /// match; [`SymplexError::NotImplemented`] if an expression cannot be
    /// rendered.
    pub fn lean_steps(
        &self,
        names: &PolyhedronLeanNames<'_>,
        opts: &LeanOpts,
    ) -> Result<PolyhedronLeanSteps, SymplexError> {
        if names.hyps.len() != self.hyps.len() {
            return Err(invalid(format!(
                "{} hypothesis names for {} hypotheses",
                names.hyps.len(),
                self.hyps.len()
            )));
        }
        let hj = names.param_nonneg;
        let hk = names.shift_nonneg;
        let mut haves: Vec<String> = Vec::new();
        let mut done: Vec<String> = Vec::new();
        let mut hints: Vec<String> = Vec::new();

        // Name of the parameter power `jᵃ (j − j₀)ᵇ`, introducing the
        // `have` chain that builds it (K factors first, then J factors).
        let chain = |base: &str,
                     a: u32,
                     b: u32,
                     haves: &mut Vec<String>,
                     done: &mut Vec<String>|
         -> String {
            let mut prev = base.to_string();
            for bb in 1..=b {
                let name = format!("{base}{}", "K".repeat(bb as usize));
                push_have(haves, done, &name, hk, &prev);
                prev = name;
            }
            for aa in 1..=a {
                let name = format!(
                    "{base}{}{}",
                    "J".repeat(aa as usize),
                    "K".repeat(b as usize)
                );
                push_have(haves, done, &name, hj, &prev);
                prev = name;
            }
            prev
        };
        // Pure parameter powers `pJ…K…`: `pJ` is `hj` and `pK` is `hk`
        // themselves; higher powers are chained from them (K factors first).
        let pure = |a: u32,
                    b: u32,
                    haves: &mut Vec<String>,
                    done: &mut Vec<String>|
         -> Option<String> {
            match (a, b) {
                (0, 0) => None,
                (1, 0) => Some(hj.to_string()),
                (0, 1) => Some(hk.to_string()),
                _ => {
                    let mut prev;
                    if b >= 1 {
                        prev = hk.to_string();
                        for bb in 2..=b {
                            let name = format!("p{}", "K".repeat(bb as usize));
                            push_have(haves, done, &name, hk, &prev);
                            prev = name;
                        }
                        for aa in 1..=a {
                            let name =
                                format!("p{}{}", "J".repeat(aa as usize), "K".repeat(b as usize));
                            push_have(haves, done, &name, hj, &prev);
                            prev = name;
                        }
                    } else {
                        prev = hj.to_string();
                        for aa in 2..=a {
                            let name = format!("p{}", "J".repeat(aa as usize));
                            push_have(haves, done, &name, hj, &prev);
                            prev = name;
                        }
                    }
                    Some(prev)
                }
            }
        };

        for t in &self.terms {
            let name = match t.hyps.as_slice() {
                [] => pure(t.var_power, t.shift_power, &mut haves, &mut done),
                [k] => Some(chain(
                    names.hyps[*k],
                    t.var_power,
                    t.shift_power,
                    &mut haves,
                    &mut done,
                )),
                [k, l] => {
                    let base = format!("{}x{}", names.hyps[*k], names.hyps[*l]);
                    if !done.contains(&base) {
                        done.push(base.clone());
                        haves.push(format!(
                            "have {base} := mul_nonneg {} {}",
                            names.hyps[*k], names.hyps[*l]
                        ));
                    }
                    Some(chain(
                        &base,
                        t.var_power,
                        t.shift_power,
                        &mut haves,
                        &mut done,
                    ))
                }
                _ => return Err(invalid("internal: term with more than two hypotheses")),
            };
            if let Some(n) = name
                && !hints.contains(&n)
            {
                hints.push(n);
            }
        }

        // λ > 0 from 0 ≤ j and its powers (or the shifted atom's powers).
        let mut lambda_hints: Vec<String> = Vec::new();
        let lambda = if self.lambda_is_one() {
            None
        } else {
            let atom_hyp = match &self.param {
                Some(p) if p.var_nonneg => hj,
                _ => hk,
            };
            lambda_hints.push(atom_hyp.to_string());
            for (a, c) in self.lambda.iter().enumerate().skip(2) {
                if c.is_zero() {
                    continue;
                }
                let n = match &self.param {
                    Some(p) if p.var_nonneg => pure(a as u32, 0, &mut haves, &mut done),
                    _ => pure(0, a as u32, &mut haves, &mut done),
                };
                if let Some(n) = n
                    && !lambda_hints.contains(&n)
                {
                    lambda_hints.push(n);
                }
            }
            Some(self.lambda().to_lean_with(opts)?)
        };

        let goal = self.goal.to_ex().to_lean_with(opts)?;
        let hint_list = if hints.is_empty() {
            String::new()
        } else {
            format!(" [{}]", hints.join(", "))
        };
        let closing = match &lambda {
            None => vec![format!("linarith only{hint_list}")],
            // Emptiness: λ·(−c) = Σ ≥ 0 with λ ≥ 1 is already a linear
            // contradiction once λ's monomials are supplied as atoms.
            Some(_) if self.proves_emptiness() => {
                let mut all = hints.clone();
                for h in &lambda_hints {
                    if !all.contains(h) {
                        all.push(h.clone());
                    }
                }
                vec![format!("linarith only [{}]", all.join(", "))]
            }
            Some(l) => vec![
                format!(
                    "have hg : (0 : {}) ≤ ({l}) * ({goal}) := by",
                    opts.real_type
                ),
                format!("  linarith only{hint_list}"),
                format!(
                    "have hg' := nonneg_of_mul_nonneg_right hg (by linarith only [{}])",
                    lambda_hints.join(", ")
                ),
                "linarith only [hg']".to_string(),
            ],
        };
        Ok(PolyhedronLeanSteps {
            haves,
            hints,
            lambda_hints,
            lambda,
            closing,
        })
    }

    /// A Lean 4 / Mathlib theorem proving `0 ≤ goal` from `j₀ ≤ j` and
    /// `0 ≤ hₖ` for every hypothesis.
    ///
    /// The statement quantifies over the free variables and the parameter
    /// as reals, names the hypotheses `h0, h1, …` (`hj` for the parameter
    /// bound; unused ones get a leading underscore), derives `hJ0 : 0 ≤ j`
    /// and `hK0 : 0 ≤ j − j₀` when they are needed, builds every product
    /// with `mul_nonneg`, and closes with `linarith only […]` — after
    /// `nonneg_of_mul_nonneg_right` when `λ ≠ 1`.  A negative constant goal
    /// (the emptiness question) concludes `False`.  Wrapped to Mathlib's
    /// line width; every shape is compile-checked against Mathlib.
    ///
    /// # Errors
    ///
    /// [`SymplexError::NotImplemented`] if an expression cannot be rendered.
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
            .goal
            .gens()
            .iter()
            .map(|g| lean_ident(&g.to_string()))
            .collect();
        // Hypotheses the certificate does not use get a leading underscore
        // so Mathlib's unused-variable linter stays quiet.
        let used = self.used_hyps();
        let hyp_names: Vec<String> = (0..self.hyps.len())
            .map(|k| format!("{}h{k}", if used.contains(&k) { "" } else { "_" }))
            .collect();
        let hyp_refs: Vec<&str> = hyp_names.iter().map(String::as_str).collect();
        let steps = self.lean_steps(
            &PolyhedronLeanNames {
                hyps: &hyp_refs,
                param_nonneg: "hJ0",
                shift_nonneg: "hK0",
            },
            opts,
        )?;
        let mentions = |name: &str| {
            steps.haves.iter().chain(&steps.closing).any(|l| {
                l.split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '\''))
                    .any(|w| w == name)
            })
        };
        let mut binders: Vec<String> = Vec::new();
        let mut prelude: Vec<String> = Vec::new();
        if let Some(p) = &self.param {
            // The binder list declares the plain identifier; every mention
            // uses the caller's rendering (`symbol_text`), e.g. `(j : ℝ)`.
            let v = opts.symbol(&p.var.to_string());
            let lo = p.lo.to_lean_with(opts)?;
            let hj_used = p.var_nonneg && mentions("hJ0");
            let hk_used = p.uses_shift() && mentions("hK0");
            binders.push(format!(
                "({}hj : {lo} ≤ {v})",
                if hj_used || hk_used { "" } else { "_" }
            ));
            if hj_used {
                prelude.push(format!("have hJ0 : (0 : {real}) ≤ {v} := by linarith"));
            }
            if hk_used {
                // `by linarith` rather than `sub_nonneg.mpr hj`: the shifted
                // atom renders canonically (`j + 1` for `j₀ = −1`), which
                // `sub_nonneg` would not match syntactically.
                let k = (&p.var - &p.lo).to_lean_with(opts)?;
                prelude.push(format!("have hK0 : (0 : {real}) ≤ {k} := by linarith"));
            }
        }
        for (name, h) in hyp_names.iter().zip(&self.hyps) {
            binders.push(format!("({name} : 0 ≤ {})", h.to_ex().to_lean_with(opts)?));
        }
        let conclusion = if self.proves_emptiness() {
            "False".to_string()
        } else {
            format!("0 ≤ {}", self.goal.to_ex().to_lean_with(opts)?)
        };
        let mut text = format!(
            "theorem {} ({} : {real}) {} :\n    {conclusion} := by\n",
            lean_ident(theorem_name),
            vars.join(" "),
            binders.join(" ")
        );
        for l in &prelude {
            text.push_str("  ");
            text.push_str(l);
            text.push('\n');
        }
        text.push_str(&steps.to_block("  "));
        Ok(wrap_lean(&text, MATHLIB_LINE_WIDTH))
    }
}

fn push_have(
    haves: &mut Vec<String>,
    done: &mut Vec<String>,
    name: &str,
    factor: &str,
    prev: &str,
) {
    if !done.iter().any(|d| d == name) {
        done.push(name.to_string());
        haves.push(format!("have {name} := mul_nonneg {factor} {prev}"));
    }
}

impl Certificate for PolyhedronCertificate {
    fn goal(&self) -> &Poly {
        PolyhedronCertificate::goal(self)
    }
    fn verify(&self) -> bool {
        PolyhedronCertificate::verify(self)
    }
    fn to_lean_with(&self, theorem_name: &str, opts: &LeanOpts) -> Result<String, SymplexError> {
        PolyhedronCertificate::to_lean_with(self, theorem_name, opts)
    }
    fn to_json(&self) -> Result<String, SymplexError> {
        PolyhedronCertificate::to_json(self)
    }
    fn from_json(ctx: &Context, json: &str) -> Result<Self, SymplexError> {
        PolyhedronCertificate::from_json(ctx, json)
    }
}

impl fmt::Display for PolyhedronCertificate {
    /// `λ*(goal) = w₁*j*h0 + w₂*(j - j₀)*h1 + …; h0 = …, h1 = …; j ≥ j₀` — the
    /// hypotheses are abbreviated `hₖ` so the identity stays readable.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let lambda = self.lambda();
        if self.lambda_is_one() {
            write!(f, "{} = ", self.goal.to_ex())?;
        } else {
            write!(f, "({lambda})*({}) = ", self.goal.to_ex())?;
        }
        let shift = self.param.as_ref().map(|p| format!("({})", &p.var - &p.lo));
        let mut first = true;
        for t in &self.terms {
            if !first {
                write!(f, " + ")?;
            }
            first = false;
            let mut factors: Vec<String> = Vec::new();
            if let Some(p) = &self.param {
                match t.var_power {
                    0 => {}
                    1 => factors.push(p.var.to_string()),
                    a => factors.push(format!("{}^{a}", p.var)),
                }
                if let Some(s) = &shift {
                    match t.shift_power {
                        0 => {}
                        1 => factors.push(s.clone()),
                        b => factors.push(format!("{s}^{b}")),
                    }
                }
            }
            for &k in &t.hyps {
                factors.push(format!("h{k}"));
            }
            if factors.is_empty() {
                write!(f, "{}", t.weight)?;
            } else if t.weight.is_one() {
                write!(f, "{}", factors.join("*"))?;
            } else {
                write!(f, "{}*{}", t.weight, factors.join("*"))?;
            }
        }
        if first {
            write!(f, "0")?;
        }
        for (k, h) in self.hyps.iter().enumerate() {
            write!(f, "{} h{k} = {}", if k == 0 { ";" } else { "," }, h.to_ex())?;
        }
        if let Some(p) = &self.param {
            write!(f, "; {} ≥ {}", p.var, p.lo)?;
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Search
// ═══════════════════════════════════════════════════════════════════════════

/// The basis of one LP stage: labelled product columns with their costs.
struct StageBasis {
    degree: u32,
    lambda_degree: u32,
    pairwise: bool,
    labels: Vec<PolyhedronTerm>,
    columns: Vec<Poly>,
    cost: Vec<Q>,
}

/// A prover for a **fixed** hypothesis set and parameter: parses the
/// hypotheses once, builds each stage's product basis once, and then
/// certifies any number of goals (or the emptiness of the set) against
/// them.  [`prove_nonnegative_on_polyhedron`] is
/// `PolyhedronProver::new(hyps, param, opts)?.prove(goal)`.
///
/// The free variables are the symbols of the hypotheses other than the
/// parameter, in name order; a goal may only use those (and the parameter).
///
/// ```
/// use symplex::prelude::*;
/// use symplex::certificates::{PolyhedronOpts, PolyhedronProver};
///
/// let ctx = Context::new();
/// let (j, r, t) = (ctx.symbol("j"), ctx.symbol("r"), ctx.symbol("t"));
/// let hyps = [r.clone(), ctx.rational(1, 2) - &r, t.clone(), 1 - &t, (&j * 2 + 1) * &t - &j * &r - 1];
/// let prover = PolyhedronProver::new(&hyps, Some((&j, &ctx.int(2))), &PolyhedronOpts::default()).unwrap();
/// // Every facet of the cell is a goal on the cell.
/// for h in &hyps {
///     assert!(prover.prove(h).unwrap().is_proved());
/// }
/// assert!(prover.prove(&((&j * 2 + 1) * &t * 4 - &j * &r * 4 - &r - 3)).unwrap().is_proved());
/// assert!(!prover.prove_empty().unwrap().is_proved());   // the cell is not empty
/// ```
pub struct PolyhedronProver {
    ctx: Context,
    gens: Vec<Ex>,
    hyps: Vec<Poly>,
    /// The hypotheses as exact arena-free polynomials, for refutation.
    hyps_exact: Vec<Exact>,
    param: Option<(Param, Q)>,
    opts: PolyhedronOpts,
    stages: Vec<StageBasis>,
    one: Poly,
    /// `jᵃ` and `(j − j₀)ᵃ` for `a = 0..=top`.
    var_pows: Vec<Poly>,
    shift_pows: Vec<Poly>,
}

impl PolyhedronProver {
    /// Parse the hypotheses and build the bases of every stage of `opts`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] for an empty hypothesis list, a
    /// non-polynomial or parametric-coefficient hypothesis, a non-symbol
    /// parameter or a non-literal `j₀`.
    pub fn new(
        hyps: &[Ex],
        param: Option<(&Ex, &Ex)>,
        opts: &PolyhedronOpts,
    ) -> Result<Self, SymplexError> {
        let Some(first) = hyps.first() else {
            return Err(invalid("at least one hypothesis is required"));
        };
        let ctx = first.context();

        let param_info = match param {
            Some((var, lo)) => {
                let lo = lo.eval();
                let Some(lo_q) = lo.as_rational() else {
                    return Err(invalid(format!(
                        "the parameter bound must be a rational literal, got `{lo}`"
                    )));
                };
                if var.free_symbols().len() != 1 || var.free_symbols()[0] != *var {
                    return Err(invalid(format!(
                        "the parameter must be a symbol, got `{var}`"
                    )));
                }
                Some((
                    Param {
                        var: var.clone(),
                        lo,
                        var_nonneg: !lo_q.is_negative(),
                        lo_zero: lo_q.is_zero(),
                    },
                    lo_q,
                ))
            }
            None => None,
        };

        // Generators: free variables of the hypotheses (name order), then the parameter.
        let mut names: Vec<(String, Ex)> = Vec::new();
        for e in hyps {
            for s in e.free_symbols() {
                if param.is_some_and(|(v, _)| *v == s) {
                    continue;
                }
                let n = s.to_string();
                if !names.iter().any(|(m, _)| *m == n) {
                    names.push((n, s));
                }
            }
        }
        names.sort_by(|a, b| a.0.cmp(&b.0));
        let mut gens: Vec<Ex> = names.into_iter().map(|(_, s)| s).collect();
        if let Some((p, _)) = &param_info {
            gens.push(p.var.clone());
        }
        if gens.is_empty() {
            return Err(invalid("the hypotheses contain no variables"));
        }
        let gen_refs: Vec<&Ex> = gens.iter().collect();
        let hyp_polys: Vec<Poly> = hyps
            .iter()
            .map(|h| to_poly(h, &gen_refs, "hypothesis"))
            .collect::<Result<_, _>>()?;
        let one = Poly::one(&ctx, &gen_refs)?;

        // Parameter atom powers, high enough for every stage.
        let top = opts
            .stages()
            .iter()
            .map(|&(d, l, _)| (d + 1).max(l))
            .max()
            .unwrap_or(1) as usize;
        let (var_pows, shift_pows) = match &param_info {
            Some((p, _)) => {
                let var =
                    Poly::new(&p.var, &gen_refs).ok_or_else(|| invalid("internal: parameter"))?;
                let shift = Poly::new(&(&p.var - &p.lo), &gen_refs)
                    .ok_or_else(|| invalid("internal: parameter shift"))?;
                let mut vp = vec![one.clone()];
                let mut sp = vec![one.clone()];
                for _ in 0..top {
                    let lv = vp.last().cloned().unwrap_or_else(|| one.clone());
                    let ls = sp.last().cloned().unwrap_or_else(|| one.clone());
                    vp.push(lv.mul(&var)?);
                    sp.push(ls.mul(&shift)?);
                }
                (vp, sp)
            }
            None => (vec![one.clone()], vec![one.clone()]),
        };

        let hyps_exact = hyp_polys
            .iter()
            .map(|h| {
                h.to_multipoly()
                    .ok_or_else(|| invalid("internal: non-rational hypothesis coefficient"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut prover = PolyhedronProver {
            ctx,
            gens,
            hyps: hyp_polys,
            hyps_exact,
            param: param_info,
            opts: opts.clone(),
            stages: Vec::new(),
            one,
            var_pows,
            shift_pows,
        };
        for (degree, lambda_degree, pairwise) in opts.stages() {
            let basis = prover.build_stage(degree, lambda_degree, pairwise)?;
            prover.stages.push(basis);
        }
        Ok(prover)
    }

    /// The hypotheses as polynomials, in input order.
    pub fn hyps(&self) -> &[Poly] {
        &self.hyps
    }

    /// The generators: free variables (name order) then the parameter.
    pub fn gens(&self) -> &[Ex] {
        &self.gens
    }

    /// The parameter `(var, lo)`, if any.
    pub fn parameter(&self) -> Option<(&Ex, &Ex)> {
        self.param.as_ref().map(|(p, _)| (&p.var, &p.lo))
    }

    /// The search options.
    pub fn opts(&self) -> &PolyhedronOpts {
        &self.opts
    }

    /// The parameter multipliers `(a, b)` for `jᵃ (j − j₀)ᵇ` with `a + b ≤ max`.
    fn multipliers(&self, max: u32) -> Vec<(u32, u32)> {
        let Some((p, _)) = &self.param else {
            return vec![(0, 0)];
        };
        let mut out = Vec::new();
        for a in 0..=max {
            if a > 0 && !p.var_nonneg {
                break;
            }
            for b in 0..=(max - a) {
                if b > 0 && !p.uses_shift() {
                    break;
                }
                out.push((a, b));
            }
        }
        out
    }

    fn mult_poly(&self, a: u32, b: u32) -> Result<Poly, SymplexError> {
        self.var_pows[a as usize].mul(&self.shift_pows[b as usize])
    }

    fn build_stage(
        &self,
        degree: u32,
        lambda_degree: u32,
        pairwise: bool,
    ) -> Result<StageBasis, SymplexError> {
        let mut labels: Vec<PolyhedronTerm> = Vec::new();
        let mut columns: Vec<Poly> = Vec::new();
        let mut cost: Vec<Q> = Vec::new();
        let mut push = |t: PolyhedronTerm, p: Poly, c: u32| {
            labels.push(t);
            columns.push(p);
            cost.push(Q::from_integer(BigInt::from(c)));
        };
        for (k, h) in self.hyps.iter().enumerate() {
            for (a, b) in self.multipliers(degree) {
                push(
                    PolyhedronTerm {
                        hyps: vec![k],
                        var_power: a,
                        shift_power: b,
                        weight: Q::zero(),
                    },
                    self.mult_poly(a, b)?.mul(h)?,
                    1 + 2 * (a + b),
                );
            }
        }
        if self.param.is_some() {
            for (a, b) in self.multipliers(degree + 1) {
                if a + b == 0 {
                    continue;
                }
                push(
                    PolyhedronTerm {
                        hyps: vec![],
                        var_power: a,
                        shift_power: b,
                        weight: Q::zero(),
                    },
                    self.mult_poly(a, b)?,
                    1 + 2 * (a + b),
                );
            }
        }
        push(
            PolyhedronTerm {
                hyps: vec![],
                var_power: 0,
                shift_power: 0,
                weight: Q::zero(),
            },
            self.one.clone(),
            1,
        );
        if pairwise {
            for k in 0..self.hyps.len() {
                for l in k..self.hyps.len() {
                    let hh = self.hyps[k].mul(&self.hyps[l])?;
                    for (a, b) in self.multipliers(1) {
                        push(
                            PolyhedronTerm {
                                hyps: vec![k, l],
                                var_power: a,
                                shift_power: b,
                                weight: Q::zero(),
                            },
                            self.mult_poly(a, b)?.mul(&hh)?,
                            1 + 2 * (a + b),
                        );
                    }
                }
            }
        }
        Ok(StageBasis {
            degree,
            lambda_degree,
            pairwise,
            labels,
            columns,
            cost,
        })
    }

    /// Parse a goal in this prover's generators.
    fn goal_poly(&self, goal: &Ex) -> Result<Poly, SymplexError> {
        let gen_refs: Vec<&Ex> = self.gens.iter().collect();
        for s in goal.free_symbols() {
            if !self.gens.contains(&s) {
                return Err(invalid(format!(
                    "goal `{goal}` mentions `{s}`, which is not a variable of the hypotheses"
                )));
            }
        }
        to_poly(goal, &gen_refs, "goal")
    }

    /// Prove `goal ≥ 0` on the set (for every admissible parameter value),
    /// refute it with an exact point, or report `Unknown`.  See
    /// [`prove_nonnegative_on_polyhedron`].
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if the goal is not a
    /// rational-coefficient polynomial in the prover's variables.
    pub fn prove(&self, goal: &Ex) -> Result<PolyhedronOutcome, SymplexError> {
        let goal_poly = self.goal_poly(goal)?;
        self.prove_poly(&goal_poly)
    }

    /// [`prove`](Self::prove) for a goal that is already a [`Poly`] — the
    /// entry point for tools that keep their polynomials exact
    /// (`MultiPoly` → [`Poly::from_multipoly`]) and want to skip the
    /// expression round trip.  The goal's generators may be any subset of
    /// the prover's ([`gens`](Self::gens)) in any order; a coefficient must
    /// be rational.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if a generator of the goal is not a
    /// variable of the hypotheses, or a coefficient is symbolic.
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::certificates::{PolyhedronOpts, PolyhedronOutcome, PolyhedronProver};
    /// use symplex::multipoly::MultiPoly;
    /// use symplex::poly_ex::Poly;
    ///
    /// let ctx = Context::new();
    /// let (j, r) = (ctx.symbol("j"), ctx.symbol("r"));
    /// // r ≥ 0 and j·(1 − r) ≥ 0 for j ≥ 1; goal j − j·r ≥ 0.
    /// let prover = PolyhedronProver::new(&[r.clone(), &j * (1 - &r)], Some((&j, &ctx.int(1))), &PolyhedronOpts::default())?;
    /// let [mj, mr]: [MultiPoly; 2] = [MultiPoly::var(2, 0), MultiPoly::var(2, 1)];   // (j, r) order
    /// let goal = Poly::from_multipoly(&ctx, &[&j, &r], &mj.sub(&mj.mul(&mr)))?;
    /// assert!(matches!(prover.prove_poly(&goal)?, PolyhedronOutcome::Proved(_)));
    /// # Ok::<(), SymplexError>(())
    /// ```
    pub fn prove_poly(&self, goal: &Poly) -> Result<PolyhedronOutcome, SymplexError> {
        if goal.gens() == self.gens.as_slice() {
            return self.prove_poly_aligned(goal);
        }
        // Re-map the exponent vectors onto the prover's generator order.
        let positions: Vec<usize> = goal
            .gens()
            .iter()
            .map(|g| {
                self.gens.iter().position(|s| s == g).ok_or_else(|| {
                    invalid(format!(
                        "goal mentions `{g}`, which is not a variable of the hypotheses"
                    ))
                })
            })
            .collect::<Result<_, _>>()?;
        let width = self.gens.len();
        let terms: Vec<(Vec<u32>, Ex)> = goal
            .terms_iter()
            .map(|(m, c)| {
                let mut e = vec![0u32; width];
                for (k, &pos) in positions.iter().enumerate() {
                    e[pos] = m.get(k).copied().unwrap_or(0);
                }
                (e, c.clone())
            })
            .collect();
        let gen_refs: Vec<&Ex> = self.gens.iter().collect();
        let aligned = Poly::from_terms(&self.ctx, &gen_refs, terms)?;
        if !aligned.has_rational_coeffs() {
            return Err(invalid("the goal has a symbolic coefficient"));
        }
        self.prove_poly_aligned(&aligned)
    }

    /// Prove that the set is empty for every admissible parameter value
    /// (the goal `−1`), exhibit a point of it, or report `Unknown`.  See
    /// [`prove_polyhedron_empty`].
    ///
    /// # Errors
    ///
    /// Only internal failures (the goal is a constant).
    pub fn prove_empty(&self) -> Result<PolyhedronOutcome, SymplexError> {
        let gen_refs: Vec<&Ex> = self.gens.iter().collect();
        let minus_one = Poly::constant(&self.ctx, &gen_refs, &self.ctx.int(-1))?;
        self.prove_poly_aligned(&minus_one)
    }

    /// The search proper, for a goal over exactly the prover's generators.
    ///
    /// The cheapest stage runs first: most true goals are certified there,
    /// and a certified goal cannot have a counterexample, so the exact
    /// refutation (ten sampled parameter values, one small LP each) is
    /// only paid by goals the first stage does not settle.  Outcomes are
    /// exactly those of "refute first": a goal is refuted iff it is false,
    /// and proved with the same certificate iff some stage finds one.
    fn prove_poly_aligned(&self, goal: &Poly) -> Result<PolyhedronOutcome, SymplexError> {
        let mut tried = (0u32, 0u32, false);
        let mut stages = self.stages.iter();
        if let Some(first) = stages.next() {
            tried = (first.degree, first.lambda_degree, first.pairwise);
            if let Some(cert) = self.search_stage(goal, first)? {
                return Ok(PolyhedronOutcome::Proved(cert));
            }
        }
        // Exact refutation on sampled parameter values.
        let goal_exact = goal
            .to_multipoly()
            .ok_or_else(|| invalid("internal: non-rational goal coefficient"))?;
        if let Some((point, param_value, value)) = refute(
            &goal_exact,
            &self.hyps_exact,
            &self.gens,
            self.param.as_ref(),
        ) {
            return Ok(Outcome::Refuted {
                point,
                param_value,
                value,
            });
        }
        // The remaining stages, smallest basis first.
        for stage in stages {
            tried = (
                tried.0.max(stage.degree),
                tried.1.max(stage.lambda_degree),
                tried.2 || stage.pairwise,
            );
            if let Some(cert) = self.search_stage(goal, stage)? {
                return Ok(PolyhedronOutcome::Proved(cert));
            }
        }
        Ok(Outcome::Unknown(PolyhedronUnknown {
            degree: tried.0,
            lambda_degree: tried.1,
            pairwise: tried.2,
        }))
    }

    /// One LP stage.  `Ok(Some(cert))` with a verified certificate,
    /// `Ok(None)` when the LP is infeasible.
    fn search_stage(
        &self,
        goal: &Poly,
        stage: &StageBasis,
    ) -> Result<Option<PolyhedronCertificate>, SymplexError> {
        let n_basis = stage.columns.len();
        let use_var = self.param.as_ref().is_some_and(|(p, _)| p.var_nonneg);
        // λ columns: −atomᵃ·goal for a = 1..=lambda_degree.
        let lambda_cols: Vec<Poly> = if self.param.is_some() {
            (1..=stage.lambda_degree as usize)
                .map(|a| {
                    let atom = if use_var {
                        &self.var_pows[a]
                    } else {
                        &self.shift_pows[a]
                    };
                    atom.mul(goal).map(|p| p.neg())
                })
                .collect::<Result<_, _>>()?
        } else {
            Vec::new()
        };
        let mut cost = stage.cost.clone();
        for a in 1..=lambda_cols.len() {
            cost.push(Q::from_integer(BigInt::from(1 + 20 * a as u32)));
        }

        // Monomial rows.
        let mut all: Vec<&Poly> = stage.columns.iter().chain(&lambda_cols).collect();
        all.push(goal);
        let monos = Poly::monomial_basis(&all)?;
        let coeff = |p: &Poly, m: &[u32]| -> Result<Q, SymplexError> {
            p.coeff_monomial(m)?
                .as_rational()
                .ok_or_else(|| invalid("internal: non-rational coefficient"))
        };
        let mut lp = LpProblem::minimize(cost);
        for m in &monos {
            let mut row: Vec<Q> = Vec::with_capacity(n_basis + lambda_cols.len());
            for c in stage.columns.iter().chain(&lambda_cols) {
                row.push(coeff(c, m)?);
            }
            lp = lp.eq(row, coeff(goal, m)?);
        }
        let started = std::time::Instant::now();
        let sol = lp.solve()?;
        tracing::debug!(
            target: "symplex::certificates::polyhedron",
            degree = stage.degree,
            lambda_degree = stage.lambda_degree,
            pairwise = stage.pairwise,
            rows = monos.len(),
            cols = n_basis + lambda_cols.len(),
            status = ?sol.status,
            micros = started.elapsed().as_micros() as u64,
            "polyhedron stage LP"
        );
        if sol.status != LpStatus::Optimal {
            return Ok(None);
        }
        let terms: Vec<PolyhedronTerm> = stage
            .labels
            .iter()
            .zip(&sol.x[..n_basis])
            .filter(|(_, w)| w.is_positive())
            .map(|(t, w)| PolyhedronTerm {
                weight: w.clone(),
                ..t.clone()
            })
            .collect();
        let mut lambda: Vec<Q> = vec![Q::one()];
        lambda.extend(sol.x[n_basis..].iter().cloned());
        while lambda.len() > 1 && lambda.last().is_some_and(Zero::is_zero) {
            lambda.pop();
        }
        let cert = PolyhedronCertificate {
            goal: goal.clone(),
            hyps: self.hyps.clone(),
            param: self.param.as_ref().map(|(p, _)| p.clone()),
            lambda,
            terms,
        };
        if !cert.verify() {
            return Err(SymplexError::ComputationFailed {
                operation: OP,
                reason:
                    "the LP solution did not reproduce the identity under exact re-verification"
                        .into(),
            });
        }
        Ok(Some(cert))
    }
}

/// Parse `e` as a rational-coefficient polynomial in `gens`.
fn to_poly(e: &Ex, gens: &[&Ex], what: &str) -> Result<Poly, SymplexError> {
    let p = Poly::try_new(e, gens).map_err(|err| match err {
        SymplexError::InvalidArgument { reason, .. } => invalid(format!("{what} `{e}`: {reason}")),
        other => other,
    })?;
    if !p.has_rational_coeffs() {
        return Err(invalid(format!(
            "{what} `{e}` must have rational coefficients"
        )));
    }
    Ok(p)
}

/// Prove `goal ≥ 0` on `{x : hₖ(j, x) ≥ 0 ∀k}` for every real `j ≥ j₀`
/// (`param = Some((j, j₀))`), or on the fixed polyhedron `{hₖ(x) ≥ 0}`
/// (`param = None`), by the identity described in the
/// [`certificates`](crate::certificates) module documentation; refute it
/// with an exact point of the set where the goal is negative; or report
/// `Unknown`.  To certify many goals against the same hypotheses, build a
/// [`PolyhedronProver`] once.
///
/// The free variables are the symbols of `hyps` other than the parameter,
/// in name order; the goal may only use those.  All expressions must be
/// polynomials with rational coefficients; `j₀` must be a rational literal.
/// Powers of `j` itself are only used when `j₀ ≥ 0`; powers of `j − j₀`
/// always.
///
/// Refutation samples the parameter at `j₀, j₀ + 1, …` and a few larger
/// values and, when the hypotheses and the goal are affine in the free
/// variables, minimises the goal over the polyhedron exactly; the returned
/// point is re-checked by evaluation.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for an empty hypothesis list, a
/// non-polynomial or parametric-coefficient input, a goal mentioning a
/// symbol absent from the hypotheses, or a non-literal `j₀`.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::certificates::{prove_nonnegative_on_polyhedron, PolyhedronOpts, PolyhedronOutcome};
///
/// let ctx = Context::new();
/// let (j, r) = (ctx.symbol("j"), ctx.symbol("r"));
/// // On { 0 ≤ r ≤ 1/2 } and j ≥ 1:  j·r + 1 − 2r ≥ 0  needs no λ,
/// let goal = &j * &r + 1 - &r * 2;
/// let hyps = [r.clone(), ctx.rational(1, 2) - &r];
/// let out = prove_nonnegative_on_polyhedron(&goal, &hyps, Some((&j, &ctx.int(1))), &PolyhedronOpts::default()).unwrap();
/// let cert = out.certificate().expect("proved");
/// assert!(cert.verify());
/// assert!(cert.to_lean("jr_bound").unwrap().contains("linarith only"));
///
/// // …while  j − 2·j·r ≥ 0  on the same set is false at r = 1/2 for every j > 0.
/// match prove_nonnegative_on_polyhedron(&(&j - &j * &r * 2 - 1), &hyps, Some((&j, &ctx.int(1))), &PolyhedronOpts::default()).unwrap() {
///     PolyhedronOutcome::Refuted { value, param_value, .. } => {
///         assert!(value < symplex::linprog::qi(0));
///         assert_eq!(param_value, Some(symplex::linprog::qi(1)));
///     }
///     other => panic!("{other:?}"),
/// }
/// ```
pub fn prove_nonnegative_on_polyhedron(
    goal: &Ex,
    hyps: &[Ex],
    param: Option<(&Ex, &Ex)>,
    opts: &PolyhedronOpts,
) -> Result<PolyhedronOutcome, SymplexError> {
    PolyhedronProver::new(hyps, param, opts)?.prove(goal)
}

/// Prove that `{x : hₖ(j, x) ≥ 0 ∀k}` is empty for every `j ≥ j₀`
/// (`Proved`), exhibit a point of it (`Refuted { point, .. }`), or report
/// `Unknown`.  This is [`prove_nonnegative_on_polyhedron`] with the goal
/// `−1`: the certificate `λ(j)·(−1) = Σ μ·(…)` with `μ ≥ 0` is a
/// contradiction on the set.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::certificates::{prove_polyhedron_empty, PolyhedronOpts};
///
/// let ctx = Context::new();
/// let (j, r) = (ctx.symbol("j"), ctx.symbol("r"));
/// // r ≥ 1/2 and j·r ≤ j/2 − 1 cannot both hold for j ≥ 0.
/// let hyps = [&r - ctx.rational(1, 2), &j / 2 - 1 - &j * &r];
/// let out = prove_polyhedron_empty(&hyps, Some((&j, &ctx.int(0))), &PolyhedronOpts::default()).unwrap();
/// assert!(out.is_proved());
/// ```
pub fn prove_polyhedron_empty(
    hyps: &[Ex],
    param: Option<(&Ex, &Ex)>,
    opts: &PolyhedronOpts,
) -> Result<PolyhedronOutcome, SymplexError> {
    PolyhedronProver::new(hyps, param, opts)?.prove_empty()
}

// ═══════════════════════════════════════════════════════════════════════════
// Refutation
// ═══════════════════════════════════════════════════════════════════════════

/// `(point, sampled parameter value, goal value)` of a counterexample.
type Refutation = (Vec<(Ex, Q)>, Option<Q>, Q);

/// A hypothesis or goal as an exact arena-free polynomial over the
/// prover's generators (free variables, then the parameter).
type Exact = MultiPoly<GrevLex>;

/// Search for a point of the set where the goal is negative: sample the
/// parameter, and when everything is affine in the free variables,
/// minimise the goal over the polyhedron exactly (inside a large box so
/// that an unbounded direction still yields a witness).  Pure rational
/// arithmetic on [`MultiPoly`]s; no expression arena is touched.
fn refute(
    goal: &Exact,
    hyps: &[Exact],
    gens: &[Ex],
    param: Option<&(Param, Q)>,
) -> Option<Refutation> {
    let width = gens.len();
    // The parameter, when present, is the last generator; the free
    // variables are the others.
    let n = if param.is_some() { width - 1 } else { width };
    let samples: Vec<Option<Q>> = match param {
        Some((_, lo)) => {
            let mut s: Vec<Q> = (0..=6)
                .map(|k| lo + Q::from_integer(BigInt::from(k)))
                .collect();
            for big in [10i64, 100, 1000] {
                s.push(lo + Q::from_integer(BigInt::from(big)));
            }
            s.into_iter().map(Some).collect()
        }
        None => vec![None],
    };
    // Affine data `(constant, coefficients over the free variables)` of a
    // polynomial at the sample, or `None` if it is not affine there.
    let affine_at = |p: &Exact, jv: &Option<Q>| -> Option<(Q, Vec<Q>)> {
        let at = match jv {
            Some(v) => p.eval_var(n, v),
            None => p.clone(),
        };
        let (coeffs, constant) = at.affine_form()?;
        Some((constant, coeffs[..n].to_vec()))
    };
    for jv in samples {
        let Some((g0, gc)) = affine_at(goal, &jv) else {
            continue;
        };
        let Some(affine_hyps) = hyps
            .iter()
            .map(|h| affine_at(h, &jv))
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        if n == 0 {
            // Constant goal on a set whose hypotheses are constants too.
            let feasible = affine_hyps.iter().all(|(k, _)| !k.is_negative());
            if feasible && g0.is_negative() {
                let point = match (&jv, param) {
                    (Some(v), Some((p, _))) => vec![(p.var.clone(), v.clone())],
                    _ => Vec::new(),
                };
                return Some((point, jv.clone(), g0));
            }
            continue;
        }
        let bound = Q::from_integer(BigInt::from(1_000_000));
        let mut lp = LpProblem::minimize(gc.clone());
        for i in 0..n {
            lp = lp.bounds(i, Some(-bound.clone()), Some(bound.clone()));
        }
        for (k, c) in &affine_hyps {
            lp = lp.ge(c.clone(), -k);
        }
        let sol = lp.solve().ok()?;
        if sol.status != LpStatus::Optimal {
            continue;
        }
        let value = sol.objective? + &g0;
        if !value.is_negative() {
            continue;
        }
        // Re-check by exact evaluation of the original polynomials.
        let mut values: Vec<Q> = sol.x.clone();
        if let Some(v) = &jv {
            values.push(v.clone());
        }
        if values.len() != width {
            continue;
        }
        let gv = goal.eval(&values);
        if !gv.is_negative() {
            continue;
        }
        if hyps.iter().all(|h| !h.eval(&values).is_negative()) {
            let point: Vec<(Ex, Q)> = gens.iter().cloned().zip(values).collect();
            return Some((point, jv.clone(), gv));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domains::linprog::q;

    fn setup() -> (Context, Ex, Ex, Ex) {
        let ctx = Context::new();
        let (j, r, t) = (ctx.symbol("j"), ctx.symbol("r"), ctx.symbol("t"));
        (ctx, j, r, t)
    }

    #[test]
    fn stages_are_ordered_small_to_large() {
        assert_eq!(
            PolyhedronOpts::default().stages(),
            vec![
                (1, 0, false),
                (1, 1, false),
                (2, 1, false),
                (2, 2, false),
                (3, 2, false),
                (3, 3, false),
                (2, 2, true)
            ]
        );
        assert_eq!(PolyhedronOpts::single(2, 1).stages(), vec![(2, 1, false)]);
        let no_lambda = PolyhedronOpts {
            max_lambda_degree: 0,
            pairwise: false,
            ..Default::default()
        };
        assert_eq!(
            no_lambda.stages(),
            vec![(1, 0, false), (2, 0, false), (3, 0, false)]
        );
    }

    #[test]
    fn lambda_is_needed_and_found() {
        let (ctx, j, r, t) = setup();
        let hyps = [&t - &r, &t + &j * &r - &j - 1];
        let out = prove_nonnegative_on_polyhedron(
            &(&t - 1),
            &hyps,
            Some((&j, &ctx.int(0))),
            &PolyhedronOpts::default(),
        )
        .unwrap();
        let c = out.certificate().expect("proved");
        assert!(c.verify());
        assert_eq!(c.lambda_coeffs(), &[q(1, 1), q(1, 1)]);
        assert_eq!(c.lambda(), &j + 1);
        assert_eq!(
            c.to_string(),
            "(j + 1)*(t - 1) = j*h0 + h1; h0 = -r + t, h1 = j*r - j + t - 1; j ≥ 0"
        );
        let (lhs, rhs) = c.identity();
        assert!((lhs - rhs).expand().is_zero_structural());
        // Without λ there is no certificate at any degree we try.
        let none = prove_nonnegative_on_polyhedron(
            &(&t - 1),
            &hyps,
            Some((&j, &ctx.int(0))),
            &PolyhedronOpts {
                max_lambda_degree: 0,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(matches!(
            none,
            Outcome::Unknown(PolyhedronUnknown {
                lambda_degree: 0,
                degree: 3,
                pairwise: true
            })
        ));
    }

    #[test]
    fn refutation_finds_exact_point() {
        let (ctx, j, r, t) = setup();
        let hyps = [
            r.clone(),
            ctx.rational(1, 2) - &r,
            t.clone(),
            1 - &t,
            (&j * 2 + 1) * &t - &j * &r - 1,
        ];
        match prove_nonnegative_on_polyhedron(
            &(&t - ctx.rational(1, 2) - &r),
            &hyps,
            Some((&j, &ctx.int(2))),
            &PolyhedronOpts::default(),
        )
        .unwrap()
        {
            PolyhedronOutcome::Refuted { point, value, .. } => {
                assert!(value.is_negative());
                assert_eq!(point.len(), 3);
                assert_eq!(point[2].0, j);
                assert!(point[2].1 >= q(2, 1));
            }
            other => panic!("{other:?}"),
        }
        // Emptiness refuted = a point of the cell.
        match prove_polyhedron_empty(&hyps, Some((&j, &ctx.int(2))), &PolyhedronOpts::default())
            .unwrap()
        {
            PolyhedronOutcome::Refuted { value, .. } => assert_eq!(value, q(-1, 1)),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn emptiness_certificate() {
        let (ctx, j, r, t) = setup();
        let hyps = [
            &r - ctx.rational(1, 2),
            (&j * 2 + 1) * &t - &j * &r - 1,
            ctx.rational(1, 4) - &t,
        ];
        let c = prove_polyhedron_empty(&hyps, Some((&j, &ctx.int(2))), &PolyhedronOpts::default())
            .unwrap();
        let c = c.certificate().expect("empty");
        assert!(c.proves_emptiness());
        assert!(c.verify());
        assert!(c.to_lean("e").unwrap().contains("False := by"));
    }

    #[test]
    fn no_parameter_and_pairwise() {
        let (ctx, _j, r, t) = setup();
        let fixed = [r.clone(), 1 - &r, &t - &r];
        let c = prove_nonnegative_on_polyhedron(
            &(&t * 2 - &r),
            &fixed,
            None,
            &PolyhedronOpts::default(),
        )
        .unwrap();
        let c = c.certificate().unwrap();
        assert!(c.parameter().is_none());
        assert!(c.lambda_is_one());
        assert_eq!(
            c.to_string(),
            "-r + 2*t = h0 + 2*h2; h0 = r, h1 = -r + 1, h2 = -r + t"
        );
        let sq = prove_nonnegative_on_polyhedron(
            &(&r - &r.powi(2)),
            &fixed[..2],
            None,
            &PolyhedronOpts::default(),
        )
        .unwrap();
        let sq = sq.certificate().unwrap();
        assert!(sq.uses_pairwise());
        assert_eq!(sq.terms()[0].hyps, vec![0, 1]);
        let no_pairs = prove_nonnegative_on_polyhedron(
            &(&r - &r.powi(2)),
            &fixed[..2],
            None,
            &PolyhedronOpts {
                pairwise: false,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(matches!(no_pairs, Outcome::Unknown(_)));
        let _ = ctx;
    }

    #[test]
    fn negative_parameter_bound_uses_only_shift_powers() {
        let (ctx, j, r, t) = setup();
        let hyps = [&t - &r, r.clone()];
        let c = prove_nonnegative_on_polyhedron(
            &((&j + 1) * &t),
            &hyps,
            Some((&j, &ctx.int(-1))),
            &PolyhedronOpts::default(),
        )
        .unwrap();
        let c = c.certificate().unwrap();
        assert!(
            c.terms()
                .iter()
                .all(|t| t.var_power == 0 && t.shift_power == 1)
        );
        let lean = c.to_lean("neg").unwrap();
        assert!(lean.contains("have hK0 : (0 : ℝ) ≤ j + 1 := by linarith"));
        assert!(!lean.contains("hJ0"));
    }

    #[test]
    fn lean_steps_use_caller_names() {
        let (ctx, j, r, t) = setup();
        let hyps = [&t - &r, r.clone()];
        let c = prove_nonnegative_on_polyhedron(
            &(&j * (&j - 2) * &t),
            &hyps,
            Some((&j, &ctx.int(2))),
            &PolyhedronOpts::default(),
        )
        .unwrap();
        let c = c.certificate().unwrap();
        let steps = c
            .lean_steps(
                &PolyhedronLeanNames {
                    hyps: &["e4", "e7"],
                    param_nonneg: "hJ0",
                    shift_nonneg: "hK0",
                },
                &LeanOpts::default(),
            )
            .unwrap();
        assert_eq!(
            steps.haves,
            vec![
                "have e4K := mul_nonneg hK0 e4",
                "have e4JK := mul_nonneg hJ0 e4K",
                "have e7K := mul_nonneg hK0 e7",
                "have e7JK := mul_nonneg hJ0 e7K",
            ]
        );
        assert_eq!(steps.hints, vec!["e4JK", "e7JK"]);
        assert!(steps.lambda.is_none());
        assert_eq!(steps.closing, vec!["linarith only [e4JK, e7JK]"]);
        assert_eq!(steps.to_block("    ").lines().count(), 5);
        assert!(
            c.lean_steps(
                &PolyhedronLeanNames {
                    hyps: &["e4"],
                    param_nonneg: "a",
                    shift_nonneg: "b"
                },
                &LeanOpts::default()
            )
            .is_err()
        );
    }

    #[test]
    fn invalid_inputs() {
        let (ctx, j, r, _t) = setup();
        assert!(
            prove_nonnegative_on_polyhedron(
                &r,
                &[],
                Some((&j, &ctx.int(0))),
                &PolyhedronOpts::default()
            )
            .is_err()
        );
        assert!(
            prove_nonnegative_on_polyhedron(
                &r.sin(),
                std::slice::from_ref(&r),
                None,
                &PolyhedronOpts::default()
            )
            .is_err()
        );
        assert!(
            prove_nonnegative_on_polyhedron(
                &r,
                std::slice::from_ref(&r),
                Some((&j, &ctx.pi())),
                &PolyhedronOpts::default()
            )
            .is_err()
        );
        assert!(
            prove_nonnegative_on_polyhedron(
                &r,
                std::slice::from_ref(&r),
                Some((&(&j + 1), &ctx.int(0))),
                &PolyhedronOpts::default()
            )
            .is_err()
        );
        assert!(prove_polyhedron_empty(&[], None, &PolyhedronOpts::default()).is_err());
    }
}

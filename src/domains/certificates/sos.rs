//! Sums-of-squares certificates: exact Gram-matrix decompositions
//! `g = Σₖ dₖ · pₖ(x)²` with rational `dₖ > 0` and rational-coefficient
//! polynomials `pₖ`, proving `g ≥ 0` on all of ℝⁿ.
//!
//! The search is the classical **Peyrl–Parrilo** pipeline made exact:
//!
//! 1. Write `g = mᵀ Q m` for the vector `m` of monomials of degree ≤ ½·deg g;
//!    the coefficient match is a linear system `A(Q) = b` and the question
//!    is whether it has a positive-semidefinite solution `Q`.
//! 2. Solve that SDP **numerically** with a small dense primal–dual
//!    interior-point method (no external solver; the Gram matrices here
//!    have a few dozen rows).  With a zero objective the central path
//!    converges to the analytic centre of the feasible set — the point best
//!    placed for rounding.
//! 3. If the numerical solution is singular (the goal has real zeros, so
//!    every Gram matrix is), read the kernel off its eigenvectors, make it
//!    exact by rounding a reduced row-echelon basis to small rationals, and
//!    **restrict** the search to the face `Q = B Q' Bᵀ` (facial reduction);
//!    repeat until the reduced problem has an interior.
//! 4. **Round** the numerical solution to rationals, **project** it back onto
//!    the affine constraints exactly (a rational least-norm correction), and
//!    check positive semidefiniteness exactly with a rational `L·D·Lᵀ`
//!    ([`QMatrix::ldl_psd`]).  The factorisation *is* the certificate:
//!    `mᵀ Q m = Σₖ dₖ (Lᵀm)ₖ²`.
//!
//! Every certificate is re-verified by expanding `Σ dₖ pₖ²` with exact
//! polynomial arithmetic and comparing with the goal.  Goals that are
//! non-negative but not sums of squares (Motzkin's polynomial) come back
//! as `Unknown`; goals with a negative value are refuted with an exact
//! rational point.

use std::fmt;
use std::time::{Duration, Instant};

use num_bigint::BigInt;
use num_traits::{One, Signed, ToPrimitive, Zero};

use crate::api::context::Context;
use crate::api::eq::Equation;
use crate::api::expr::Ex;
use crate::api::poly_ex::Poly;
use crate::base::errors::SymplexError;
use crate::domains::certificates::serial::{q_from_str, q_to_str};
use crate::domains::certificates::{Certificate, Outcome};
use crate::domains::exact_matrix::QMatrix;
use crate::domains::linprog::{
    Budget, BudgetHit, LpProblem, LpStatus, Q, deadline_from, deadline_passed,
};
use crate::output::lean::{LeanOpts, MATHLIB_LINE_WIDTH, lean_ident, wrap_lean};
use crate::output::tree::ExprTree;

const OP: &str = "prove_sos";

fn invalid(reason: impl Into<String>) -> SymplexError {
    SymplexError::invalid_argument(OP, reason)
}

// ═══════════════════════════════════════════════════════════════════════════
// Options, outcome, certificate
// ═══════════════════════════════════════════════════════════════════════════

/// Search limits for [`prove_sos`].
///
/// **Budget.**  A call may be bounded by a deadline
/// ([`with_deadline`](Self::with_deadline), absolute, or
/// [`with_time_limit`](Self::with_time_limit), measured from the start of
/// the call).  The interior-point loop checks it between iterations and
/// the facial-reduction loop between rounds; when it passes the answer is
/// `Unknown` with a reason starting `budget exhausted: deadline`.
///
/// `#[non_exhaustive]`: build it with [`Default`] and the `with_*`
/// builders (or assign fields on a `mut` default), so that a future option
/// is not a breaking change.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct SosOpts {
    /// Largest Gram matrix (number of monomials) the search will set up.
    pub max_basis: usize,
    /// Interior-point iterations per SDP solve.
    pub max_iterations: usize,
    /// Rounding denominators to try, in order (each `10^k`).
    pub rounding_digits: Vec<u32>,
    /// Rounds of numerically guided facial reduction.
    pub max_facial_reductions: usize,
    /// Absolute deadline for each [`prove_sos`] call.
    pub deadline: Option<Instant>,
    /// Time allowed for each [`prove_sos`] call, measured from its start.
    /// When both this and `deadline` are set the earlier one applies.
    pub time_limit: Option<Duration>,
}

impl Default for SosOpts {
    /// `max_basis = 60`, `max_iterations = 80`, digits `[1, 2, 3, 5, 7, 9,
    /// 12]` (coarse first, for small rationals in the certificate),
    /// `max_facial_reductions = 3`, no deadline.
    fn default() -> Self {
        SosOpts {
            max_basis: 60,
            max_iterations: 80,
            rounding_digits: vec![1, 2, 3, 5, 7, 9, 12],
            max_facial_reductions: 3,
            deadline: None,
            time_limit: None,
        }
    }
}

impl SosOpts {
    /// Set [`max_basis`](Self::max_basis).
    #[must_use]
    pub fn with_max_basis(mut self, max_basis: usize) -> Self {
        self.max_basis = max_basis;
        self
    }

    /// Set [`max_iterations`](Self::max_iterations).
    #[must_use]
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set [`rounding_digits`](Self::rounding_digits).
    #[must_use]
    pub fn with_rounding_digits(mut self, rounding_digits: Vec<u32>) -> Self {
        self.rounding_digits = rounding_digits;
        self
    }

    /// Set [`max_facial_reductions`](Self::max_facial_reductions).
    #[must_use]
    pub fn with_max_facial_reductions(mut self, max_facial_reductions: usize) -> Self {
        self.max_facial_reductions = max_facial_reductions;
        self
    }

    /// Set [`deadline`](Self::deadline): every call stops at this instant.
    #[must_use]
    pub fn with_deadline(mut self, deadline: Instant) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Set [`time_limit`](Self::time_limit): every call gets this long
    /// from the moment it starts.
    #[must_use]
    pub fn with_time_limit(mut self, time_limit: Duration) -> Self {
        self.time_limit = Some(time_limit);
        self
    }

    /// The deadline of a call starting now: the earlier of `deadline` and
    /// now + `time_limit` (a limit too large to represent is no limit).
    fn deadline_from_now(&self) -> Option<Instant> {
        deadline_from(self.deadline, self.time_limit)
    }
}

/// The deadline passed (the interior-point loop or a facial-reduction
/// round was cut short).
struct DeadlinePassed;

/// `Err(DeadlinePassed)` once `deadline` is in the past.
fn check_deadline(deadline: Option<Instant>) -> Result<(), DeadlinePassed> {
    if deadline_passed(deadline) {
        Err(DeadlinePassed)
    } else {
        Ok(())
    }
}

/// Result of [`prove_sos`]: an [`Outcome`] with an [`SosCertificate`] or
/// an [`SosUnknown`].  A refutation's `point` lists the variables in the
/// order given to `prove_sos`.
pub type SosOutcome = Outcome<SosCertificate, SosUnknown>;

/// Why [`prove_sos`] could not decide: the goal may be non-negative
/// without being a sum of squares (Motzkin), or the numerical search did
/// not converge.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct SosUnknown {
    /// Why the search stopped, for diagnostics.
    pub reason: String,
    /// `Some(BudgetHit::Deadline)` when the call's time limit / deadline
    /// ran out (the `reason` then starts `budget exhausted: deadline`);
    /// `None` for every other cause.  Matches
    /// `PolyhedronUnknown::budget_exhausted`.
    pub budget_exhausted: Option<BudgetHit>,
}

impl SosUnknown {
    fn new(reason: impl Into<String>) -> Self {
        SosUnknown {
            reason: reason.into(),
            budget_exhausted: None,
        }
    }

    fn deadline(reason: impl Into<String>) -> Self {
        SosUnknown {
            reason: reason.into(),
            budget_exhausted: Some(BudgetHit::Deadline),
        }
    }
}

impl fmt::Display for SosUnknown {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.reason)
    }
}

/// A verified decomposition `goal = Σₖ dₖ · pₖ²`.
#[derive(Clone, Debug)]
pub struct SosCertificate {
    goal: Poly,
    /// Monomial exponent vectors of the Gram basis `m`.
    basis: Vec<Vec<u32>>,
    /// Exact PSD Gram matrix with `goal = mᵀ Q m`.
    gram: QMatrix,
    /// `(dₖ, pₖ)` with `dₖ > 0`.
    squares: Vec<(Q, Poly)>,
}

/// Serialisable form of an [`SosCertificate`].
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SosCertificateData {
    /// The variables.
    pub vars: Vec<ExprTree>,
    /// The goal.
    pub goal: ExprTree,
    /// Monomial exponent vectors of the Gram basis.
    pub basis: Vec<Vec<u32>>,
    /// The Gram matrix, row-major, as `"p/q"` rationals.
    pub gram: Vec<String>,
}

impl SosCertificate {
    /// The goal.
    pub fn goal(&self) -> &Poly {
        &self.goal
    }

    /// The variables, in order.
    pub fn vars(&self) -> &[Ex] {
        self.goal.gens()
    }

    /// Exponent vectors of the monomials `m` with `goal = mᵀ Q m`.
    pub fn basis(&self) -> &[Vec<u32>] {
        &self.basis
    }

    /// The exact positive-semidefinite Gram matrix.
    pub fn gram(&self) -> &QMatrix {
        &self.gram
    }

    /// The squares: `(dₖ, pₖ)` with `goal = Σ dₖ pₖ²`, every `dₖ > 0`.
    pub fn squares(&self) -> &[(Q, Poly)] {
        &self.squares
    }

    /// Number of squares.
    pub fn rank(&self) -> usize {
        self.squares.len()
    }

    /// The monomial `m_i` as an expression.
    fn monomial(&self, exps: &[u32]) -> Ex {
        let ctx = self.goal.context();
        let mut acc = ctx.one();
        for (v, &e) in self.goal.gens().iter().zip(exps) {
            if e > 0 {
                acc *= v.powi(i64::from(e));
            }
        }
        acc
    }

    /// The identity `goal = Σ dₖ pₖ²` as an [`Equation`] (squares unexpanded).
    pub fn identity(&self) -> Equation {
        let ctx = self.goal.context();
        let mut rhs = ctx.zero();
        for (d, p) in &self.squares {
            rhs += ctx.from_ratio(d.clone()) * p.to_ex().powi(2);
        }
        Equation::new(self.goal.to_ex(), rhs)
    }

    /// Recompute `Σ dₖ pₖ²` exactly and compare with the goal; also checks
    /// `dₖ > 0`, `goal = mᵀ Q m` and `Q ⪰ 0`.
    pub fn verify(&self) -> bool {
        if self.squares.iter().any(|(d, _)| !d.is_positive()) {
            return false;
        }
        if !self.gram.is_positive_semidefinite() {
            return false;
        }
        let ctx = self.goal.context();
        let gens: Vec<&Ex> = self.goal.gens().iter().collect();
        let Ok(mut acc) = Poly::zero(&ctx, &gens) else {
            return false;
        };
        for (d, p) in &self.squares {
            let Ok(sq) = p.mul(p) else {
                return false;
            };
            let Ok(scaled) = sq.scale(&ctx.from_ratio(d.clone())) else {
                return false;
            };
            let Ok(sum) = acc.add(&scaled) else {
                return false;
            };
            acc = sum;
        }
        if !acc.equals(&self.goal) {
            return false;
        }
        // mᵀ Q m = goal as well.
        let n = self.basis.len();
        let Ok(mut gram_sum) = Poly::zero(&ctx, &gens) else {
            return false;
        };
        for i in 0..n {
            for j in 0..n {
                let q = &self.gram[(i, j)];
                if q.is_zero() {
                    continue;
                }
                let mono = self.monomial(&self.basis[i]) * self.monomial(&self.basis[j]);
                let Some(mp) = Poly::new(&(ctx.from_ratio(q.clone()) * mono), &gens) else {
                    return false;
                };
                let Ok(sum) = gram_sum.add(&mp) else {
                    return false;
                };
                gram_sum = sum;
            }
        }
        gram_sum.equals(&self.goal)
    }

    /// The Lean hint terms `sq_nonneg (pₖ)`, one per square, for a caller's
    /// `nlinarith [...]` / `positivity` skeleton.
    ///
    /// # Errors
    ///
    /// [`SymplexError::NotImplemented`] if a square cannot be rendered.
    pub fn lean_hints(&self, opts: &LeanOpts) -> Result<Vec<String>, SymplexError> {
        self.squares
            .iter()
            .map(|(_, p)| Ok(format!("sq_nonneg ({})", p.to_ex().to_lean_with(opts)?)))
            .collect()
    }

    /// A Lean 4 / Mathlib theorem `0 ≤ goal` over the reals.
    ///
    /// Shape:
    /// ```text
    /// theorem name (x y : ℝ) : 0 ≤ goal := by
    ///   have h : goal = d₁ * (p₁) ^ 2 + d₂ * (p₂) ^ 2 := by ring
    ///   rw [h]; positivity
    /// ```
    /// `ring` checks the exact identity, `positivity` closes the sum of
    /// non-negative terms — no search in either step.
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
        let vars: Vec<String> = self
            .goal
            .gens()
            .iter()
            .map(|g| lean_ident(&g.to_string()))
            .collect();
        let goal = self.goal.to_ex().to_lean_with(opts)?;
        let mut terms: Vec<String> = Vec::with_capacity(self.squares.len());
        for (d, p) in &self.squares {
            let inner = p.to_ex().to_lean_with(opts)?;
            let sq = format!("{} ^ 2", paren_unless_atomic(&inner));
            if d.is_one() {
                terms.push(sq);
            } else {
                let coef = self
                    .goal
                    .context()
                    .from_ratio(d.clone())
                    .to_lean_with(opts)?;
                terms.push(format!("{coef} * {sq}"));
            }
        }
        let rhs = if terms.is_empty() {
            "0".to_string()
        } else {
            terms.join(" + ")
        };
        let text = format!(
            "theorem {} ({} : {}) : 0 ≤ {goal} := by\n  have h : {goal} = {rhs} := by ring\n  rw [h]\n  positivity\n",
            lean_ident(theorem_name),
            vars.join(" "),
            opts.real_type
        );
        Ok(wrap_lean(&text, MATHLIB_LINE_WIDTH))
    }

    /// Plain data for serialisation (the Gram matrix; the squares are
    /// recomputed on load).
    pub fn to_data(&self) -> SosCertificateData {
        SosCertificateData {
            vars: self.goal.gens().iter().map(Ex::to_tree).collect(),
            goal: self.goal.to_ex().to_tree(),
            basis: self.basis.clone(),
            gram: self.gram.iter().map(q_to_str).collect(),
        }
    }

    /// Rebuild from data in `ctx` and **re-verify**.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] for malformed data or a Gram matrix
    /// that is not PSD or does not reproduce the goal.
    pub fn from_data(ctx: &Context, data: &SosCertificateData) -> Result<Self, SymplexError> {
        let vars: Vec<Ex> = data.vars.iter().map(|t| ctx.from_tree(t)).collect();
        let gens: Vec<&Ex> = vars.iter().collect();
        let goal_ex = ctx.from_tree(&data.goal);
        let goal = Poly::new(&goal_ex, &gens).ok_or_else(|| {
            invalid(format!(
                "goal `{goal_ex}` is not a polynomial in the variables"
            ))
        })?;
        let n = data.basis.len();
        if n == 0 || data.gram.len() != n * n {
            return Err(invalid("Gram matrix size does not match the basis"));
        }
        let entries: Vec<Q> = data
            .gram
            .iter()
            .map(|s| q_from_str(s, OP))
            .collect::<Result<_, _>>()?;
        let gram = QMatrix::from_flat(n, n, entries)?;
        let cert = SosCertificate::from_gram(goal, data.basis.clone(), gram)?;
        if !cert.verify() {
            return Err(invalid("the certificate data does not verify"));
        }
        Ok(cert)
    }

    /// JSON form of [`to_data`](Self::to_data).
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
        let data: SosCertificateData =
            serde_json::from_str(json).map_err(|e| invalid(format!("malformed JSON: {e}")))?;
        Self::from_data(ctx, &data)
    }

    /// Build the squares from an exact PSD Gram matrix via `L·D·Lᵀ`.
    fn from_gram(goal: Poly, basis: Vec<Vec<u32>>, gram: QMatrix) -> Result<Self, SymplexError> {
        let (l, d) = gram
            .ldl_psd()
            .ok_or_else(|| invalid("the Gram matrix is not positive semidefinite"))?;
        let ctx = goal.context();
        let gens_owned: Vec<Ex> = goal.gens().to_vec();
        let gens: Vec<&Ex> = gens_owned.iter().collect();
        let n = basis.len();
        let mut cert = SosCertificate {
            goal,
            basis,
            gram,
            squares: Vec::new(),
        };
        // pₖ = Σᵢ L_{ik} mᵢ  (column k of L).
        let mut squares = Vec::new();
        for (k, dk) in d.iter().enumerate() {
            if dk.is_zero() {
                continue;
            }
            let mut p = ctx.zero();
            for i in 0..n {
                let lik = &l[(i, k)];
                if !lik.is_zero() {
                    p += ctx.from_ratio(lik.clone()) * cert.monomial(&cert.basis[i]);
                }
            }
            let poly = Poly::new(&p, &gens).ok_or_else(|| invalid("internal: square"))?;
            squares.push((dk.clone(), poly));
        }
        cert.squares = squares;
        Ok(cert)
    }
}

/// `text` wrapped in parentheses unless it is a single identifier/number or
/// already a single parenthesised group.
fn paren_unless_atomic(text: &str) -> String {
    let atomic = !text.contains(' ')
        || (text.starts_with('(') && text.ends_with(')') && {
            // The opening paren must close only at the very end.
            let mut depth = 0i32;
            let mut closes_early = false;
            for (i, c) in text.char_indices() {
                match c {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 && i + 1 < text.len() {
                            closes_early = true;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            !closes_early
        });
    if atomic {
        text.to_string()
    } else {
        format!("({text})")
    }
}

impl Certificate for SosCertificate {
    fn goal(&self) -> &Poly {
        SosCertificate::goal(self)
    }
    fn verify(&self) -> bool {
        SosCertificate::verify(self)
    }
    fn to_lean_with(&self, theorem_name: &str, opts: &LeanOpts) -> Result<String, SymplexError> {
        SosCertificate::to_lean_with(self, theorem_name, opts)
    }
    fn to_json(&self) -> Result<String, SymplexError> {
        SosCertificate::to_json(self)
    }
    fn from_json(ctx: &Context, json: &str) -> Result<Self, SymplexError> {
        SosCertificate::from_json(ctx, json)
    }
}

impl fmt::Display for SosCertificate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Equation { lhs, rhs } = self.identity();
        write!(f, "{lhs} = {rhs}")
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Dense f64 linear algebra (symmetric, small)
// ═══════════════════════════════════════════════════════════════════════════

/// Row-major dense matrix helpers on `Vec<f64>`.
mod dense {
    /// `a · b` for `n×n` matrices.
    pub fn mul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
        let mut c = vec![0.0; n * n];
        for i in 0..n {
            for k in 0..n {
                let aik = a[i * n + k];
                if aik == 0.0 {
                    continue;
                }
                for j in 0..n {
                    c[i * n + j] += aik * b[k * n + j];
                }
            }
        }
        c
    }

    /// Cholesky factor `L` (lower) with `a = L Lᵀ`, or `None` if `a` is not
    /// numerically positive definite.
    pub fn cholesky(a: &[f64], n: usize) -> Option<Vec<f64>> {
        let mut l = vec![0.0; n * n];
        for j in 0..n {
            let mut s = a[j * n + j];
            for k in 0..j {
                s -= l[j * n + k] * l[j * n + k];
            }
            if s.is_nan() || s <= 0.0 || !s.is_finite() {
                return None;
            }
            let d = s.sqrt();
            l[j * n + j] = d;
            for i in (j + 1)..n {
                let mut s = a[i * n + j];
                for k in 0..j {
                    s -= l[i * n + k] * l[j * n + k];
                }
                l[i * n + j] = s / d;
            }
        }
        Some(l)
    }

    /// Inverse of a symmetric positive definite matrix from its Cholesky factor.
    pub fn spd_inverse(l: &[f64], n: usize) -> Vec<f64> {
        // Solve L Lᵀ X = I column by column.
        let mut inv = vec![0.0; n * n];
        for c in 0..n {
            let mut y = vec![0.0; n];
            for i in 0..n {
                let mut s = if i == c { 1.0 } else { 0.0 };
                for k in 0..i {
                    s -= l[i * n + k] * y[k];
                }
                y[i] = s / l[i * n + i];
            }
            for i in (0..n).rev() {
                let mut s = y[i];
                for k in (i + 1)..n {
                    s -= l[k * n + i] * y[k];
                }
                y[i] = s / l[i * n + i];
            }
            for i in 0..n {
                inv[i * n + c] = y[i];
            }
        }
        inv
    }

    /// `L⁻¹ a L⁻ᵀ` for a lower-triangular `L` and symmetric `a`.
    pub fn congruence_inverse(l: &[f64], a: &[f64], n: usize) -> Vec<f64> {
        // Solve L Y = A  (column by column), then W = Y L⁻ᵀ, i.e. L Wᵀ = Yᵀ.
        let mut y = vec![0.0; n * n];
        for c in 0..n {
            for i in 0..n {
                let mut s = a[i * n + c];
                for k in 0..i {
                    s -= l[i * n + k] * y[k * n + c];
                }
                y[i * n + c] = s / l[i * n + i];
            }
        }
        let mut w = vec![0.0; n * n];
        for r in 0..n {
            // Row r of W: solve L wᵀ = (row r of Y)ᵀ.
            for i in 0..n {
                let mut s = y[r * n + i];
                for k in 0..i {
                    s -= l[i * n + k] * w[r * n + k];
                }
                w[r * n + i] = s / l[i * n + i];
            }
        }
        // Symmetrise against rounding.
        for i in 0..n {
            for j in (i + 1)..n {
                let v = 0.5 * (w[i * n + j] + w[j * n + i]);
                w[i * n + j] = v;
                w[j * n + i] = v;
            }
        }
        w
    }

    /// Solve the symmetric positive definite system `m x = rhs` (Cholesky
    /// with a tiny diagonal regularisation on failure).
    pub fn solve_spd(m: &[f64], rhs: &[f64], n: usize) -> Option<Vec<f64>> {
        let mut reg = 0.0;
        for _ in 0..6 {
            let mut a = m.to_vec();
            if reg > 0.0 {
                for i in 0..n {
                    a[i * n + i] += reg;
                }
            }
            if let Some(l) = cholesky(&a, n) {
                let mut y = vec![0.0; n];
                for i in 0..n {
                    let mut s = rhs[i];
                    for k in 0..i {
                        s -= l[i * n + k] * y[k];
                    }
                    y[i] = s / l[i * n + i];
                }
                for i in (0..n).rev() {
                    let mut s = y[i];
                    for k in (i + 1)..n {
                        s -= l[k * n + i] * y[k];
                    }
                    y[i] = s / l[i * n + i];
                }
                return Some(y);
            }
            let scale = (0..n)
                .map(|i| m[i * n + i].abs())
                .fold(0.0, f64::max)
                .max(1e-300);
            reg = if reg == 0.0 {
                scale * 1e-12
            } else {
                reg * 100.0
            };
        }
        None
    }

    /// Eigen-decomposition of a symmetric `n×n` matrix (see [`sym_eigen`]).
    pub struct SymEigen {
        /// The `n` eigenvalues, in the order the Jacobi sweeps leave them
        /// on the diagonal (not sorted).
        pub values: Vec<f64>,
        /// The eigenvectors as the **columns** of a row-major `n×n` matrix:
        /// component `i` of the eigenvector for `values[c]` is
        /// `vectors[i * n + c]`.
        pub vectors: Vec<f64>,
    }

    /// Eigen-decomposition of a symmetric matrix by cyclic Jacobi rotations.
    pub fn sym_eigen(a: &[f64], n: usize) -> SymEigen {
        let mut m = a.to_vec();
        let mut v = vec![0.0; n * n];
        for i in 0..n {
            v[i * n + i] = 1.0;
        }
        for _sweep in 0..100 {
            let mut off = 0.0;
            for i in 0..n {
                for j in (i + 1)..n {
                    off += m[i * n + j] * m[i * n + j];
                }
            }
            if off < 1e-30 {
                break;
            }
            for p in 0..n {
                for q in (p + 1)..n {
                    let apq = m[p * n + q];
                    if apq.abs() < 1e-300 {
                        continue;
                    }
                    let app = m[p * n + p];
                    let aqq = m[q * n + q];
                    let theta = (aqq - app) / (2.0 * apq);
                    let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                    let t = if theta == 0.0 { 1.0 } else { t };
                    let c = 1.0 / (t * t + 1.0).sqrt();
                    let s = t * c;
                    for k in 0..n {
                        let mkp = m[k * n + p];
                        let mkq = m[k * n + q];
                        m[k * n + p] = c * mkp - s * mkq;
                        m[k * n + q] = s * mkp + c * mkq;
                    }
                    for k in 0..n {
                        let mpk = m[p * n + k];
                        let mqk = m[q * n + k];
                        m[p * n + k] = c * mpk - s * mqk;
                        m[q * n + k] = s * mpk + c * mqk;
                    }
                    for k in 0..n {
                        let vkp = v[k * n + p];
                        let vkq = v[k * n + q];
                        v[k * n + p] = c * vkp - s * vkq;
                        v[k * n + q] = s * vkp + c * vkq;
                    }
                }
            }
        }
        let eig: Vec<f64> = (0..n).map(|i| m[i * n + i]).collect();
        SymEigen {
            values: eig,
            vectors: v,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// The SDP: constraints, interior point, facial reduction, rounding
// ═══════════════════════════════════════════════════════════════════════════

/// Index of the upper-triangular entry `(i, j)`, `i ≤ j`, in an `n×n`
/// symmetric matrix stored as its upper triangle row by row.
fn tri_index(i: usize, j: usize, n: usize) -> usize {
    debug_assert!(i <= j);
    i * n - i * (i + 1) / 2 + j
}

/// A linear functional on symmetric `n×n` matrices, `Σ_{i≤j} c_{ij} Q_{ij}`,
/// with a target value.
#[derive(Clone, Debug)]
struct Constraint {
    coeffs: Vec<Q>,
    target: Q,
}

impl Constraint {
    /// The symmetric matrix `A` with `⟨A, Q⟩ = Σ c_{ij} Q_{ij}` (halving the
    /// off-diagonal coefficients), as `f64`.
    fn matrix_f64(&self, n: usize) -> Vec<f64> {
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in i..n {
                let c = &self.coeffs[tri_index(i, j, n)];
                if c.is_zero() {
                    continue;
                }
                let v = q_to_f64(c);
                if i == j {
                    a[i * n + i] = v;
                } else {
                    a[i * n + j] = v / 2.0;
                    a[j * n + i] = v / 2.0;
                }
            }
        }
        a
    }
}

fn q_to_f64(q: &Q) -> f64 {
    q.to_f64().unwrap_or_else(|| {
        let n = q.numer().to_f64().unwrap_or(0.0);
        let d = q.denom().to_f64().unwrap_or(1.0);
        n / d
    })
}

/// All exponent vectors in `nvars` variables with total degree ≤ `deg`,
/// in graded lexicographic order.
fn monomials_up_to(nvars: usize, deg: u32) -> Vec<Vec<u32>> {
    let mut out = Vec::new();
    fn rec(slot: usize, remaining: u32, cur: &mut Vec<u32>, out: &mut Vec<Vec<u32>>) {
        if slot == cur.len() {
            out.push(cur.clone());
            return;
        }
        for e in 0..=remaining {
            cur[slot] = e;
            rec(slot + 1, remaining - e, cur, out);
        }
        cur[slot] = 0;
    }
    let mut cur = vec![0u32; nvars];
    rec(0, deg, &mut cur, &mut out);
    out.sort_by_key(|m| (m.iter().sum::<u32>(), m.clone()));
    out
}

/// Monomials of degree `≤ half` whose doubled exponent vector lies in the
/// Newton polytope of `goal` (the convex hull of its exponent vectors) — the
/// only monomials that can appear in a sum-of-squares decomposition.  Each
/// candidate is one small exact LP: `2e = Σ λᵢ vᵢ`, `Σ λᵢ = 1`, `λ ≥ 0`
/// over the goal's support `vᵢ`.  Falls back to the plain degree bound if
/// an LP fails (never drops a monomial on an error); a pass of the
/// caller's `deadline` is reported as `Err(DeadlinePassed)` so the caller
/// can answer `Unknown` without building the SDP.
fn newton_pruned_basis(
    goal: &Poly,
    nvars: usize,
    half: u32,
    deadline: Option<Instant>,
) -> Result<Vec<Vec<u32>>, DeadlinePassed> {
    let all = monomials_up_to(nvars, half);
    let support: Vec<Vec<u32>> = goal.terms_iter().map(|(m, _)| m.to_vec()).collect();
    if support.len() <= 1 {
        return Ok(all);
    }
    let in_hull = |e: &[u32]| -> Result<Option<bool>, DeadlinePassed> {
        let k = support.len();
        // Variables λ₀..λ_{k−1} ≥ 0 (default bounds); rows: one per
        // coordinate plus Σλ = 1.
        let mut lp = LpProblem::minimize(vec![Q::zero(); k]);
        for (c, &target) in e.iter().enumerate() {
            let row: Vec<Q> = support
                .iter()
                .map(|v| Q::from_integer(BigInt::from(v[c])))
                .collect();
            lp = lp.eq(row, Q::from_integer(BigInt::from(2 * target)));
        }
        lp = lp.eq(vec![Q::one(); k], Q::one());
        if let Some(at) = deadline {
            lp = lp.with_budget(Budget::deadline(at));
        }
        Ok(match lp.solve().ok().map(|s| s.status) {
            Some(LpStatus::Optimal) => Some(true),
            Some(LpStatus::BudgetExhausted) => return Err(DeadlinePassed),
            Some(_) => Some(false),
            None => None,
        })
    };
    let mut kept = Vec::with_capacity(all.len());
    for m in &all {
        match in_hull(m)? {
            Some(true) => kept.push(m.clone()),
            Some(false) => {}
            None => return Ok(all),
        }
    }
    Ok(kept)
}

/// The SDP `A(Q) = b, Q ⪰ 0` for `goal = mᵀ Q m` over the monomial basis.
struct SosProblem {
    n: usize,
    constraints: Vec<Constraint>,
}

impl SosProblem {
    fn new(goal: &Poly, basis: &[Vec<u32>]) -> Result<Self, SymplexError> {
        let n = basis.len();
        // Group the entries (i ≤ j) by the monomial e_i + e_j.
        let mut groups: std::collections::BTreeMap<Vec<u32>, Vec<usize>> =
            std::collections::BTreeMap::new();
        for i in 0..n {
            for j in i..n {
                let m: Vec<u32> = basis[i].iter().zip(&basis[j]).map(|(a, b)| a + b).collect();
                groups.entry(m).or_default().push(tri_index(i, j, n));
            }
        }
        // Every monomial of the goal must be reachable.
        for (m, _) in goal.terms_iter() {
            if !groups.contains_key(m) {
                return Err(invalid(format!(
                    "the goal monomial with exponents {m:?} is not a product of two basis monomials"
                )));
            }
        }
        let tri = n * (n + 1) / 2;
        let mut constraints = Vec::with_capacity(groups.len());
        for (m, idx) in groups {
            let mut coeffs = vec![Q::zero(); tri];
            for &k in &idx {
                // Entry (i, j) with i < j appears twice in mᵀ Q m.
                let (i, j) = tri_pair(k, n);
                coeffs[k] = if i == j {
                    Q::one()
                } else {
                    Q::from_integer(BigInt::from(2))
                };
            }
            let target = goal
                .coeff_monomial(&m)?
                .as_rational()
                .ok_or_else(|| invalid("goal must have rational coefficients"))?;
            constraints.push(Constraint { coeffs, target });
        }
        Ok(SosProblem { n, constraints })
    }

    /// Restrict to the face `Q = B Q' Bᵀ` (`B` is `n × r`): the constraints
    /// become `⟨Bᵀ A B, Q'⟩ = b`.
    fn restrict(&self, b: &QMatrix) -> Result<SosProblem, SymplexError> {
        let n = self.n;
        let r = b.ncols();
        let tri_r = r * (r + 1) / 2;
        let mut constraints = Vec::with_capacity(self.constraints.len());
        for c in &self.constraints {
            // A as an exact symmetric matrix.
            let a = QMatrix::from_fn(n, n, |i, j| {
                let (lo, hi) = if i <= j { (i, j) } else { (j, i) };
                let v = &c.coeffs[tri_index(lo, hi, n)];
                if i == j {
                    v.clone()
                } else {
                    v / Q::from_integer(BigInt::from(2))
                }
            });
            let bab = &(&b.transpose() * &a) * b; // r × r symmetric
            let mut coeffs = vec![Q::zero(); tri_r];
            for i in 0..r {
                for j in i..r {
                    let v = &bab[(i, j)];
                    coeffs[tri_index(i, j, r)] = if i == j {
                        v.clone()
                    } else {
                        v * Q::from_integer(BigInt::from(2))
                    };
                }
            }
            constraints.push(Constraint {
                coeffs,
                target: c.target.clone(),
            });
        }
        let mut p = SosProblem { n: r, constraints };
        p.remove_dependent_rows()?;
        Ok(p)
    }

    /// Drop linearly dependent constraint rows (exactly), failing if the
    /// system is inconsistent.
    fn remove_dependent_rows(&mut self) -> Result<(), SymplexError> {
        let m = self.constraints.len();
        let w = self.n * (self.n + 1) / 2;
        if m == 0 {
            return Ok(());
        }
        let aug = QMatrix::from_fn(m, w + 1, |i, j| {
            if j < w {
                self.constraints[i].coeffs[j].clone()
            } else {
                self.constraints[i].target.clone()
            }
        });
        let (r, pivots) = aug.rref_limited(w);
        let rank = pivots.len();
        if (rank..m).any(|i| !r[(i, w)].is_zero()) {
            return Err(SymplexError::ComputationFailed {
                operation: OP,
                reason: "the reduced constraint system is inconsistent (the numerical kernel was not an exact face)".into(),
            });
        }
        // Keep the RREF rows: an equivalent, independent constraint set.
        self.constraints = (0..rank)
            .map(|i| Constraint {
                coeffs: (0..w).map(|j| r[(i, j)].clone()).collect(),
                target: r[(i, w)].clone(),
            })
            .collect();
        Ok(())
    }

    /// Primal–dual interior-point method (HKM direction, Mehrotra
    /// predictor–corrector) for the feasibility SDP with zero objective.
    /// Returns the primal iterate `X` (row-major), or `None` if the method
    /// did not converge (infeasible or ill-posed); `Err` if `deadline`
    /// passed between two iterations.
    fn solve_numeric(
        &self,
        max_iterations: usize,
        deadline: Option<Instant>,
    ) -> Result<Option<Vec<f64>>, DeadlinePassed> {
        let n = self.n;
        let m = self.constraints.len();
        let mats: Vec<Vec<f64>> = self.constraints.iter().map(|c| c.matrix_f64(n)).collect();
        let b: Vec<f64> = self
            .constraints
            .iter()
            .map(|c| q_to_f64(&c.target))
            .collect();
        let bnorm = b.iter().map(|v| v * v).sum::<f64>().sqrt().max(1.0);
        let inner = |a: &[f64], c: &[f64]| -> f64 { a.iter().zip(c).map(|(x, y)| x * y).sum() };
        let scale = b.iter().map(|v| v.abs()).fold(0.0, f64::max).max(1.0);

        let mut x = vec![0.0; n * n];
        let mut z = vec![0.0; n * n];
        for i in 0..n {
            x[i * n + i] = scale;
            z[i * n + i] = 1.0;
        }
        let mut y = vec![0.0; m];
        let mut best: Option<(f64, Vec<f64>)> = None;

        // Largest step in (0, 1] keeping `base + α·dir` positive definite:
        // with base = L Lᵀ, α_max = 1 / max(0, −λ_min(L⁻¹ dir L⁻ᵀ)),
        // refined by a Cholesky check (and backtracking if rounding bites).
        let step = |base: &[f64], dir: &[f64]| -> f64 {
            let mut alpha = match dense::cholesky(base, n) {
                Some(l) => {
                    let w = dense::congruence_inverse(&l, dir, n);
                    let eig = dense::sym_eigen(&w, n).values;
                    let lmin = eig.iter().cloned().fold(f64::INFINITY, f64::min);
                    if lmin >= 0.0 {
                        1.0
                    } else {
                        (-1.0 / lmin).min(1.0)
                    }
                }
                None => 1.0,
            };
            for _ in 0..60 {
                let trial: Vec<f64> = base.iter().zip(dir).map(|(b, d)| b + alpha * d).collect();
                if dense::cholesky(&trial, n).is_some() {
                    return alpha;
                }
                alpha *= 0.9;
            }
            0.0
        };
        let mut stall = 0usize;

        for _it in 0..max_iterations {
            check_deadline(deadline)?;
            // Residuals.
            let rp: Vec<f64> = (0..m).map(|k| b[k] - inner(&mats[k], &x)).collect();
            let mut rd = vec![0.0; n * n]; // −(Aᵀy + Z)  (C = 0)
            for k in 0..m {
                for (r, a) in rd.iter_mut().zip(&mats[k]) {
                    *r -= y[k] * a;
                }
            }
            for (r, zz) in rd.iter_mut().zip(&z) {
                *r -= zz;
            }
            let mu = inner(&x, &z) / n as f64;
            let rp_norm = rp.iter().map(|v| v * v).sum::<f64>().sqrt();
            let rd_norm = rd.iter().map(|v| v * v).sum::<f64>().sqrt();
            let merit = rp_norm / bnorm + rd_norm + mu;
            if best.as_ref().is_none_or(|(bm, _)| merit < *bm) {
                best = Some((merit, x.clone()));
                stall = 0;
            } else {
                stall += 1;
                if stall >= 8 {
                    break;
                }
            }
            // With a zero objective the central path does not move once the
            // iterate is feasible (it is the analytic centre for every μ), so
            // μ only has to be small enough to expose the kernel of a
            // rank-deficient face; pushing it further only lets Z⁻¹ blow up.
            let xscale = (0..n).map(|i| x[i * n + i]).fold(0.0, f64::max).max(1.0);
            if rp_norm <= 1e-9 * bnorm && rd_norm <= 1e-9 && mu <= 1e-9 * xscale {
                return Ok(Some(x));
            }

            // Schur complement M_kl = ⟨A_k, X A_l Z⁻¹⟩ (shared by both solves).
            // A numerical breakdown (Z no longer positive definite in floating
            // point, a singular Schur complement) ends the iteration; the best
            // iterate so far is returned below.
            let Some(lz) = dense::cholesky(&z, n) else {
                break;
            };
            let zinv = dense::spd_inverse(&lz, n);
            let t: Vec<Vec<f64>> = mats
                .iter()
                .map(|a| dense::mul(&dense::mul(&x, a, n), &zinv, n))
                .collect();
            let mut mm = vec![0.0; m * m];
            for k in 0..m {
                for l in 0..m {
                    mm[k * m + l] = inner(&mats[k], &t[l]);
                }
            }
            for k in 0..m {
                for l in (k + 1)..m {
                    let v = 0.5 * (mm[k * m + l] + mm[l * m + k]);
                    mm[k * m + l] = v;
                    mm[l * m + k] = v;
                }
            }
            let xz = dense::mul(&x, &z, n);
            let xrz = dense::mul(&dense::mul(&x, &rd, n), &zinv, n);

            // One Newton solve for a given right-hand side of the
            // complementarity equation `X ΔZ + ΔX Z = R`.
            let solve_dir = |r_c: &[f64]| -> Option<(Vec<f64>, Vec<f64>, Vec<f64>)> {
                // A_k(ΔX) = rp_k with ΔX = (R − X ΔZ) Z⁻¹, ΔZ = rd − AᵀΔy:
                //   Σ_l M_kl Δy_l = rp_k − ⟨A_k, R Z⁻¹⟩ + ⟨A_k, X rd Z⁻¹⟩.
                let rz = dense::mul(r_c, &zinv, n);
                let mut rhs = vec![0.0; m];
                for k in 0..m {
                    rhs[k] = rp[k] - inner(&mats[k], &rz) + inner(&mats[k], &xrz);
                }
                let dy = dense::solve_spd(&mm, &rhs, m)?;
                let mut dz = rd.clone();
                for k in 0..m {
                    for (d, a) in dz.iter_mut().zip(&mats[k]) {
                        *d -= dy[k] * a;
                    }
                }
                let xdz = dense::mul(&x, &dz, n);
                let mut w = vec![0.0; n * n];
                for i in 0..n * n {
                    w[i] = r_c[i] - xdz[i];
                }
                let dx_raw = dense::mul(&w, &zinv, n);
                let mut dx = vec![0.0; n * n];
                for i in 0..n {
                    for j in 0..n {
                        dx[i * n + j] = 0.5 * (dx_raw[i * n + j] + dx_raw[j * n + i]);
                    }
                }
                Some((dx, dy, dz))
            };

            // Predictor: affine-scaling direction (σ = 0) sets the centering
            // parameter; the corrector keeps the iterate well inside the cone
            // (σ ≥ 0.1) so that on a rank-deficient face the small
            // eigenvalues shrink uniformly and the kernel is read cleanly.
            let r_aff: Vec<f64> = xz.iter().map(|v| -v).collect();
            let Some((dx_a, _dy_a, dz_a)) = solve_dir(&r_aff) else {
                break;
            };
            let ap_a = step(&x, &dx_a);
            let ad_a = step(&z, &dz_a);
            let x_a: Vec<f64> = x.iter().zip(&dx_a).map(|(a, d)| a + ap_a * d).collect();
            let z_a: Vec<f64> = z.iter().zip(&dz_a).map(|(a, d)| a + ad_a * d).collect();
            let mu_aff = inner(&x_a, &z_a) / n as f64;
            let sigma = (mu_aff / mu).clamp(0.0, 1.0).powi(3).clamp(0.1, 0.5);

            // Corrector: σμ I − XZ − ΔX_aff ΔZ_aff.
            let cross = dense::mul(&dx_a, &dz_a, n);
            let mut r_c = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    let id = if i == j { sigma * mu } else { 0.0 };
                    r_c[i * n + j] = id - xz[i * n + j] - cross[i * n + j];
                }
            }
            let Some((dx, dy, dz)) = solve_dir(&r_c) else {
                break;
            };
            let ap = (0.95 * step(&x, &dx)).min(1.0);
            let ad = (0.95 * step(&z, &dz)).min(1.0);
            if ap <= 0.0 && ad <= 0.0 {
                break;
            }
            for (xx, d) in x.iter_mut().zip(&dx) {
                *xx += ap * d;
            }
            for (yy, d) in y.iter_mut().zip(&dy) {
                *yy += ad * d;
            }
            for (zz, d) in z.iter_mut().zip(&dz) {
                *zz += ad * d;
            }
        }
        // Not fully converged: return the best iterate if it is nearly
        // feasible (the exact rounding step is the real test).
        Ok(best.and_then(|(merit, xb)| if merit < 1e-5 { Some(xb) } else { None }))
    }

    /// Round `x` to `10^{-digits}` and project exactly onto `A(Q) = b`
    /// (weighted least-norm correction, weights 1 on the diagonal and 2 off
    /// it so the correction is the Frobenius-nearest one).  Returns the
    /// exact symmetric matrix.
    fn round_and_project(&self, x: &[f64], digits: u32) -> Option<QMatrix> {
        let n = self.n;
        let w = n * (n + 1) / 2;
        let denom = BigInt::from(10u32).pow(digits);
        let mut q = vec![Q::zero(); w];
        for i in 0..n {
            for j in i..n {
                let v = 0.5 * (x[i * n + j] + x[j * n + i]);
                let scaled = (v * q_to_f64(&Q::from_integer(denom.clone()))).round();
                if !scaled.is_finite() {
                    return None;
                }
                let num = BigInt::from(scaled as i128);
                q[tri_index(i, j, n)] = Q::new(num, denom.clone());
            }
        }
        // Residual r = b − A q.
        let m = self.constraints.len();
        let residual: Vec<Q> = self
            .constraints
            .iter()
            .map(|c| {
                let mut s = c.target.clone();
                for (cc, qq) in c.coeffs.iter().zip(&q) {
                    if !cc.is_zero() && !qq.is_zero() {
                        s -= cc * qq;
                    }
                }
                s
            })
            .collect();
        if residual.iter().all(Zero::is_zero) {
            return Some(self.expand(&q));
        }
        // Δq = W⁻¹ Aᵀ ν with (A W⁻¹ Aᵀ) ν = r,  W = diag(1 on diag, 2 off).
        let winv = |k: usize| -> Q {
            let (i, j) = tri_pair(k, n);
            if i == j {
                Q::one()
            } else {
                Q::new(BigInt::one(), BigInt::from(2))
            }
        };
        let gram = QMatrix::from_fn(m, m, |a, b| {
            let (ca, cb) = (&self.constraints[a].coeffs, &self.constraints[b].coeffs);
            let mut s = Q::zero();
            for k in 0..w {
                if !ca[k].is_zero() && !cb[k].is_zero() {
                    s += &ca[k] * &cb[k] * winv(k);
                }
            }
            s
        });
        let rhs = QMatrix::from_fn(m, 1, |i, _| residual[i].clone());
        let nu = gram.solve(&rhs).ok()?;
        for (k, qk) in q.iter_mut().enumerate() {
            let mut delta = Q::zero();
            for a in 0..m {
                let c = &self.constraints[a].coeffs[k];
                if !c.is_zero() {
                    delta += c * &nu[(a, 0)];
                }
            }
            if !delta.is_zero() {
                *qk += delta * winv(k);
            }
        }
        Some(self.expand(&q))
    }

    /// Upper-triangle vector → full symmetric `QMatrix`.
    fn expand(&self, q: &[Q]) -> QMatrix {
        let n = self.n;
        QMatrix::from_fn(n, n, |i, j| {
            let (lo, hi) = if i <= j { (i, j) } else { (j, i) };
            q[tri_index(lo, hi, n)].clone()
        })
    }
}

/// Inverse of [`tri_index`].
fn tri_pair(k: usize, n: usize) -> (usize, usize) {
    let mut i = 0;
    let mut start = 0;
    while start + (n - i) <= k {
        start += n - i;
        i += 1;
    }
    (i, i + (k - start))
}

/// Rationalise `v` with a small denominator (continued fractions, denominator
/// ≤ `max_den`, error ≤ `tol`), or `None`.
fn rationalize(v: f64, max_den: u64, tol: f64) -> Option<Q> {
    if !v.is_finite() {
        return None;
    }
    let (mut h0, mut h1) = (0i128, 1i128);
    let (mut k0, mut k1) = (1i128, 0i128);
    let mut x = v;
    for _ in 0..40 {
        let a = x.floor();
        let ai = a as i128;
        let h2 = ai * h1 + h0;
        let k2 = ai * k1 + k0;
        if k2 == 0 || k2.unsigned_abs() > u128::from(max_den) {
            break;
        }
        h0 = h1;
        h1 = h2;
        k0 = k1;
        k1 = k2;
        let approx = h1 as f64 / k1 as f64;
        if (approx - v).abs() <= tol {
            return Some(Q::new(BigInt::from(h1), BigInt::from(k1)));
        }
        let frac = x - a;
        if frac.abs() < 1e-15 {
            break;
        }
        x = 1.0 / frac;
    }
    None
}

/// Numerical kernel of a symmetric PSD `x` (eigenvalues below `1e-7 · λ_max`)
/// as orthonormal rows, or `None` if there is none (or it is everything).
fn numeric_kernel(x: &[f64], n: usize) -> Option<(Vec<Vec<f64>>, f64)> {
    let dense::SymEigen {
        values: eig,
        vectors: vecs,
    } = dense::sym_eigen(x, n);
    let lmax = eig.iter().cloned().fold(0.0, f64::max).max(1e-300);
    let kernel: Vec<usize> = (0..n).filter(|&i| eig[i] < 1e-7 * lmax).collect();
    if kernel.is_empty() || kernel.len() == n {
        return None;
    }
    // Accuracy of the kernel eigenvectors: (largest kernel eigenvalue) /
    // (gap to the smallest range eigenvalue).
    let kmax = kernel.iter().map(|&i| eig[i].abs()).fold(0.0, f64::max);
    let gap = (0..n)
        .filter(|i| !kernel.contains(i))
        .map(|i| eig[i])
        .fold(f64::INFINITY, f64::min)
        .max(1e-300);
    let accuracy = (kmax / gap).max(1e-12);
    Some((
        kernel
            .iter()
            .map(|&c| (0..n).map(|i| vecs[i * n + c]).collect())
            .collect(),
        accuracy,
    ))
}

/// A rational basis of the kernel spanned by `rows`, when the kernel is a
/// rational subspace: RREF, then rounding to small rationals.
fn rationalize_kernel(rows: &[Vec<f64>], n: usize) -> Option<QMatrix> {
    let r = rows.len();
    let mut k: Vec<f64> = rows.iter().flatten().copied().collect();
    let mut pr = 0;
    for col in 0..n {
        if pr >= r {
            break;
        }
        let Some(p) = (pr..r).max_by(|&a, &b| {
            k[a * n + col]
                .abs()
                .partial_cmp(&k[b * n + col].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        }) else {
            break;
        };
        if k[p * n + col].abs() < 1e-8 {
            continue;
        }
        for c in 0..n {
            k.swap(pr * n + c, p * n + c);
        }
        let piv = k[pr * n + col];
        for c in 0..n {
            k[pr * n + c] /= piv;
        }
        for row in 0..r {
            if row == pr {
                continue;
            }
            let f = k[row * n + col];
            if f != 0.0 {
                for c in 0..n {
                    k[row * n + c] -= f * k[pr * n + c];
                }
            }
        }
        pr += 1;
    }
    let mut out: Vec<Vec<Q>> = Vec::new();
    for row in 0..pr {
        let mut exact = Vec::with_capacity(n);
        for c in 0..n {
            let v = k[row * n + c];
            let v = if v.abs() < 1e-9 { 0.0 } else { v };
            exact.push(rationalize(v, 1000, 1e-9)?);
        }
        // The rationalised row must lie in the numerical kernel to high
        // accuracy (its projection onto the orthonormal kernel rows must
        // reproduce it); this rejects coincidental rational approximations
        // of an irrational kernel.
        let vf: Vec<f64> = exact.iter().map(q_to_f64).collect();
        if !in_span(&vf, rows, 1e-7) {
            return None;
        }
        out.push(exact);
    }
    if out.is_empty() {
        return None;
    }
    QMatrix::new(out).ok()
}

/// Is `v` within relative distance `tol` of the span of the orthonormal
/// `rows`?
fn in_span(v: &[f64], rows: &[Vec<f64>], tol: f64) -> bool {
    let norm = v.iter().map(|a| a * a).sum::<f64>().sqrt();
    if norm == 0.0 {
        return true;
    }
    let mut residual = v.to_vec();
    for r in rows {
        let c: f64 = r.iter().zip(v).map(|(a, b)| a * b).sum();
        for (res, ri) in residual.iter_mut().zip(r) {
            *res -= c * ri;
        }
    }
    residual.iter().map(|a| a * a).sum::<f64>().sqrt() <= tol * norm
}

/// LLL reduction (`δ = 3/4`) of the rows of `b` (integer vectors of equal
/// length): exact integer basis operations with a floating-point
/// Gram–Schmidt recomputed at every step.  The lattices here are small
/// (dimension ≤ 60, entries ≲ 10⁹), well inside `f64`'s exact range for
/// the size-reduction quotients; relations found are re-checked exactly
/// downstream.
fn lll_reduce(mut b: Vec<Vec<BigInt>>) -> Vec<Vec<BigInt>> {
    let n = b.len();
    if n <= 1 {
        return b;
    }
    let dim = b[0].len();
    let to_f = |u: &[BigInt]| -> Vec<f64> { u.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect() };
    let dot = |u: &[f64], v: &[f64]| -> f64 { u.iter().zip(v).map(|(a, c)| a * c).sum() };
    // Gram–Schmidt: (μ, ‖b*ᵢ‖²).
    let gram_schmidt = |b: &[Vec<BigInt>]| -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut bstar: Vec<Vec<f64>> = Vec::with_capacity(n);
        let mut norms: Vec<f64> = Vec::with_capacity(n);
        let mut mu: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
        for i in 0..n {
            let bi = to_f(&b[i]);
            let mut v = bi.clone();
            for j in 0..i {
                let m = if norms[j] <= 0.0 {
                    0.0
                } else {
                    dot(&bi, &bstar[j]) / norms[j]
                };
                for (vk, bj) in v.iter_mut().zip(&bstar[j]) {
                    *vk -= m * bj;
                }
                mu[i][j] = m;
            }
            norms.push(dot(&v, &v));
            bstar.push(v);
        }
        (mu, norms)
    };
    let mut k = 1usize;
    let mut steps = 0usize;
    while k < n && steps < 8_000 {
        steps += 1;
        let (mu, _) = gram_schmidt(&b);
        let mut mu_k = mu[k].clone();
        for j in (0..k).rev() {
            if mu_k[j].abs() > 0.5 {
                let qf = mu_k[j].round();
                if !qf.is_finite() {
                    return b;
                }
                let q = BigInt::from(qf as i128);
                let bj = b[j].clone();
                for (bk, v) in b[k].iter_mut().zip(&bj) {
                    *bk -= &q * v;
                }
                mu_k[j] -= qf;
                for i in 0..j {
                    mu_k[i] -= qf * mu[j][i];
                }
            }
        }
        let (mu, norms) = gram_schmidt(&b);
        let lhs = norms[k];
        let rhs = (0.75 - mu[k][k - 1] * mu[k][k - 1]) * norms[k - 1];
        if lhs >= rhs || norms[k - 1] <= 0.0 {
            k += 1;
        } else {
            b.swap(k, k - 1);
            k = k.saturating_sub(1).max(1);
        }
    }
    let _ = dim;
    b
}

/// Integer relations of the numerical kernel: short integer vectors `ℓ`
/// with `K ℓ ≈ 0`, found by LLL on the lattice `[I | N·Kᵀ]`, sorted by how
/// well they annihilate the (unrounded) kernel.  Their span approximates
/// the largest rational subspace of `K⊥`.
fn integer_relations(rows: &[Vec<f64>], n: usize, accuracy: f64) -> Vec<Vec<BigInt>> {
    // Scale the kernel so that a genuine relation's residual coordinate is
    // O(1) while coincidental near-relations (residual ≫ accuracy) become
    // long lattice vectors that LLL pushes to the back.
    let scale = (0.1 / accuracy).clamp(1e6, 1e12);
    let basis: Vec<Vec<BigInt>> = (0..n)
        .map(|i| {
            let mut v: Vec<BigInt> = (0..n)
                .map(|j| {
                    if i == j {
                        BigInt::one()
                    } else {
                        BigInt::zero()
                    }
                })
                .collect();
            for row in rows {
                v.push(BigInt::from((row[i] * scale).round() as i128));
            }
            v
        })
        .collect();
    let reduced = lll_reduce(basis);
    // A genuine relation is a *short* integer vector that annihilates the
    // unrounded kernel to its numerical accuracy; near-relations with larger
    // residuals are not kept (the exact consistency check downstream weeds
    // out anything that slips through).  Shortest first.
    let mut scored: Vec<(f64, Vec<BigInt>)> = Vec::new();
    for v in reduced {
        let lf: Vec<f64> = v[..n].iter().map(|x| x.to_f64().unwrap_or(0.0)).collect();
        let lnorm = lf.iter().map(|a| a * a).sum::<f64>().sqrt();
        if lnorm == 0.0 || lnorm > 1e4 {
            continue;
        }
        let residual = rows
            .iter()
            .map(|row| row.iter().zip(&lf).map(|(a, b)| a * b).sum::<f64>().abs())
            .fold(0.0, f64::max);
        // Tolerance follows the kernel's conditioning (a converged solve
        // gives ~1e-9; a poorly separated kernel eigenvalue gives more).
        let floor = if accuracy <= 1e-12 { 1e-11 } else { 2e-6 };
        let tol = (10.0 * accuracy * lnorm).clamp(floor, 1e-3);
        if residual <= tol {
            scored.push((lnorm, v[..n].to_vec()));
        }
    }
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().map(|(_, v)| v).collect()
}

/// A polynomial with `f64` coefficients over exponent vectors, for fast
/// evaluation of values, gradients and Hessians.
struct FloatPoly {
    terms: Vec<(Vec<u32>, f64)>,
    nvars: usize,
}

impl FloatPoly {
    fn from_poly(p: &Poly) -> Self {
        FloatPoly {
            terms: p
                .terms_iter()
                .map(|(m, c)| {
                    (
                        m.to_vec(),
                        c.as_rational().map(|q| q_to_f64(&q)).unwrap_or(0.0),
                    )
                })
                .collect(),
            nvars: p.gens().len(),
        }
    }

    fn value(&self, x: &[f64]) -> f64 {
        self.terms
            .iter()
            .map(|(m, c)| {
                c * m
                    .iter()
                    .zip(x)
                    .map(|(&e, v)| v.powi(e as i32))
                    .product::<f64>()
            })
            .sum()
    }

    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        let mut g = vec![0.0; self.nvars];
        for (m, c) in &self.terms {
            for i in 0..self.nvars {
                if m[i] == 0 {
                    continue;
                }
                let mut t = c * m[i] as f64;
                for j in 0..self.nvars {
                    let e = if j == i { m[j] - 1 } else { m[j] };
                    t *= x[j].powi(e as i32);
                }
                g[i] += t;
            }
        }
        g
    }

    fn hessian(&self, x: &[f64]) -> Vec<f64> {
        let n = self.nvars;
        let mut h = vec![0.0; n * n];
        for (m, c) in &self.terms {
            for i in 0..n {
                for j in 0..n {
                    let (mi, mj) = (m[i], m[j]);
                    let factor = if i == j {
                        if mi < 2 {
                            continue;
                        }
                        (mi * (mi - 1)) as f64
                    } else {
                        if mi == 0 || mj == 0 {
                            continue;
                        }
                        (mi * mj) as f64
                    };
                    let mut t = c * factor;
                    for k in 0..n {
                        let mut e = m[k];
                        if k == i {
                            e -= 1;
                        }
                        if k == j {
                            e -= 1;
                        }
                        t *= x[k].powi(e as i32);
                    }
                    h[i * n + j] += t;
                }
            }
        }
        h
    }
}

/// Solve the small dense system `a x = b` by Gaussian elimination with
/// partial pivoting; `None` if singular.
fn solve_dense(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut m: Vec<f64> = a.to_vec();
    let mut r = b.to_vec();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| {
            m[i * n + c]
                .abs()
                .partial_cmp(&m[j * n + c].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        if m[p * n + c].abs() < 1e-300 {
            return None;
        }
        if p != c {
            for k in 0..n {
                m.swap(c * n + k, p * n + k);
            }
            r.swap(c, p);
        }
        for i in (c + 1)..n {
            let f = m[i * n + c] / m[c * n + c];
            if f != 0.0 {
                for k in c..n {
                    m[i * n + k] -= f * m[c * n + k];
                }
                r[i] -= f * r[c];
            }
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sacc = r[i];
        for k in (i + 1)..n {
            sacc -= m[i * n + k] * x[k];
        }
        x[i] = sacc / m[i * n + i];
    }
    Some(x)
}

/// Refine an approximate real zero of the non-negative polynomial `g`
/// (a minimum, so `∇g = 0`) by Newton's method on the gradient; `Some` if
/// it converges to a point where `g` vanishes to double precision.
fn refine_zero(g: &FloatPoly, start: &[f64]) -> Option<Vec<f64>> {
    let n = g.nvars;
    let mut x = start.to_vec();
    for _ in 0..40 {
        let grad = g.gradient(&x);
        let gnorm = grad.iter().map(|v| v * v).sum::<f64>().sqrt();
        if gnorm < 1e-14 {
            break;
        }
        let h = g.hessian(&x);
        let mut step = solve_dense(&h, &grad, n)?;
        // Damp very large steps.
        let snorm = step.iter().map(|v| v * v).sum::<f64>().sqrt();
        if snorm > 10.0 {
            for v in step.iter_mut() {
                *v *= 10.0 / snorm;
            }
        }
        for (xi, si) in x.iter_mut().zip(&step) {
            *xi -= si;
        }
        if x.iter().any(|v| !v.is_finite()) {
            return None;
        }
    }
    let scale = 1.0 + x.iter().map(|v| v.abs()).fold(0.0, f64::max).powi(4);
    if g.value(&x).abs() <= 1e-12 * scale && g.gradient(&x).iter().all(|v| v.abs() <= 1e-9 * scale)
    {
        Some(x)
    } else {
        None
    }
}

/// Real zeros of `g` read off the numerical kernel of `x` (each kernel
/// vector approximates a monomial vector `m(z)` up to scale when the kernel
/// is one-dimensional, and a combination otherwise), refined by Newton and
/// deduplicated.  Returns the monomial vectors `m(z)` of the zeros found.
fn kernel_zeros(g: &Poly, basis: &[Vec<u32>], rows: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    let nvars = g.gens().len();
    let fp = FloatPoly::from_poly(g);
    // Indices of the constant and the degree-one monomials in the basis.
    let Some(one) = basis.iter().position(|m| m.iter().all(|&e| e == 0)) else {
        return Vec::new();
    };
    let lin: Vec<Option<usize>> = (0..nvars)
        .map(|v| {
            basis
                .iter()
                .position(|m| m.iter().enumerate().all(|(i, &e)| e == u32::from(i == v)))
        })
        .collect();
    if lin.iter().any(Option::is_none) {
        return Vec::new();
    }
    let mut zeros: Vec<Vec<f64>> = Vec::new();
    let mut starts: Vec<Vec<f64>> = Vec::new();
    for row in rows {
        if row[one].abs() > 1e-9 {
            starts.push(lin.iter().map(|i| row[i.unwrap_or(0)] / row[one]).collect());
        }
    }
    // A degenerate kernel eigenspace mixes the m(zᵢ); add grid starts so
    // every real zero has a nearby start (cheap for a few variables).
    if nvars <= 3 {
        let grid = [-2.0, -0.7, 0.0, 0.7, 2.0];
        let total = grid.len().pow(nvars as u32);
        for k in 0..total {
            let mut idx = k;
            let mut pt = Vec::with_capacity(nvars);
            for _ in 0..nvars {
                pt.push(grid[idx % grid.len()]);
                idx /= grid.len();
            }
            starts.push(pt);
        }
    }
    for start in starts {
        if let Some(z) = refine_zero(&fp, &start)
            && !zeros.iter().any(|w| {
                w.iter()
                    .zip(&z)
                    .all(|(a, b)| (a - b).abs() < 1e-9 * (1.0 + a.abs()))
            })
        {
            zeros.push(z);
        }
    }
    zeros
        .iter()
        .map(|z| {
            let v: Vec<f64> = basis
                .iter()
                .map(|m| m.iter().zip(z).map(|(&e, zi)| zi.powi(e as i32)).product())
                .collect();
            let norm = v.iter().map(|a| a * a).sum::<f64>().sqrt();
            v.into_iter().map(|a| a / norm).collect()
        })
        .filter(|_| n > 0)
        .collect()
}

/// Candidate rational directions spanning (a rational subspace of) `K⊥`,
/// most reliable first: the exact complement when the kernel is a rational
/// subspace, otherwise the integer relations of the kernel.  Each candidate
/// is an `n × 1` column; they are linearly independent.
fn rational_face_candidates(x: &[f64], n: usize, goal: &Poly, basis: &[Vec<u32>]) -> Vec<QMatrix> {
    let Some((mut rows, mut accuracy)) = numeric_kernel(x, n) else {
        return Vec::new();
    };
    // If the kernel comes from real zeros of the goal, Newton-refine them:
    // the monomial vectors of the refined zeros span the same kernel to
    // double precision, which lets exact relations be told from
    // coincidental near-relations.
    if basis.len() == n {
        let zs = kernel_zeros(goal, basis, &rows, n);
        if zs.len() >= rows.len() && !zs.is_empty() {
            // Orthonormalise (Gram–Schmidt) for the projection tests.
            let mut ortho: Vec<Vec<f64>> = Vec::new();
            for v in zs {
                let mut w = v.clone();
                for u in &ortho {
                    let c: f64 = u.iter().zip(&v).map(|(a, b)| a * b).sum();
                    for (wi, ui) in w.iter_mut().zip(u) {
                        *wi -= c * ui;
                    }
                }
                let norm = w.iter().map(|a| a * a).sum::<f64>().sqrt();
                if norm > 1e-9 {
                    ortho.push(w.into_iter().map(|a| a / norm).collect());
                }
            }
            if ortho.len() == rows.len() {
                rows = ortho;
                accuracy = 1e-13;
            }
        }
    }
    let mut cands: Vec<QMatrix> = Vec::new();
    if accuracy < 1e-6
        && let Some(k) = rationalize_kernel(&rows, n)
    {
        cands.extend(k.nullspace());
    }
    if cands.is_empty() {
        for rel in integer_relations(&rows, n, accuracy) {
            let col = QMatrix::col_vector(rel.iter().map(|v| Q::from_integer(v.clone())).collect());
            // Keep only directions independent of those already chosen.
            let mut all: Vec<&QMatrix> = cands.iter().collect();
            all.push(&col);
            if let Ok(m) = QMatrix::hstack(&all)
                && m.rank() == all.len()
            {
                cands.push(col);
            }
        }
    }
    // Never restrict to the whole space; keep the shortest directions.
    cands.truncate(n - 1);
    cands
}

// ═══════════════════════════════════════════════════════════════════════════
// Driver
// ═══════════════════════════════════════════════════════════════════════════

/// Exact evaluation of `goal` at a rational point.
fn value_at(goal: &Poly, point: &[Q]) -> Option<Q> {
    let ctx = goal.context();
    let vals: Vec<Ex> = point.iter().map(|q| ctx.from_ratio(q.clone())).collect();
    let refs: Vec<&Ex> = vals.iter().collect();
    goal.eval(&refs).ok()?.as_rational()
}

/// Look for a point where the goal is negative: a coarse grid, then a
/// numerical minimisation from the best grid point, rationalised.
fn refute(goal: &Poly) -> Option<(Vec<Q>, Q)> {
    let n = goal.gens().len();
    let total = grid_size(n)?;
    if total > 50_000 {
        return None;
    }
    let mut idx = vec![0usize; n];
    let mut best: Option<(Vec<Q>, Q)> = None;
    let grid = refutation_grid();
    loop {
        let point: Vec<Q> = idx.iter().map(|&k| grid[k].clone()).collect();
        if let Some(v) = value_at(goal, &point) {
            if v.is_negative() {
                return Some((point, v));
            }
            if best.as_ref().is_none_or(|(_, b)| v < *b) {
                best = Some((point.clone(), v));
            }
        }
        let mut pos = 0;
        loop {
            if pos == n {
                break;
            }
            idx[pos] += 1;
            if idx[pos] < grid.len() {
                break;
            }
            idx[pos] = 0;
            pos += 1;
        }
        if pos == n {
            break;
        }
    }
    refute_by_descent(goal, best)
}

/// The sample values of the refutation grid, per coordinate.
fn refutation_grid() -> Vec<Q> {
    [-3i64, -1, 0, 1, 3]
        .iter()
        .map(|&v| Q::from_integer(BigInt::from(v)))
        .chain([
            Q::new(BigInt::from(1), BigInt::from(2)),
            Q::new(BigInt::from(-1), BigInt::from(2)),
        ])
        .collect()
}

/// Number of grid points in `n` variables, `None` on overflow.
fn grid_size(n: usize) -> Option<usize> {
    refutation_grid().len().checked_pow(u32::try_from(n).ok()?)
}

/// Numerical descent (Nelder–Mead on f64) from the best grid point, then
/// an exact check of the rationalised minimiser.
fn refute_by_descent(goal: &Poly, best: Option<(Vec<Q>, Q)>) -> Option<(Vec<Q>, Q)> {
    let (start, _) = best?;
    let ctx = goal.context();
    let f = |p: &[f64]| -> f64 {
        let vals: Vec<Ex> = p
            .iter()
            .map(|&v| ctx.from_f64(v).unwrap_or_else(|_| ctx.zero()))
            .collect();
        let refs: Vec<&Ex> = vals.iter().collect();
        goal.eval(&refs)
            .ok()
            .and_then(|e| e.eval_f64().ok())
            .unwrap_or(f64::INFINITY)
    };
    let x0: Vec<f64> = start.iter().map(q_to_f64).collect();
    let Ok(res) = crate::domains::optimize::nelder_mead(
        f,
        &x0,
        &crate::domains::optimize::MinimizeOpts::default(),
    ) else {
        return None;
    };
    if res.fun < -1e-9 {
        // Rationalise the minimiser and check exactly.
        for digits in [2u32, 4, 6, 9] {
            let den = BigInt::from(10u32).pow(digits);
            let point: Vec<Q> = res
                .x
                .iter()
                .map(|&v| {
                    Q::new(
                        BigInt::from((v * q_to_f64(&Q::from_integer(den.clone()))).round() as i128),
                        den.clone(),
                    )
                })
                .collect();
            if let Some(v) = value_at(goal, &point)
                && v.is_negative()
            {
                return Some((point, v));
            }
        }
    }
    None
}

/// Prove `goal ≥ 0` on ℝⁿ by an exact sum-of-squares decomposition, refute
/// it with an exact point, or report `Unknown`.
///
/// `goal` must be a polynomial with rational coefficients in `vars` (all
/// of its symbols must be listed).  A goal of odd degree, or whose
/// leading form is not a sum of squares, cannot be SOS; the search
/// notices the first case immediately and the others through the SDP.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if `vars` is empty, `goal` is not a
/// rational-coefficient polynomial in `vars`, or the Gram basis would
/// exceed `opts.max_basis`.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::certificates::{prove_sos, SosOpts, SosOutcome};
///
/// let ctx = Context::new();
/// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
/// // (x − 1)² + (y − 1)²: a zero at (1, 1), so every Gram matrix is singular —
/// // the search finds the face and an exact decomposition on it.
/// let goal = (&x - 1).powi(2) + (&y - 1).powi(2);
/// let out = prove_sos(&goal.expand(), &[x.clone(), y.clone()], &SosOpts::default()).unwrap();
/// let cert = out.certificate().expect("SOS");
/// assert!(cert.verify());
/// assert_eq!(cert.rank(), 2);
/// assert!(cert.to_lean("two_squares").unwrap().contains("positivity"));
///
/// // x⁴ + y⁴ − 4xy + 2 is not non-negative: refuted at (1, 1) where it is 0 … try x⁴ + y⁴ − 4xy + 1.
/// match prove_sos(&(x.powi(4) + y.powi(4) - 4 * &x * &y + 1), &[x.clone(), y.clone()], &SosOpts::default()).unwrap() {
///     SosOutcome::Refuted { value, .. } => assert!(value < symplex::linprog::qi(0)),
///     other => panic!("{other:?}"),
/// }
/// ```
pub fn prove_sos(goal: &Ex, vars: &[Ex], opts: &SosOpts) -> Result<SosOutcome, SymplexError> {
    // The time limit is measured from the start of the call: the deadline
    // is fixed here, before parsing, refutation and the Newton-polytope LPs.
    let deadline = opts.deadline_from_now();
    if vars.is_empty() {
        return Err(invalid("at least one variable is required"));
    }
    let gens: Vec<&Ex> = vars.iter().collect();
    for s in goal.free_symbols() {
        if !vars.contains(&s) {
            return Err(invalid(format!(
                "goal `{goal}` mentions `{s}`, which is not one of the variables"
            )));
        }
    }
    let goal_poly = Poly::try_new(goal, &gens).map_err(|e| match e {
        SymplexError::InvalidArgument { reason, .. } => invalid(format!("goal `{goal}`: {reason}")),
        other => other,
    })?;
    if !goal_poly.has_rational_coeffs() {
        return Err(invalid("goal must have rational coefficients"));
    }
    if goal_poly.is_zero() {
        let basis = vec![vec![0u32; vars.len()]];
        let gram = QMatrix::zeros(1, 1);
        return Ok(SosOutcome::Proved(SosCertificate::from_gram(
            goal_poly, basis, gram,
        )?));
    }
    let deg = goal_poly.total_degree().unwrap_or(0);
    let budget_exhausted = |where_: &str| {
        Ok(Outcome::Unknown(SosUnknown::deadline(format!(
            "budget exhausted: deadline passed {where_}"
        ))))
    };
    // 1. Cheap refutation (a bounded grid plus one descent; a
    //    counterexample is decisive, so this stage is not budget-bound).
    if let Some((point, value)) = refute(&goal_poly) {
        let point = vars.iter().cloned().zip(point).collect();
        return Ok(Outcome::Refuted {
            point,
            value,
            param_value: None,
        });
    }
    if deg % 2 == 1 {
        return Ok(Outcome::Unknown(SosUnknown::new(
            "the goal has odd degree, so it is not a sum of squares (and not bounded below)",
        )));
    }
    if let Some(c) = goal_poly
        .coeff_monomial(&vec![0u32; vars.len()])
        .ok()
        .and_then(|c| c.as_rational())
        && goal_poly.is_ground()
    {
        // A constant.
        return if c.is_negative() {
            Ok(Outcome::Refuted {
                point: vars.iter().map(|v| (v.clone(), Q::zero())).collect(),
                value: c,
                param_value: None,
            })
        } else {
            let basis = vec![vec![0u32; vars.len()]];
            let gram = QMatrix::from_fn(1, 1, |_, _| c.clone());
            Ok(SosOutcome::Proved(SosCertificate::from_gram(
                goal_poly, basis, gram,
            )?))
        };
    }

    // 2. Gram basis: monomials of degree ≤ deg/2 whose doubled exponent lies
    //    in the goal's Newton polytope — a monomial outside it cannot occur
    //    in any square of the decomposition (Reznick), so pruning loses no
    //    certificate and keeps sparse goals within `max_basis`.
    let basis = match newton_pruned_basis(&goal_poly, vars.len(), deg / 2, deadline) {
        Ok(basis) => basis,
        Err(DeadlinePassed) => return budget_exhausted("during the Newton-polytope pruning"),
    };
    if basis.len() > opts.max_basis {
        return Err(invalid(format!(
            "the Gram basis has {} monomials, above max_basis = {}",
            basis.len(),
            opts.max_basis
        )));
    }
    if check_deadline(deadline).is_err() {
        return budget_exhausted("before the SDP was assembled");
    }
    let problem = SosProblem::new(&goal_poly, &basis)?;

    // 3. Solve, reduce, round.
    let mut face: Option<QMatrix> = None; // B with Q = B Q' Bᵀ
    let mut current = problem;
    for round in 0..=opts.max_facial_reductions {
        let out_of_time = |where_: &str| {
            Ok(Outcome::Unknown(SosUnknown::deadline(format!(
                "budget exhausted: deadline passed {where_} (facial-reduction round {round} of {})",
                opts.max_facial_reductions
            ))))
        };
        if check_deadline(deadline).is_err() {
            return out_of_time("before the SDP solve");
        }
        let x = match current.solve_numeric(opts.max_iterations, deadline) {
            Ok(Some(x)) => x,
            Ok(None) => {
                return Ok(Outcome::Unknown(SosUnknown::new(
                    "the interior-point method did not converge (no sum-of-squares decomposition of this degree, or a numerically hard one)",
                )));
            }
            Err(DeadlinePassed) => return out_of_time("inside the interior-point loop"),
        };
        for &digits in &opts.rounding_digits {
            if let Some(q_small) = current.round_and_project(&x, digits) {
                let q_full = match &face {
                    Some(b) => &(b * &q_small) * &b.transpose(),
                    None => q_small,
                };
                if q_full.is_positive_semidefinite()
                    && let Ok(cert) =
                        SosCertificate::from_gram(goal_poly.clone(), basis.clone(), q_full)
                    && cert.verify()
                {
                    return Ok(SosOutcome::Proved(cert));
                }
            }
        }
        // Rounding failed: the solution is (numerically) singular.  Find a
        // rational face containing the solution's range and restrict to it,
        // dropping the least reliable candidate directions until the
        // restricted coefficient system is exactly consistent.
        let cands = rational_face_candidates(&x, current.n, &goal_poly, &basis);
        if cands.is_empty() {
            return Ok(Outcome::Unknown(SosUnknown::new(
                "rounding the interior-point solution did not give an exact PSD Gram matrix, and no rational face was found to reduce to",
            )));
        }
        let mut restricted: Option<(QMatrix, SosProblem)> = None;
        for k in (1..=cands.len()).rev() {
            let cols: Vec<&QMatrix> = cands[..k].iter().collect();
            let Ok(b_small) = QMatrix::hstack(&cols) else {
                continue;
            };
            if let Ok(p) = current.restrict(&b_small) {
                restricted = Some((b_small, p));
                break;
            }
        }
        let Some((b_small, next)) = restricted else {
            return Ok(Outcome::Unknown(SosUnknown::new(
                "the numerically detected face is not consistent with the coefficient constraints",
            )));
        };
        face = Some(match &face {
            Some(b) => b * &b_small,
            None => b_small,
        });
        current = next;
    }
    Ok(Outcome::Unknown(SosUnknown::new(format!(
        "no exact decomposition after {} rounds of facial reduction",
        opts.max_facial_reductions
    ))))
}

/// Convenience: `Some(true)` with a verified SOS certificate, `Some(false)`
/// with an exact counterexample, `None` when undecided.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::certificates::is_sos;
///
/// let ctx = Context::new();
/// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
/// assert_eq!(is_sos(&(x.powi(2) - 2 * &x * &y + y.powi(2) + 1), &[x.clone(), y.clone()]), Some(true));
/// assert_eq!(is_sos(&(&x * &y), &[x.clone(), y.clone()]), Some(false));
/// ```
pub fn is_sos(goal: &Ex, vars: &[Ex]) -> Option<bool> {
    match prove_sos(goal, vars, &SosOpts::default()) {
        Ok(Outcome::Proved(_)) => Some(true),
        Ok(Outcome::Refuted { .. }) => Some(false),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domains::linprog::q;

    fn setup() -> (Context, Ex, Ex) {
        let ctx = Context::new();
        (ctx.clone(), ctx.symbol("x"), ctx.symbol("y"))
    }

    fn sos(goal: &Ex, vars: &[Ex]) -> SosCertificate {
        match prove_sos(goal, vars, &SosOpts::default()).unwrap() {
            SosOutcome::Proved(c) => {
                assert!(c.verify(), "{c}");
                c
            }
            other => panic!("{goal}: {other:?}"),
        }
    }

    #[test]
    fn tri_indexing_round_trips() {
        for n in 1..6 {
            let mut k = 0;
            for i in 0..n {
                for j in i..n {
                    assert_eq!(tri_index(i, j, n), k);
                    assert_eq!(tri_pair(k, n), (i, j));
                    k += 1;
                }
            }
        }
    }

    #[test]
    fn rationalize_small_denominators() {
        assert_eq!(rationalize(0.5, 100, 1e-9), Some(q(1, 2)));
        assert_eq!(rationalize(-0.333333333, 100, 1e-6), Some(q(-1, 3)));
        assert_eq!(rationalize(2.0, 100, 1e-9), Some(q(2, 1)));
        assert_eq!(rationalize(0.0, 100, 1e-9), Some(q(0, 1)));
        assert_eq!(rationalize(0.142857142857, 10, 1e-6), Some(q(1, 7)));
        assert!(rationalize(std::f64::consts::PI, 10, 1e-9).is_none());
    }

    #[test]
    fn strictly_positive_quadratic() {
        let (_ctx, x, y) = setup();
        let g = x.powi(2) - &x * &y + y.powi(2) + 1;
        let c = sos(&g, &[x.clone(), y.clone()]);
        assert!(c.rank() >= 1);
        assert!(c.gram().is_positive_semidefinite());
        let Equation { lhs, rhs } = c.identity();
        assert!((lhs - rhs).expand().is_zero_structural());
    }

    #[test]
    fn interior_zero_needs_facial_reduction() {
        let (_ctx, x, y) = setup();
        let g = ((&x - 1).powi(2) + (&y - 1).powi(2)).expand();
        let c = sos(&g, &[x.clone(), y.clone()]);
        assert_eq!(c.rank(), 2);
        let g2 = ((&x - &y).powi(2) * (&x + 1).powi(2)).expand();
        let c2 = sos(&g2, &[x.clone(), y.clone()]);
        assert!(c2.rank() >= 1);
    }

    #[test]
    fn quartics_and_univariate() {
        let (ctx, x, y) = setup();
        // x⁴ + y⁴ − 2x²y² + ε…: (x² − y²)² is a square with a whole curve of zeros.
        let g = (x.powi(2) - y.powi(2)).powi(2).expand();
        let c = sos(&g, &[x.clone(), y.clone()]);
        assert_eq!(c.rank(), 1);
        // Univariate: x⁴ − 2x³ + 2x² − 2x + 1 = (x² + 1)(x − 1)².
        let u = (x.powi(2) + 1) * (&x - 1).powi(2);
        let c = sos(&u.expand(), std::slice::from_ref(&x));
        assert!(c.verify());
        // Positive definite quartic with rational fill.
        let g = x.powi(4) + y.powi(4) + &x * &y * ctx.rational(1, 2) + 1;
        sos(&g, &[x.clone(), y.clone()]);
    }

    #[test]
    fn refuted_and_unknown() {
        let (_ctx, x, y) = setup();
        match prove_sos(&(&x * &y), &[x.clone(), y.clone()], &SosOpts::default()).unwrap() {
            SosOutcome::Refuted { value, point, .. } => {
                assert!(value.is_negative());
                assert_eq!(point.len(), 2);
            }
            other => panic!("{other:?}"),
        }
        match prove_sos(
            &(x.powi(3) + 1),
            std::slice::from_ref(&x),
            &SosOpts::default(),
        )
        .unwrap()
        {
            SosOutcome::Refuted { value, .. } => assert!(value.is_negative()),
            other => panic!("{other:?}"),
        }
        // Motzkin: non-negative, not SOS.
        let m = x.powi(4) * y.powi(2) + x.powi(2) * y.powi(4) - 3 * x.powi(2) * y.powi(2) + 1;
        match prove_sos(&m, &[x.clone(), y.clone()], &SosOpts::default()).unwrap() {
            SosOutcome::Unknown(_) => {}
            other => panic!("Motzkin should be Unknown, got {other:?}"),
        }
        assert_eq!(is_sos(&m, &[x.clone(), y.clone()]), None);
    }

    #[test]
    fn errors_and_edge_cases() {
        let (ctx, x, y) = setup();
        assert!(prove_sos(&x, &[], &SosOpts::default()).is_err());
        assert!(prove_sos(&x.sin(), std::slice::from_ref(&x), &SosOpts::default()).is_err());
        assert!(prove_sos(&(&x + &y), std::slice::from_ref(&x), &SosOpts::default()).is_err());
        // Newton-polytope pruning: x² + y² needs only the basis {x, y}
        // (the constant is outside the hull of {(2,0), (0,2)}), so a
        // budget of 2 succeeds and a budget of 1 is refused.
        let two = SosOpts::default().with_max_basis(2);
        assert!(
            prove_sos(&(x.powi(2) + y.powi(2)), &[x.clone(), y.clone()], &two)
                .unwrap()
                .is_proved()
        );
        let small = SosOpts::default().with_max_basis(1);
        assert!(prove_sos(&(x.powi(2) + y.powi(2)), &[x.clone(), y.clone()], &small).is_err());
        // A sparse degree-8 goal: x⁸ + y⁸ + 1 has 45 monomials of degree ≤ 4
        // but only the 6 with doubled exponents in the hull of
        // {(8,0), (0,8), (0,0)} — the axes' even powers — survive, so it
        // fits a small budget and is certified.
        let sparse = prove_sos(
            &(x.powi(8) + y.powi(8) + 1),
            &[x.clone(), y.clone()],
            &SosOpts::default().with_max_basis(15),
        )
        .unwrap();
        let c = sparse.certificate().expect("sum of even powers");
        assert!(c.basis().len() <= 15, "{}", c.basis().len());
        assert!(c.verify());
        let zero = sos(&ctx.zero(), std::slice::from_ref(&x));
        assert_eq!(zero.rank(), 0);
        let c = sos(&ctx.int(3), std::slice::from_ref(&x));
        assert_eq!(c.rank(), 1);
        match prove_sos(&ctx.int(-2), std::slice::from_ref(&x), &SosOpts::default()).unwrap() {
            SosOutcome::Refuted { value, .. } => assert_eq!(value, q(-2, 1)),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn json_round_trip_reverifies() {
        let (_ctx, x, y) = setup();
        let g = ((&x - 1).powi(2) + (&y - 1).powi(2)).expand();
        let c = sos(&g, &[x.clone(), y.clone()]);
        let json = c.to_json().unwrap();
        let back = SosCertificate::from_json(&Context::new(), &json).unwrap();
        assert_eq!(back.to_string(), c.to_string());
        let mut d = c.to_data();
        d.gram[0] = "5/1".to_string();
        assert!(SosCertificate::from_data(&Context::new(), &d).is_err());
    }
}

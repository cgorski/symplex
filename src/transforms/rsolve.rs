//! Recurrence-relation solver (linear, constant-coefficient, with forcing).
//!
//! Sequences are described by coefficient lists rather than by an `a(n)`
//! function-application node, so the API is purely expression based:
//!
//! - [`rsolve_linear`] — `c₀·a(n) + c₁·a(n+1) + … + c_k·a(n+k) = f(n)` with
//!   constant coefficients.  Characteristic roots come from the univariate
//!   solver (repeated roots → `nʲ·rⁿ`, complex pairs → `ρⁿ·cos(nθ)`,
//!   `ρⁿ·sin(nθ)`), a particular solution is found by undetermined
//!   coefficients for `poly(n)·bⁿ` forcing (resonance handled), and initial
//!   values are fitted with `linsolve`.
//! - [`rsolve_first_order`] — `a(n+1) = p(n)·a(n) + q(n)` with variable
//!   coefficients, via the product formula `a(n) = P(n)·(a₀ + Σ q(k)/P(k+1))`
//!   where `P(n) = Π_{k<n} p(k)` is expressed with factorials / Gamma ratios
//!   for polynomial `p`.
//!
//! # Examples
//!
//! ```
//! use symplex::prelude::*;
//! use symplex::rsolve::rsolve_linear;
//!
//! let ctx = Context::new();
//! let n = ctx.symbol("n");
//! // Fibonacci: a(n+2) - a(n+1) - a(n) = 0, a(0) = 0, a(1) = 1  →  Binet
//! let coeffs = [ctx.int(-1), ctx.int(-1), ctx.int(1)];
//! let f = rsolve_linear(&coeffs, None, &n, &[ctx.int(0), ctx.int(1)]).unwrap();
//! let fib = [0.0, 1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0];
//! for (k, expected) in fib.iter().enumerate() {
//!     let v = f.subs_i64(&n, k as i64).eval_f64().unwrap();
//!     assert!((v - expected).abs() < 1e-9, "a({k}) = {v}");
//! }
//! ```

use crate::api::expr::Ex;
use crate::api::expr_solve_ext::{LinearSolution, linsolve};
use crate::base::errors::SymplexError;
use crate::base::node::ExprNode;
use crate::base::numeric::Q;

/// A parsed forcing term `c · n^d · bⁿ`.
struct ForcingTerm {
    coeff: Ex,
    degree: usize,
    base: Ex,
}

/// Name of the `k`-th arbitrary constant.
fn constant(ctx: &crate::api::context::Context, k: usize) -> Ex {
    ctx.symbol(&format!("C{k}"))
}

/// Rational value of a numeric literal.
fn as_rational(e: &Ex) -> Option<Q> {
    e.inner.read().arena.as_num(e.raw_id()).cloned()
}

/// Parse one additive forcing term (free of the sequence) into
/// `c · n^d · bⁿ`.  `bⁿ` may appear as `b^n`, `b^(n + m)` (the shift is
/// folded into `c`) or `b^(k·n)` (base becomes `b^k`).
fn parse_forcing_term(term: &Ex, n: &Ex) -> Option<ForcingTerm> {
    let ctx = term.context();
    let factors: Vec<Ex> = if term.expr_type() == crate::api::expr::ExprType::Mul {
        term.args()
    } else {
        vec![term.clone()]
    };
    let mut coeff = ctx.one();
    let mut degree = 0usize;
    let mut base: Option<Ex> = None;
    for f in factors {
        if &f == n {
            degree += 1;
            continue;
        }
        if !f.contains(n) {
            coeff = &coeff * &f;
            continue;
        }
        let node = {
            let inner = f.inner.read();
            inner.arena.node(f.raw_id()).clone()
        };
        match node {
            ExprNode::Pow(b, e) => {
                let b = f.wrap(b);
                let e = f.wrap(e);
                if &b == n && !e.contains(n) {
                    // n^d
                    let d = as_rational(&e)?;
                    if !d.is_integer() || num_traits::Signed::is_negative(&d) {
                        return None;
                    }
                    degree += usize::try_from(d.to_integer()).ok()?;
                } else if !b.contains(n) {
                    // b^(k·n + m)
                    if base.is_some() {
                        return None;
                    }
                    let coeffs = {
                        let mut inner = f.inner.write();
                        crate::transforms::solve::symbolic_poly_coeffs(
                            &mut inner.arena,
                            e.raw_id(),
                            n.raw_id(),
                        )
                    }?;
                    if coeffs.len() != 2 {
                        return None;
                    }
                    let m = f.wrap(coeffs[0]);
                    let k = f.wrap(coeffs[1]);
                    // b^(k n + m) = (b^k)^n · b^m
                    let bk = if k.is_one_structural() {
                        b.clone()
                    } else {
                        b.pow(&k)
                    };
                    let bm = b.pow(&m).eval();
                    coeff = &coeff * &bm;
                    base = Some(bk.eval());
                } else {
                    return None;
                }
            }
            ExprNode::Exp(arg) => {
                // e^(k n + m)
                if base.is_some() {
                    return None;
                }
                let arg = f.wrap(arg);
                let coeffs = {
                    let mut inner = f.inner.write();
                    crate::transforms::solve::symbolic_poly_coeffs(
                        &mut inner.arena,
                        arg.raw_id(),
                        n.raw_id(),
                    )
                }?;
                if coeffs.len() != 2 {
                    return None;
                }
                let m = f.wrap(coeffs[0]);
                let k = f.wrap(coeffs[1]);
                coeff = &coeff * &m.exp().eval();
                base = Some(k.exp().eval());
            }
            _ => return None,
        }
    }
    Some(ForcingTerm {
        coeff: coeff.eval(),
        degree,
        base: base.unwrap_or_else(|| ctx.one()),
    })
}

/// Build `n^j` (with `n^0 = 1`, `n^1 = n`).
fn n_pow(n: &Ex, j: usize) -> Ex {
    match j {
        0 => n.context().one(),
        1 => n.clone(),
        _ => n.powi(j as i64),
    }
}

/// Node budget for the closed form (before and after fitting the initial
/// values); exceeding it is reported as "expression swell" instead of
/// letting simplification of nested radicals spin.
const RSOLVE_BUDGET: usize = crate::base::config::EXPRESSION_BUDGET / 4;

/// Fit the constants of `a(n) = Σ C_k r_kⁿ` (all roots simple, no forcing)
/// to `a(0), …, a(m−1)` through the closed-form inverse of the Vandermonde
/// matrix: with `L_k(x) = Π_{i≠k} (x − r_i)/(r_k − r_i) = Σ_j ℓ_{kj} xʲ`,
/// `C_k = Σ_j ℓ_{kj} a_j`.  This avoids symbolic Gaussian elimination on
/// algebraic (`RootOf` / radical) entries, which is slow and swells.
///
/// Returns `None` when the shape does not apply (repeated roots, a root
/// `0`, complex pairs in real form, a particular solution, or fewer
/// initial values than roots).
fn fit_vandermonde(roots: &[(Ex, usize)], n: &Ex, ics: &[Ex]) -> Option<Ex> {
    let ctx = n.context();
    if roots.len() != ics.len() || roots.is_empty() {
        return None;
    }
    let i_unit = ctx.i_unit();
    if roots
        .iter()
        .any(|(r, m)| *m != 1 || r.is_zero_structural() || r.contains(&i_unit))
    {
        return None;
    }
    let rs: Vec<Ex> = roots.iter().map(|(r, _)| r.clone()).collect();
    let m = rs.len();
    let mut closed = ctx.zero();
    for k in 0..m {
        // Numerator coefficients of Π_{i≠k} (x − r_i), ascending in x.
        let mut num: Vec<Ex> = vec![ctx.one()];
        let mut denom = ctx.one();
        for (i, ri) in rs.iter().enumerate() {
            if i == k {
                continue;
            }
            let mut next: Vec<Ex> = vec![ctx.zero(); num.len() + 1];
            for (j, c) in num.iter().enumerate() {
                next[j + 1] = (&next[j + 1] + c).eval();
                next[j] = (&next[j] - &(c * ri)).eval();
            }
            num = next;
            let d = (&rs[k] - ri).eval();
            if d.is_zero_structural() {
                return None; // repeated root that slipped through
            }
            denom = (&denom * &d).eval();
        }
        let mut ck = ctx.zero();
        for (j, a) in ics.iter().enumerate() {
            if a.is_zero_structural() {
                continue;
            }
            ck = (&ck + &(&num[j] * a)).eval();
        }
        let ck = (&ck / &denom).eval();
        if ck.is_zero_structural() {
            continue;
        }
        let term = if rs[k].is_one_structural() {
            ck
        } else {
            &ck * &rs[k].pow(n)
        };
        closed = &closed + &term;
    }
    Some(closed.eval())
}

/// Tree size of `e` (nodes, without sharing), saturating at `cap + 1` so
/// that a caller comparing against `cap` sees the overflow.
fn tree_size_capped(e: &Ex, cap: usize) -> usize {
    let inner = e.inner.read();
    crate::transforms::pattern::tree_size_capped(&inner.arena, e.raw_id(), cap.saturating_add(1))
}

fn swell_check(exprs: &[&Ex]) -> Result<(), SymplexError> {
    let mut total = 0usize;
    for e in exprs {
        total += tree_size_capped(e, RSOLVE_BUDGET - total.min(RSOLVE_BUDGET));
        if total > RSOLVE_BUDGET {
            return Err(SymplexError::ComputationFailed {
                operation: "rsolve_linear",
                reason: format!(
                    "expression swell: the closed form exceeds the budget of {RSOLVE_BUDGET} \
                     expression nodes"
                ),
            });
        }
    }
    Ok(())
}

/// Characteristic roots with multiplicities (`(root, multiplicity)`).
///
/// The characteristic polynomial is factored over ℤ.  Linear and quadratic
/// irreducible factors are solved exactly (rationals, quadratic radicals);
/// the roots of an irreducible factor of degree ≥ 3 are represented as
/// `RootOf(factor, k)` — Cardano / Ferrari radicals for those roots are
/// nested cube roots (often with unevaluable `re`/`im` parts) whose
/// simplification does not terminate in practice, whereas `RootOf` values
/// evaluate numerically to any precision.
fn characteristic_roots(coeffs: &[Ex], n: &Ex) -> Result<Vec<(Ex, usize)>, SymplexError> {
    let ctx = n.context();
    // Bound variable of the characteristic polynomial (it only survives
    // inside `RootOf` nodes, so it must differ from the index symbol).
    let r = if format!("{n}") == "r" {
        ctx.symbol("_r")
    } else {
        ctx.symbol("r")
    };
    // Build Σ c_k r^k as a rational polynomial when possible.
    let mut rat_coeffs = Vec::with_capacity(coeffs.len());
    for c in coeffs {
        match as_rational(&c.eval()) {
            Some(v) => rat_coeffs.push(v),
            None => {
                return Err(SymplexError::InvalidArgument {
                    operation: "rsolve_linear",
                    reason: format!("coefficient {c} is not a rational number"),
                });
            }
        }
    }
    let poly = crate::poly::Poly::from_coeffs(rat_coeffs);
    let (_content, factors) = poly.factor_over_z();
    let mut roots: Vec<(Ex, usize)> = Vec::new();
    for (factor, mult) in factors {
        let mult = mult as usize;
        let degree = factor.degree().unwrap_or(0);
        if degree == 0 {
            continue;
        }
        let f_expr = {
            let mut inner = ctx.inner.write();
            crate::poly::polybridge::poly_to_expr(&mut inner.arena, &factor, r.raw_id())
        };
        if degree >= 3 {
            for k in 0..degree {
                let root = {
                    let mut inner = ctx.inner.write();
                    let idx = inner.arena.int(k as i64);
                    inner
                        .arena
                        .intern(ExprNode::RootOf(f_expr, r.raw_id(), idx))
                };
                roots.push((r.wrap(root), mult));
            }
            continue;
        }
        let sols = {
            let mut inner = ctx.inner.write();
            crate::transforms::solve::solve(&mut inner.arena, f_expr, r.raw_id())
        };
        if sols.len() != degree {
            return Err(SymplexError::ComputationFailed {
                operation: "rsolve_linear",
                reason: "could not find the characteristic roots".into(),
            });
        }
        for s in sols {
            let root = r.wrap(s.value).eval();
            if root.has_unevaluated() {
                return Err(SymplexError::ComputationFailed {
                    operation: "rsolve_linear",
                    reason: format!("characteristic root is not in closed form: {root}"),
                });
            }
            roots.push((root, mult));
        }
    }
    Ok(roots)
}

/// Homogeneous basis sequences for the given characteristic roots.
///
/// Real root `r` of multiplicity `m` → `n^j·rⁿ` (j < m).  A complex pair
/// `α ± βi` → `n^j·ρⁿ·cos(nθ)`, `n^j·ρⁿ·sin(nθ)` with `ρ = |α+βi|`,
/// `θ = atan2(β, α)`.
fn homogeneous_basis(roots: &[(Ex, usize)], n: &Ex) -> Vec<Ex> {
    let ctx = n.context();
    let i_unit = ctx.i_unit();
    let mut basis = Vec::new();
    let mut used = vec![false; roots.len()];
    for i in 0..roots.len() {
        if used[i] {
            continue;
        }
        used[i] = true;
        let (root, mult) = &roots[i];
        if !root.contains(&i_unit) {
            for j in 0..*mult {
                let term = if root.is_zero_structural() {
                    // Root 0: only the finitely many shifted deltas — skip.
                    continue;
                } else if root.is_one_structural() {
                    n_pow(n, j)
                } else {
                    &n_pow(n, j) * &root.pow(n)
                };
                basis.push(term.eval());
            }
            continue;
        }
        let re = root.re().eval().simplify();
        let im = root.im().eval().simplify();
        // Consume the conjugate.
        let neg_im = (-&im).eval().simplify();
        for k in (i + 1)..roots.len() {
            if used[k] {
                continue;
            }
            let re2 = roots[k].0.re().eval().simplify();
            let im2 = roots[k].0.im().eval().simplify();
            if re2 == re && im2 == neg_im {
                used[k] = true;
                break;
            }
        }
        let rho = (&re.powi(2) + &im.powi(2)).sqrt().eval().simplify();
        let theta = im.atan2(&re).eval().simplify();
        for j in 0..*mult {
            let envelope = if rho.is_one_structural() {
                n_pow(n, j)
            } else {
                &n_pow(n, j) * &rho.pow(n)
            };
            let n_theta = n * &theta;
            basis.push((&envelope * &n_theta.cos()).eval());
            basis.push((&envelope * &n_theta.sin()).eval());
        }
    }
    basis
}

/// Particular solution for forcing `Σ_d q_d·n^d · bⁿ` by undetermined
/// coefficients: trial `n^s·(A₀ + … + A_d·n^d)·bⁿ` where `s` is the
/// multiplicity of `b` as a characteristic root.  The unknowns are solved
/// with [`linsolve`] after dividing the recurrence by `bⁿ`.
fn particular_solution(
    coeffs: &[Ex],
    q: &[Ex],
    base: &Ex,
    roots: &[(Ex, usize)],
    n: &Ex,
) -> Result<Ex, SymplexError> {
    let ctx = n.context();
    let s = roots
        .iter()
        .find(|(r, _)| (r - base).eval().simplify().is_zero_structural())
        .map(|(_, m)| *m)
        .unwrap_or(0);
    let d = q.len() - 1;
    let unknowns: Vec<Ex> = (0..=d).map(|j| ctx.symbol(&format!("__A{j}"))).collect();
    // trial(n) / bⁿ  =  n^s · Σ A_j n^j
    let trial_poly = |shift: i64| -> Ex {
        let m = n + shift;
        let mut acc = ctx.zero();
        for (j, a) in unknowns.iter().enumerate() {
            acc = &acc + &(a * &n_pow(&m, j));
        }
        &acc * &n_pow(&m, s)
    };
    // Σ_k c_k · trial(n+k) / bⁿ = Σ_k c_k · b^k · trial_poly(n+k)
    let mut lhs = ctx.zero();
    for (k, c) in coeffs.iter().enumerate() {
        if c.is_zero_structural() {
            continue;
        }
        let bk = base.powi(k as i64).eval();
        lhs = &lhs + &(&(c * &bk) * &trial_poly(k as i64));
    }
    let mut rhs = ctx.zero();
    for (j, qj) in q.iter().enumerate() {
        rhs = &rhs + &(qj * &n_pow(n, j));
    }
    let residual = (&lhs - &rhs).expand().eval();
    // Coefficients in n must all vanish.
    let eqs: Vec<Ex> = {
        let mut inner = ctx.inner.write();
        let coeffs = crate::transforms::solve::symbolic_poly_coeffs(
            &mut inner.arena,
            residual.raw_id(),
            n.raw_id(),
        )
        .ok_or_else(|| SymplexError::ComputationFailed {
            operation: "rsolve_linear",
            reason: "forcing residual is not polynomial in n".into(),
        })?;
        coeffs.into_iter().map(|id| residual.wrap(id)).collect()
    };
    let eqs: Vec<Ex> = eqs
        .into_iter()
        .filter(|e| !e.is_zero_structural())
        .collect();
    if eqs.is_empty() {
        return Ok(ctx.zero());
    }
    match linsolve(&eqs, &unknowns)? {
        LinearSolution::Unique(pairs) => {
            let mut sol = trial_poly(0);
            for (a, v) in &pairs {
                sol = sol.subs(a, v);
            }
            let sol = if base.is_one_structural() {
                sol
            } else {
                &sol * &base.pow(n)
            };
            Ok(sol.eval().simplify())
        }
        LinearSolution::Parametric { solution, free } => {
            // Underdetermined trial: set free unknowns to zero.
            let mut sol = trial_poly(0);
            for (a, v) in &solution {
                let v = if free.contains(a) {
                    ctx.zero()
                } else {
                    v.clone()
                };
                sol = sol.subs(a, &v);
            }
            for f in &free {
                sol = sol.subs(f, &ctx.zero());
            }
            let sol = if base.is_one_structural() {
                sol
            } else {
                &sol * &base.pow(n)
            };
            Ok(sol.eval().simplify())
        }
        LinearSolution::Inconsistent => Err(SymplexError::ComputationFailed {
            operation: "rsolve_linear",
            reason: "undetermined coefficients failed for the forcing term".into(),
        }),
    }
}

/// Solve the linear constant-coefficient recurrence
///
/// `c₀·a(n) + c₁·a(n+1) + … + c_k·a(n+k) = f(n)`
///
/// for a closed form `a(n)`.
///
/// # Arguments
///
/// - `coeffs` — `[c₀, c₁, …, c_k]`, rational constants, `c_k ≠ 0`.
/// - `forcing` — optional `f(n)`, a sum of terms `c·n^d·bⁿ` (polynomials,
///   exponentials and their products; `b^(n+m)`, `exp(k·n)` accepted).
/// - `n` — the index symbol.
/// - `ics` — optional initial values `a(0), a(1), …` (at most `k`).
///   Constants not determined by `ics` remain as `C1, C2, …`.
///
/// # Errors
///
/// - [`SymplexError::InvalidArgument`] for fewer than two coefficients, a
///   zero leading coefficient, non-rational coefficients, or too many
///   initial values.
/// - [`SymplexError::ComputationFailed`] if the characteristic roots are
///   not available in closed form, the forcing is not of the supported
///   shape, or the initial values are inconsistent.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::rsolve::rsolve_linear;
///
/// let ctx = Context::new();
/// let n = ctx.symbol("n");
/// // Towers of Hanoi: a(n+1) = 2·a(n) + 1, a(0) = 0  →  2ⁿ − 1
/// let sol = rsolve_linear(&[ctx.int(-2), ctx.int(1)], Some(&ctx.int(1)), &n, &[ctx.int(0)]).unwrap();
/// assert_eq!(format!("{sol}"), "2^n - 1");
/// ```
pub fn rsolve_linear(
    coeffs: &[Ex],
    forcing: Option<&Ex>,
    n: &Ex,
    ics: &[Ex],
) -> Result<Ex, SymplexError> {
    if coeffs.len() < 2 {
        return Err(SymplexError::InvalidArgument {
            operation: "rsolve_linear",
            reason: "need at least two coefficients (order ≥ 1)".into(),
        });
    }
    for c in coeffs {
        let _ = n.checked_id(c);
    }
    for v in ics {
        let _ = n.checked_id(v);
    }
    let order = coeffs.len() - 1;
    if coeffs[order].eval().is_zero_structural() {
        return Err(SymplexError::InvalidArgument {
            operation: "rsolve_linear",
            reason: "leading coefficient must be nonzero".into(),
        });
    }
    if ics.len() > order {
        return Err(SymplexError::InvalidArgument {
            operation: "rsolve_linear",
            reason: format!("at most {order} initial values allowed, got {}", ics.len()),
        });
    }
    let ctx = n.context();
    let coeffs: Vec<Ex> = coeffs.iter().map(|c| c.eval()).collect();

    let roots = characteristic_roots(&coeffs, n)?;
    let basis = homogeneous_basis(&roots, n);
    let constants: Vec<Ex> = (1..=basis.len()).map(|k| constant(&ctx, k)).collect();
    let mut general = ctx.zero();
    for (c, b) in constants.iter().zip(&basis) {
        general = &general + &(c * b);
    }

    // Particular solution: group forcing terms by base.
    if let Some(f) = forcing {
        let _ = n.checked_id(f);
        let f = f.expand().eval();
        if !f.is_zero_structural() {
            let terms: Vec<Ex> = if f.expr_type() == crate::api::expr::ExprType::Add {
                f.args()
            } else {
                vec![f.clone()]
            };
            let mut groups: Vec<(Ex, Vec<Ex>)> = Vec::new();
            for t in &terms {
                let ft =
                    parse_forcing_term(t, n).ok_or_else(|| SymplexError::ComputationFailed {
                        operation: "rsolve_linear",
                        reason: format!("unsupported forcing term: {t}"),
                    })?;
                let entry = match groups.iter_mut().find(|(b, _)| *b == ft.base) {
                    Some(e) => e,
                    None => {
                        groups.push((ft.base.clone(), Vec::new()));
                        groups
                            .last_mut()
                            .ok_or_else(|| SymplexError::ComputationFailed {
                                operation: "rsolve_linear",
                                reason: "internal grouping error".into(),
                            })?
                    }
                };
                if entry.1.len() <= ft.degree {
                    entry.1.resize(ft.degree + 1, ctx.zero());
                }
                entry.1[ft.degree] = (&entry.1[ft.degree] + &ft.coeff).eval();
            }
            for (base, q) in &groups {
                let yp = particular_solution(&coeffs, q, base, &roots, n)?;
                general = &general + &yp;
            }
        }
    }
    let general = general.eval();
    swell_check(&[&general])?;

    if ics.is_empty() || constants.is_empty() {
        return Ok(general.simplify());
    }

    // Homogeneous with simple roots: closed-form Vandermonde inverse.
    if forcing.is_none_or(|f| f.eval().is_zero_structural())
        && let Some(fitted) = fit_vandermonde(&roots, n, ics)
    {
        swell_check(&[&fitted])?;
        return Ok(fitted.simplify());
    }

    // Fit initial values a(0..len).
    let eqs: Vec<Ex> = ics
        .iter()
        .enumerate()
        .map(|(k, v)| (general.subs_i64(n, k as i64).eval() - v).eval())
        .collect();
    let fitted =
        crate::api::expr_solve_ext::fit_constants(&general, &constants, &eqs, "rsolve_linear")?;
    swell_check(&[&fitted])?;
    Ok(fitted)
}

/// Closed form of `P(n) = Π_{k=0}^{n-1} p(k)` for constant or linear `p`.
///
/// - constant `c` → `cⁿ`
/// - `α·k + β` → `αⁿ · Γ(n + β/α) / Γ(β/α)` (as `(n + β/α − 1)!/(β/α − 1)!`
///   when `β/α` is a positive integer; `n!` for `k + 1`)
///
/// Other shapes return an unevaluated `Product` node.
fn product_closed_form(p: &Ex, k: &Ex, n: &Ex) -> Ex {
    let ctx = n.context();
    if !p.contains(k) {
        return p.pow(n);
    }
    let coeffs = {
        let mut inner = ctx.inner.write();
        crate::transforms::solve::symbolic_poly_coeffs(&mut inner.arena, p.raw_id(), k.raw_id())
    };
    if let Some(c) = coeffs
        && c.len() == 2
    {
        let beta = p.wrap(c[0]);
        let alpha = p.wrap(c[1]);
        let shift = (&beta / &alpha).eval();
        let alpha_pow = if alpha.is_one_structural() {
            ctx.one()
        } else {
            alpha.pow(n)
        };
        if let Some(r) = as_rational(&shift)
            && r.is_integer()
            && num_traits::Signed::is_positive(&r)
        {
            // Γ(n + m)/Γ(m) = (n + m − 1)! / (m − 1)!
            let m: i64 = r.to_integer().try_into().unwrap_or(1);
            let numer = (n + (m - 1)).factorial();
            let denom = ctx.int(m - 1).factorial().eval();
            return (&alpha_pow * &(&numer / &denom)).eval();
        }
        let g_num = (n + &shift).gamma();
        let g_den = shift.gamma();
        return (&alpha_pow * &(&g_num / &g_den)).eval();
    }
    let upper = n - 1;
    Ex::symbolic_product(p, k, &ctx.int(0), &upper)
}

/// Solve the first-order recurrence with variable coefficients
///
/// `a(n+1) = p(n)·a(n) + q(n)`,   `a(0) = a0`
///
/// via `a(n) = P(n)·(a0 + Σ_{k=0}^{n-1} q(k)/P(k+1))` with
/// `P(n) = Π_{k=0}^{n-1} p(k)`.  The product is given in closed form for
/// constant or linear `p` (powers, factorials, Gamma ratios); the sum is
/// passed through [`Ex::closed_form_sum`] and left as a formal `Sum` if no
/// closed form is known.  Without `a0`, the constant `C1` is used.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::rsolve::rsolve_first_order;
///
/// let ctx = Context::new();
/// let n = ctx.symbol("n");
/// // a(n+1) = (n+1)·a(n), a(0) = 1  →  n!
/// let sol = rsolve_first_order(&(&n + 1), &ctx.int(0), &n, Some(&ctx.int(1))).unwrap();
/// assert_eq!(format!("{sol}"), "n!");
/// ```
pub fn rsolve_first_order(p: &Ex, q: &Ex, n: &Ex, a0: Option<&Ex>) -> Result<Ex, SymplexError> {
    let _ = n.checked_id(p);
    let _ = n.checked_id(q);
    let ctx = n.context();
    if p.eval().is_zero_structural() {
        return Err(SymplexError::InvalidArgument {
            operation: "rsolve_first_order",
            reason: "p(n) must be nonzero".into(),
        });
    }
    let k = ctx.symbol("__k_rsolve");
    let p_k = p.subs(n, &k);
    let big_p = product_closed_form(&p_k, &k, n);
    let start = match a0 {
        Some(v) => {
            let _ = n.checked_id(v);
            v.clone()
        }
        None => constant(&ctx, 1),
    };
    let q_eval = q.eval();
    if q_eval.is_zero_structural() {
        return Ok((&big_p * &start).eval().simplify());
    }
    // Σ_{k=0}^{n-1} q(k) / P(k+1)
    let q_k = q.subs(n, &k);
    let p_k1 = big_p.subs(n, &(&k + 1)).eval();
    let body = (&q_k / &p_k1).eval();
    let upper = n - 1;
    let sum = Ex::symbolic_sum(&body, &k, &ctx.int(0), &upper).closed_form_sum();
    let total = (&big_p * &(&start + &sum)).eval();
    let s = total.simplify();
    Ok(if s.count_ops() <= total.count_ops() {
        s
    } else {
        total
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::context::Context;

    fn at(f: &Ex, n: &Ex, k: i64) -> f64 {
        f.subs_i64(n, k).eval_f64().unwrap()
    }

    #[test]
    fn hanoi() {
        let ctx = Context::new();
        let n = ctx.symbol("n");
        let sol = rsolve_linear(
            &[ctx.int(-2), ctx.int(1)],
            Some(&ctx.int(1)),
            &n,
            &[ctx.int(0)],
        )
        .unwrap();
        for k in 0..8 {
            assert!((at(&sol, &n, k) - (2f64.powi(k as i32) - 1.0)).abs() < 1e-9);
        }
    }

    #[test]
    fn fibonacci_binet() {
        let ctx = Context::new();
        let n = ctx.symbol("n");
        let sol = rsolve_linear(
            &[ctx.int(-1), ctx.int(-1), ctx.int(1)],
            None,
            &n,
            &[ctx.int(0), ctx.int(1)],
        )
        .unwrap();
        let (mut a, mut b) = (0.0, 1.0);
        for k in 0..12 {
            assert!(
                (at(&sol, &n, k) - a).abs() < 1e-8,
                "a({k}) = {}",
                at(&sol, &n, k)
            );
            let c = a + b;
            a = b;
            b = c;
        }
        assert!(format!("{sol}").contains("sqrt(5)"));
    }

    #[test]
    fn repeated_root() {
        // a(n+2) - 4a(n+1) + 4a(n) = 0 → (C1 + C2 n) 2^n
        let ctx = Context::new();
        let n = ctx.symbol("n");
        let sol = rsolve_linear(&[ctx.int(4), ctx.int(-4), ctx.int(1)], None, &n, &[]).unwrap();
        let s = format!("{sol}");
        assert!(
            s.contains("C1") && s.contains("C2") && s.contains("2^n"),
            "{s}"
        );
    }

    #[test]
    fn complex_roots_real_form() {
        // a(n+2) + a(n) = 0, a(0)=1, a(1)=0 → cos(nπ/2)
        let ctx = Context::new();
        let n = ctx.symbol("n");
        let sol = rsolve_linear(
            &[ctx.int(1), ctx.int(0), ctx.int(1)],
            None,
            &n,
            &[ctx.int(1), ctx.int(0)],
        )
        .unwrap();
        let expected = [1.0, 0.0, -1.0, 0.0, 1.0, 0.0];
        for (k, e) in expected.iter().enumerate() {
            assert!((at(&sol, &n, k as i64) - e).abs() < 1e-9, "a({k})");
        }
    }

    #[test]
    fn polynomial_forcing_with_resonance() {
        // a(n+1) - a(n) = n, a(0) = 0 → n(n-1)/2
        let ctx = Context::new();
        let n = ctx.symbol("n");
        let sol = rsolve_linear(&[ctx.int(-1), ctx.int(1)], Some(&n), &n, &[ctx.int(0)]).unwrap();
        for k in 0..8 {
            let kf = k as f64;
            assert!((at(&sol, &n, k) - kf * (kf - 1.0) / 2.0).abs() < 1e-9);
        }
    }

    #[test]
    fn exponential_forcing() {
        // a(n+1) - 2a(n) = 3^n, a(0) = 0 → 3^n - 2^n
        let ctx = Context::new();
        let n = ctx.symbol("n");
        let three_n = ctx.int(3).pow(&n);
        let sol = rsolve_linear(
            &[ctx.int(-2), ctx.int(1)],
            Some(&three_n),
            &n,
            &[ctx.int(0)],
        )
        .unwrap();
        for k in 0..8 {
            let e = 3f64.powi(k as i32) - 2f64.powi(k as i32);
            assert!((at(&sol, &n, k) - e).abs() < 1e-8);
        }
    }

    #[test]
    fn factorial_first_order() {
        let ctx = Context::new();
        let n = ctx.symbol("n");
        let sol = rsolve_first_order(&(&n + 1), &ctx.int(0), &n, Some(&ctx.int(1))).unwrap();
        assert_eq!(format!("{sol}"), "n!");
        let mut f = 1.0;
        for k in 0..7 {
            if k > 0 {
                f *= k as f64;
            }
            assert!((at(&sol, &n, k) - f).abs() < 1e-9);
        }
    }

    #[test]
    fn first_order_constant_with_forcing() {
        // a(n+1) = 2 a(n) + 1, a0 = 0 via the first-order path → 2^n - 1
        let ctx = Context::new();
        let n = ctx.symbol("n");
        let sol = rsolve_first_order(&ctx.int(2), &ctx.int(1), &n, Some(&ctx.int(0))).unwrap();
        for k in 0..8 {
            assert!(
                (at(&sol, &n, k) - (2f64.powi(k as i32) - 1.0)).abs() < 1e-9,
                "a({k}) = {} from {sol}",
                at(&sol, &n, k)
            );
        }
    }

    #[test]
    fn rejects_bad_input() {
        let ctx = Context::new();
        let n = ctx.symbol("n");
        assert!(rsolve_linear(&[ctx.int(1)], None, &n, &[]).is_err());
        assert!(rsolve_linear(&[ctx.int(1), ctx.int(0)], None, &n, &[]).is_err());
        assert!(
            rsolve_linear(
                &[ctx.int(1), ctx.int(1)],
                None,
                &n,
                &[ctx.int(0), ctx.int(1)]
            )
            .is_err()
        );
    }
}

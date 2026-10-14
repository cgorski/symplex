//! Compute residues of functions at poles of any order.
//!
//! The residue of `f(x)` at `x = a` is the coefficient of `1/(x−a)` in the
//! Laurent expansion of `f` around `a`.
//!
//! # Algorithm
//!
//! 1. Shift: `g(t) = f(a + t)` and split `g = N(t)/D(t)`.
//! 2. If `D` is a polynomial in `t`, its multiplicity `m` at `t = 0` is the
//!    index of the first coefficient proved nonzero, every earlier one
//!    proved 0 ([`coefficient_is_zero`]; an undecided coefficient refuses).
//!    With `D̃(t) = D(t)/tᵐ`
//!    (exact, non-vanishing at 0), the residue is the Taylor coefficient
//!    `Res = 1/(m−1)! · dᵐ⁻¹/dtᵐ⁻¹ [N(t)/D̃(t)]` at `t = 0`, which is a plain
//!    substitution — no limits needed.  This handles algebraic points such
//!    as `a = i` where polynomial cancellation over ℚ is unavailable.
//! 3. Otherwise (transcendental denominators such as `1/sin t`), the order
//!    is probed with `lim tᵏ g(t)` for `k = 1, …, 6` and the same derivative
//!    formula is evaluated as a limit.
//! 4. Both formulas presuppose a pole of finite order: `N` (and, in step 3,
//!    `D`) must be *analytic* at `t = 0`.  That is certified structurally
//!    ([`analytic_at_zero`]); a finite value `N(0)` is not enough —
//!    `t·e^{1/t}` is `0` at `t = 0` after `0·x → 0`, and `t²·sin(1/t)` has
//!    real limits although `t = 0` is an essential singularity.
//! 5. When the certificate fails, `t = 0` may be an essential singularity.
//!    If `g` is single-valued and analytic on the Riemann sphere except at
//!    `0` and finitely many poles — rational functions of `t` times entire
//!    functions (`exp`, `sin`, `cos`, `sinh`, `cosh`, `erf`, `erfc`, `c^w`)
//!    of Laurent polynomials in `1/t` ([`singular_polys_off_origin`]) —
//!    the residue theorem on the sphere gives
//!    `Res₀ g = −Res_∞ g − Σ_{p ≠ 0} Res_p g`, where the other poles `p` are
//!    the (explicitly solved) roots of the polynomial denominators and
//!    `Res_∞`, `Res_p` are residues at poles of finite order (steps 2–3).
//!    This is the coefficient of `t⁻¹` of the product of the Laurent
//!    series: for `g = P(t)·h(1/t)` with `P` a polynomial it is the finite
//!    sum `Σₙ pₙ₋₁ hₙ` (`Res(z²e^{1/z}, 0) = 1/6`, `Res(z²sin(1/z), 0) =
//!    −1/6`); for a rational `P` with other poles the infinite series sums
//!    to the closed form (`Res(e^{1/z}/(z − 1), 0) = 1 − e`).  When the
//!    series does not close this way (`e^z·e^{1/z}`: `∞` is essential too,
//!    the value is `I₁(2)`), the residue is refused.
//!
//! If nothing above succeeds, an error is returned and the caller keeps a
//! formal `Residue` node.

use num_bigint::BigInt;
use num_traits::{One, Signed, Zero};

use crate::base::arena::Arena;
use crate::base::errors::SymplexError;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::Q;
use crate::base::truth::{Evidence, SignSet, Truth};
use crate::base::walk;
use crate::calculus::calculus_util;
use crate::poly::Poly;
use crate::transforms::{diff, eval, subs};

/// Maximum pole order probed by the limit-based fallback.
const MAX_PROBE_ORDER: usize = 6;

fn failed(reason: impl Into<String>) -> SymplexError {
    SymplexError::computation_failed("residue", reason)
}

/// Compute the residue of `expr` at `var = point`.
///
/// See the module documentation for the algorithm.  Returns `Err` when the
/// residue cannot be determined (essential singularity, unknown pole
/// order, or a non-symbol `var`).
pub(crate) fn residue(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Result<ExprId, SymplexError> {
    residue_with(arena, expr, var, point, true)
}

/// [`residue`], with the essential-singularity route (module step 5)
/// enabled or not.  The route itself computes residues at poles only
/// (`essential = false`), so the mutual calls stop after one level.
fn residue_with(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    essential: bool,
) -> Result<ExprId, SymplexError> {
    if !matches!(arena.node(var), ExprNode::Symbol(_)) {
        return Err(failed("residue variable must be a symbol"));
    }
    if point == arena.infinity() {
        return residue_at_infinity_with(arena, expr, var, essential);
    }
    let result = residue_inner(arena, expr, var, point, essential)?;
    // The limit engine can leak internal dummy symbols into a half-finished
    // result; a residue may only mention symbols of the input.
    let mut allowed: Vec<ExprId> = walk::free_symbols(arena, expr);
    allowed.extend(walk::free_symbols(arena, point));
    if walk::free_symbols(arena, result)
        .iter()
        .any(|s| !allowed.contains(s))
    {
        return Err(failed("limit engine returned an incomplete result"));
    }
    Ok(result)
}

/// [`residue`] without the final symbol-leak check.
fn residue_inner(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    essential: bool,
) -> Result<ExprId, SymplexError> {
    // Shift the pole to the origin: g(t) = f(point + t).
    let t = arena.symbol("__res_t");
    let shifted_var = arena.add(&[point, t]);
    let g = subs::subs(arena, expr, var, shifted_var);
    let g = eval::eval(arena, g);

    if !walk::contains(arena, g, t) {
        // Constant in t: analytic, residue 0 (unless it is itself non-finite).
        return Ok(arena.zero());
    }

    match residue_at_pole(arena, g, t) {
        Err(pole_err) if essential => essential_residue(arena, g, t).map_err(|e| {
            failed(format!(
                "{}; as an essential singularity: {}",
                pole_reason(&pole_err),
                pole_reason(&e)
            ))
        }),
        other => other,
    }
}

/// The reason text of a [`failed`] error (the whole message otherwise).
fn pole_reason(e: &SymplexError) -> String {
    match e {
        SymplexError::ComputationFailed { reason, .. } => reason.clone(),
        other => other.to_string(),
    }
}

/// Residue of `g` at `t = 0` when it is a pole of finite order (module
/// steps 2–4); an error for anything else.
fn residue_at_pole(arena: &mut Arena, g: ExprId, t: ExprId) -> Result<ExprId, SymplexError> {
    // Clear nested fractions (e.g. from f(1/t)) so that the denominator is a
    // genuine polynomial whenever f is rational.
    let g = crate::transforms::integrate::clear_nested_fractions(arena, g, t);
    let (numer, denom) = crate::poly::polybridge::as_numer_denom(arena, g);

    if let Some(coeffs) = calculus_util::poly_coeffs_symbolic(arena, denom, t) {
        return residue_polynomial_denominator(arena, numer, &coeffs, t);
    }

    // The real-axis limits below determine the order of a *pole*; they say
    // nothing about an essential singularity (`sin(1/t)` is bounded on the
    // real axis).  Require a quotient of functions analytic at 0.
    if !analytic_at_zero(arena, numer, t) || !analytic_at_zero(arena, denom, t) {
        return Err(failed(
            "not a quotient of functions analytic at the point (essential singularity or branch point?)",
        ));
    }
    residue_by_limits(arena, g, t)
}

/// Residue when the shifted denominator is a polynomial in `t` with the
/// given coefficients.
fn residue_polynomial_denominator(
    arena: &mut Arena,
    numer: ExprId,
    coeffs: &[ExprId],
    t: ExprId,
) -> Result<ExprId, SymplexError> {
    // Multiplicity of t = 0: index of the first coefficient proved nonzero,
    // every earlier one proved 0.
    let mut m = coeffs.len();
    for (k, &c) in coeffs.iter().enumerate() {
        match coefficient_is_zero(arena, c) {
            Truth::True => {}
            Truth::False => {
                m = k;
                break;
            }
            Truth::Unknown => {
                return Err(failed(format!(
                    "cannot decide whether the coefficient of t^{k} of the denominator is 0"
                )));
            }
        }
    }
    if m == coeffs.len() {
        return Err(failed("denominator is identically zero"));
    }

    // Numerator must be analytic at t = 0 (a finite value N(0) is not
    // enough: t·e^{1/t} evaluates to 0 there).
    let n0 = subs::subs(arena, numer, t, arena.zero());
    let n0 = eval::eval(arena, n0);
    if !is_finite_constant(arena, n0, t) || !analytic_at_zero(arena, numer, t) {
        return Err(failed(
            "numerator is not analytic at the point (essential singularity or branch point?)",
        ));
    }

    if m == 0 {
        // No pole: the function is analytic here.
        return Ok(arena.zero());
    }

    // D̃(t) = Σ_{k ≥ m} c_k t^{k−m}
    let mut terms: Vec<ExprId> = Vec::with_capacity(coeffs.len() - m);
    for (k, &c) in coeffs.iter().enumerate().skip(m) {
        let e = arena.int((k - m) as i64);
        let tp = arena.pow(t, e);
        terms.push(arena.mul(&[c, tp]));
    }
    let d_tilde = arena.add(&terms);

    // h(t) = N(t)/D̃(t); Res = h^{(m−1)}(0) / (m−1)!
    let mut h = arena.div(numer, d_tilde);
    for _ in 1..m {
        h = diff::diff(arena, h, t);
    }
    let h0 = subs::subs(arena, h, t, arena.zero());
    let h0 = eval::eval(arena, h0);
    if !is_finite_constant(arena, h0, t) {
        return Err(failed("could not evaluate the Laurent coefficient"));
    }
    let res = divide_by_factorial(arena, h0, m - 1);
    Ok(eval::eval(arena, res))
}

/// Limit-based fallback for non-polynomial denominators.
fn residue_by_limits(arena: &mut Arena, g: ExprId, t: ExprId) -> Result<ExprId, SymplexError> {
    let zero = arena.zero();
    for m in 1..=MAX_PROBE_ORDER {
        let e = arena.int(m as i64);
        let tm = arena.pow(t, e);
        let scaled = arena.mul(&[tm, g]);
        let scaled = eval::eval(arena, scaled);
        let Ok(l) = crate::calculus::limit::limit(arena, scaled, t, zero) else {
            continue;
        };
        if !is_finite_constant(arena, l, t) {
            continue;
        }
        if m == 1 {
            return Ok(eval::eval(arena, l));
        }
        // Order ≤ m established: Res = lim d^{m−1}/dt^{m−1}[tᵐ g] / (m−1)!
        let mut h = scaled;
        for _ in 1..m {
            h = diff::diff(arena, h, t);
        }
        let h = eval::eval(arena, h);
        let l2 = crate::calculus::limit::limit(arena, h, t, zero)?;
        if !is_finite_constant(arena, l2, t) {
            return Err(failed("Laurent coefficient limit is not finite"));
        }
        let res = divide_by_factorial(arena, l2, m - 1);
        return Ok(eval::eval(arena, res));
    }
    Err(failed(
        "could not determine the pole order (essential singularity or order > 6)",
    ))
}

/// Residue at an essential singularity at `t = 0` by the residue theorem
/// on the Riemann sphere (module step 5):
/// `Res₀ g = −Res_∞ g − Σ_{p ≠ 0} Res_p g`.
///
/// Valid because [`singular_polys_off_origin`] certifies that `g` is
/// analytic on `ℂ ∖ {0}` except at the roots of the returned polynomials,
/// all of which are found explicitly ([`nonzero_roots`]); the residues at
/// those roots and at `∞` are computed with the pole formulas only, which
/// refuse anything but a pole of finite order (so a second essential
/// singularity, at `∞` or elsewhere, makes the whole computation refuse).
fn essential_residue(arena: &mut Arena, g: ExprId, t: ExprId) -> Result<ExprId, SymplexError> {
    let polys = singular_polys_off_origin(arena, g, t).ok_or_else(|| {
        failed("not a rational function times entire functions of polynomials in 1/(z − z0)")
    })?;
    let roots = nonzero_roots(arena, &polys, t)
        .ok_or_else(|| failed("the other poles could not all be found exactly"))?;
    let mut parts: Vec<ExprId> = Vec::with_capacity(roots.len() + 1);
    parts.push(residue_at_infinity_with(arena, g, t, false)?);
    for p in roots {
        parts.push(residue_inner(arena, g, t, p, false)?);
    }
    let sum = arena.add(&parts);
    let neg = arena.neg(sum);
    let res = eval::eval(arena, neg);
    if !is_finite_constant(arena, res, t) {
        return Err(failed(
            "the residues of the other singularities are not finite",
        ));
    }
    Ok(res)
}

/// Certify that `g` is single-valued and analytic on `ℂ ∖ {0}` except at
/// the roots of finitely many polynomials in `t`, and return those
/// polynomials (the bases of negative integer powers).
///
/// Every subexpression depending on `t` is checked with an explicit stack:
/// sums, products and non-negative integer powers; the entire functions
/// `exp`, `sin`, `cos`, `sinh`, `cosh`, `erf`, `erfc` and `c^w` (`c ≠ 0`
/// free of `t`); negative integer powers of products of polynomials in `t`
/// (poles at their roots) and of exponentials (which never vanish).
/// Anything else — fractional powers of `t`, `ln`, `tan`, `1/(e^{1/t} − 1)`
/// (poles accumulating at 0) — gives `None`.
fn singular_polys_off_origin(arena: &mut Arena, g: ExprId, t: ExprId) -> Option<Vec<ExprId>> {
    let mut polys: Vec<ExprId> = Vec::new();
    let mut stack = vec![g];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) || !walk::contains(arena, id, t) {
            continue;
        }
        let node = arena.node(id).clone();
        let ok = match node {
            ExprNode::Symbol(_) | ExprNode::Add(_) | ExprNode::Mul(_) | ExprNode::Neg(_) => true,
            ExprNode::Exp(_)
            | ExprNode::Sin(_)
            | ExprNode::Cos(_)
            | ExprNode::Sinh(_)
            | ExprNode::Cosh(_)
            | ExprNode::Erf(_)
            | ExprNode::Erfc(_) => true,
            ExprNode::Pow(b, x) => {
                let int_exp = arena.as_num(x).filter(|q| q.is_integer()).cloned();
                match int_exp {
                    Some(q) if !num_traits::Signed::is_negative(&q) => true,
                    Some(_) => {
                        let factors: Vec<ExprId> = match arena.node(b) {
                            ExprNode::Mul(fs) => fs.to_vec(),
                            _ => vec![b],
                        };
                        let mut all = true;
                        for f in factors {
                            if !walk::contains(arena, f, t)
                                || matches!(arena.node(f), ExprNode::Exp(_))
                            {
                                continue;
                            }
                            if calculus_util::poly_coeffs_symbolic(arena, f, t).is_some() {
                                polys.push(f);
                            } else {
                                all = false;
                                break;
                            }
                        }
                        all
                    }
                    // c^w = exp(w·ln c) is entire in w for a constant c ≠ 0.
                    None => {
                        !walk::contains(arena, b, t) && {
                            let pv = point_value(arena, b);
                            pv.nonzero
                        }
                    }
                }
            }
            _ => false,
        };
        if !ok {
            return None;
        }
        node.for_each_child(|c| stack.push(c));
    }
    Some(polys)
}

/// All distinct nonzero roots of the polynomials `polys` in `t`, or `None`
/// unless they are certainly all found.
///
/// Polynomials with rational coefficients are multiplied together and made
/// square-free, so a root shared by two factors or repeated is counted
/// once; the solver's distinct roots are complete exactly when their
/// number is the degree.  A polynomial with symbolic coefficients (at most
/// one) must have as many distinct roots found as its degree.
fn nonzero_roots(arena: &mut Arena, polys: &[ExprId], t: ExprId) -> Option<Vec<ExprId>> {
    use crate::poly::Poly;
    let mut rational = Poly::from_coeffs(vec![num_rational::Ratio::one()]);
    let mut symbolic: Vec<ExprId> = Vec::new();
    for &p in polys {
        let coeffs = calculus_util::poly_coeffs_symbolic(arena, p, t)?;
        // Divide out the root at 0: p̃(t) = p(t)/t^m, with p̃(0) proved
        // nonzero.
        let mut m = None;
        for (k, &c) in coeffs.iter().enumerate() {
            match coefficient_is_zero(arena, c) {
                Truth::True => {}
                Truth::False => {
                    m = Some(k);
                    break;
                }
                Truth::Unknown => return None,
            }
        }
        let m = m?;
        let tail = &coeffs[m..];
        if tail.len() <= 1 {
            continue;
        }
        let nums: Option<Vec<num_rational::Ratio<BigInt>>> =
            tail.iter().map(|&c| arena.as_num(c).cloned()).collect();
        if let Some(nums) = nums {
            rational = rational.mul(&Poly::from_coeffs(nums));
        } else {
            let mut terms: Vec<ExprId> = Vec::with_capacity(tail.len());
            for (k, &c) in tail.iter().enumerate() {
                let e = arena.int(k as i64);
                let tk = arena.pow(t, e);
                terms.push(arena.mul(&[c, tk]));
            }
            let q = arena.add(&terms);
            if !symbolic.contains(&q) {
                symbolic.push(q);
            }
        }
    }
    if symbolic.len() > 1 {
        return None;
    }
    let mut roots: Vec<ExprId> = Vec::new();
    let mut solve_all = |arena: &mut Arena, q: ExprId, degree: usize| -> Option<()> {
        let found = match crate::transforms::solve::solve_classified_real(arena, q, t)
            .points_if_complete()
        {
            Some(found) if found.len() == degree => found,
            _ => return None,
        };
        for s in found {
            let v = eval::eval(arena, s);
            if arena.is_zero_structural(v) || walk::contains(arena, v, t) {
                return None;
            }
            if !roots.contains(&v) {
                roots.push(v);
            }
        }
        Some(())
    };
    let sqf = rational.square_free_part();
    let degree = sqf.degree().unwrap_or(0);
    if degree > 0 {
        let q = crate::poly::polybridge::poly_to_expr(arena, &sqf, t);
        solve_all(arena, q, degree)?;
    }
    if let Some(&q) = symbolic.first() {
        let degree = calculus_util::poly_coeffs_symbolic(arena, q, t)?.len() - 1;
        solve_all(arena, q, degree)?;
    }
    Some(roots)
}

/// Residue at infinity: `Res_{z=∞} f = −Res_{t=0} f(1/t)/t²`.
pub(crate) fn residue_at_infinity(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
) -> Result<ExprId, SymplexError> {
    residue_at_infinity_with(arena, expr, var, true)
}

/// [`residue_at_infinity`] with the essential-singularity route enabled or
/// not (see [`residue_with`]).
fn residue_at_infinity_with(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    essential: bool,
) -> Result<ExprId, SymplexError> {
    if !matches!(arena.node(var), ExprNode::Symbol(_)) {
        return Err(failed("residue variable must be a symbol"));
    }
    let t = arena.symbol("__res_inf_t");
    let one = arena.one();
    let inv_t = arena.div(one, t);
    let f_inv = subs::subs(arena, expr, var, inv_t);
    let neg_two = arena.int(-2);
    let t_m2 = arena.pow(t, neg_two);
    let h = arena.mul(&[f_inv, t_m2]);
    let h = eval::eval(arena, h);
    let zero = arena.zero();
    let r = residue_with(arena, h, t, zero, essential)?;
    let neg = arena.neg(r);
    Ok(eval::eval(arena, neg))
}

// ═══════════════════════════════════════════════════════════════════════════
// Residue sums of rational functions (contour integrals)
// ═══════════════════════════════════════════════════════════════════════════

/// Where a pole lies relative to the real axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PolePlace {
    /// `Im ρ > 0`.
    Upper,
    /// `Im ρ < 0`.
    Lower,
    /// `ρ` real.
    Real,
}

/// A rational function `N/D` with `N, D ∈ ℚ[x]`, `D` not constant.
pub(crate) struct RationalQ {
    /// The numerator `N`.
    pub(crate) numer: Poly,
    /// The denominator `D`.
    pub(crate) denom: Poly,
}

impl RationalQ {
    /// `deg D − deg N` (`N = 0` counts as degree `−∞`, i.e. a large value).
    pub(crate) fn degree_gap(&self) -> i64 {
        let dd = self.denom.degree().map_or(0, |d| d as i64);
        match self.numer.degree() {
            Some(dn) => dd - dn as i64,
            None => i64::MAX / 4,
        }
    }

    /// Is `N/D` an even or an odd function (`None`: neither)?  (An
    /// `Option<bool>` up to 0.49, whose `false` meant "odd".)
    pub(crate) fn parity(&self) -> Option<Parity> {
        let par = |p: &Poly| -> Option<Parity> {
            let even = p.coeffs().iter().skip(1).step_by(2).all(Zero::is_zero);
            let odd = p.coeffs().iter().step_by(2).all(Zero::is_zero);
            match (even, odd) {
                (true, _) => Some(Parity::Even),
                (false, true) => Some(Parity::Odd),
                _ => None,
            }
        };
        if self.numer.is_zero() {
            return Some(Parity::Even);
        }
        Some(if par(&self.numer)? == par(&self.denom)? {
            Parity::Even
        } else {
            Parity::Odd
        })
    }
}

/// Even (`f(−x) = f(x)`) or odd (`f(−x) = −f(x)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Parity {
    /// `f(−x) = f(x)`.
    Even,
    /// `f(−x) = −f(x)`.
    Odd,
}

/// Largest total degree of a rational function [`rational_over_q`]
/// expands (it guards `(x + 1)¹⁰⁰⁰` from being multiplied out).
const MAX_RATIONAL_DEGREE: u64 = 40;

/// A bound on the degree in `var` of the numerator and denominator of
/// `e` (sums, products, integer powers); `None` above `u64` or for a
/// non-integer power.
fn rational_degree_bound(arena: &Arena, e: ExprId, var: ExprId) -> Option<u64> {
    let mut deg: rustc_hash::FxHashMap<ExprId, u64> = rustc_hash::FxHashMap::default();
    for id in walk::post_order_ids(arena, e) {
        let d = if id == var {
            1
        } else {
            match arena.node(id) {
                ExprNode::Add(ch) => ch.iter().try_fold(0u64, |acc, c| {
                    acc.checked_add(deg.get(c).copied().unwrap_or(0))
                })?,
                ExprNode::Mul(ch) => ch.iter().try_fold(0u64, |acc, c| {
                    acc.checked_add(deg.get(c).copied().unwrap_or(0))
                })?,
                ExprNode::Neg(c) => deg.get(c).copied().unwrap_or(0),
                ExprNode::Pow(b, k) => {
                    let k = arena.as_num(*k).filter(|q| q.is_integer())?;
                    let k = u64::try_from(k.to_integer().magnitude()).ok()?;
                    deg.get(b).copied().unwrap_or(0).checked_mul(k)?
                }
                _ => 0,
            }
        };
        deg.insert(id, d);
    }
    deg.get(&e).copied()
}

/// `r` as `N/D` with `N, D ∈ ℚ[x]` and `D` not constant (`None` for
/// anything else, including symbolic coefficients and degrees above
/// [`MAX_RATIONAL_DEGREE`]).
pub(crate) fn rational_over_q(arena: &mut Arena, r: ExprId, var: ExprId) -> Option<RationalQ> {
    if rational_degree_bound(arena, r, var)? > MAX_RATIONAL_DEGREE {
        return None;
    }
    let (n, d) = crate::poly::polybridge::as_numer_denom(arena, r);
    let n = crate::transforms::expand::expand(arena, n);
    let n = eval::eval(arena, n);
    let d = crate::transforms::expand::expand(arena, d);
    let d = eval::eval(arena, d);
    let numer = crate::poly::polybridge::expr_to_poly(arena, n, var)?;
    let denom = crate::poly::polybridge::expr_to_poly(arena, d, var)?;
    if denom.degree()? == 0 {
        return None;
    }
    let g = Poly::gcd(&numer, &denom);
    if g.degree().is_some_and(|k| k > 0) {
        let numer = numer.div_rem(&g).0;
        let denom = denom.div_rem(&g).0;
        return Some(RationalQ { numer, denom });
    }
    Some(RationalQ { numer, denom })
}

/// One distinct pole `ρ` of a rational function: a root of the
/// irreducible factor `factor` of the denominator `D = factor^mult ·
/// cofactor`.
struct RationalPole {
    factor: Poly,
    cofactor: Poly,
    mult: usize,
    place: PolePlace,
    /// The sign of `Re ρ`: one atom, exact or from the certified disk, or
    /// `UNKNOWN` when the disk does not decide ([`ball_re_signs`]).
    re_sign: SignSet,
    /// `ρ` written out: a rational, a quadratic surd, or `RootOf(factor, k)`.
    root: ExprId,
}

/// The sign of a rational as a single atom.
fn sign_of_q(q: &Q) -> SignSet {
    if q.is_positive() {
        SignSet::POSITIVE
    } else if q.is_negative() {
        SignSet::NEGATIVE
    } else {
        SignSet::ZERO
    }
}

/// The sign of the real part of each root of the irreducible `g` (degree
/// ≥ 3) from its certified disk `balls[k]`: decided when `|Re|` exceeds
/// the radius.  The real parts left undecided are exactly 0 when `g` is
/// even, `g(t) = h(t²)`, and their number is twice that of the negative
/// roots of `h` (Sturm): an irreducible `g` with a root `iy` also has
/// `−iy = conj(iy)`, so `g(−t) = ±g(t)` and `g` is even, with `(iy)² =
/// −y²` a negative root of `h`; and a disk always contains its root.
/// Otherwise they stay `UNKNOWN`.
fn ball_re_signs(g: &Poly, balls: &[crate::poly::roots::RootBall]) -> Vec<SignSet> {
    let mut out = Vec::with_capacity(balls.len());
    let mut undecided = 0usize;
    for ball in balls {
        let decided = ball
            .radius
            .as_ref()
            .is_some_and(|r| ball.value.0.abs().cmp(r).unwrap_or(-1) > 0);
        if decided {
            out.push(
                if crate::base::bigfloat::is_strictly_positive(&ball.value.0) {
                    SignSet::POSITIVE
                } else {
                    SignSet::NEGATIVE
                },
            );
        } else {
            out.push(SignSet::UNKNOWN);
            undecided += 1;
        }
    }
    let even = g.coeffs().iter().skip(1).step_by(2).all(Zero::is_zero);
    if undecided > 0 && even {
        let h = Poly::from_coeffs(g.coeffs().iter().step_by(2).cloned().collect());
        let lo = -crate::poly::sturm::cauchy_bound(&h) - Q::one();
        let negative = crate::poly::sturm::SturmChain::new(&h).count_roots_in(&lo, &Q::zero());
        if 2 * negative == undecided {
            for s in &mut out {
                if *s == SignSet::UNKNOWN {
                    *s = SignSet::ZERO;
                }
            }
        }
    }
    out
}

/// Precision (bits) of the root balls that place the poles.
const POLE_PREC: usize = 128;

/// Largest degree of the denominator whose poles [`rational_poles`] places.
const MAX_POLE_DEGREE: usize = 24;

/// Every distinct pole of `1/D`, placed relative to the real axis.
///
/// `D` is factored over ℤ.  A linear factor has its rational root and an
/// irreducible quadratic `a·x² + b·x + c` its two roots in radicals,
/// `(−b ± √(b² − 4ac))/(2a)`, placed by the sign of the discriminant and of
/// `a`.  The roots of a factor of degree ≥ 3 are `RootOf(factor, k)`,
/// placed by the certified inclusion disks that `RootOf` is evaluated with
/// (the same calls define the index order): a disk certified real, or one
/// whose imaginary part exceeds its radius.  `None` when a disk is not
/// certified, when the number of real disks differs from the Sturm count,
/// or above [`MAX_POLE_DEGREE`].  `bound` is the variable of the `RootOf`
/// polynomials.  The sign of each real part is exact for degrees 1 and 2
/// and certified by the disks above ([`ball_re_signs`]).
fn rational_poles(arena: &mut Arena, d: &Poly, bound: ExprId) -> Option<Vec<RationalPole>> {
    if d.degree()? > MAX_POLE_DEGREE {
        return None;
    }
    let (_, factors) = d.factor_over_z();
    let mut out = Vec::new();
    for (factor, mult) in &factors {
        let deg = factor.degree()?;
        if deg == 0 {
            continue;
        }
        let mult = *mult as usize;
        let cofactor = {
            let (q, r) = d.div_rem(&factor.pow(mult));
            if !r.is_zero() {
                return None;
            }
            q
        };
        match deg {
            1 => {
                let rho = -factor.coeff(0) / factor.coeff(1);
                let re_sign = sign_of_q(&rho);
                let root = arena.num_ratio(rho);
                out.push(RationalPole {
                    factor: factor.clone(),
                    cofactor,
                    mult,
                    place: PolePlace::Real,
                    re_sign,
                    root,
                });
            }
            2 => {
                let (a, b, c) = (factor.coeff(2), factor.coeff(1), factor.coeff(0));
                let disc = &b * &b - Q::from_integer(BigInt::from(4)) * &a * &c;
                let two_a = Q::from_integer(BigInt::from(2)) * &a;
                let centre_q = -b / &two_a;
                // Real roots: of opposite signs when their product `c/a` is
                // negative (the `+` one is the larger for `a > 0`), else
                // both of the sign of their sum `2·centre`.
                let product = &c / &a;
                let centre = arena.num_ratio(centre_q.clone());
                let real = disc.is_positive();
                let r = arena.num_ratio(if real { disc.clone() } else { -disc.clone() });
                let root_disc = arena.sqrt(r);
                let half = arena.num_ratio(Q::one() / &two_a);
                let offset = if real {
                    arena.mul(&[half, root_disc])
                } else {
                    let i = arena.i_unit();
                    arena.mul(&[half, root_disc, i])
                };
                for plus in [true, false] {
                    let signed = if plus { offset } else { arena.neg(offset) };
                    let root = arena.add(&[centre, signed]);
                    let root = eval::eval(arena, root);
                    let place = if real {
                        PolePlace::Real
                    } else if plus == two_a.is_positive() {
                        PolePlace::Upper
                    } else {
                        PolePlace::Lower
                    };
                    let re_sign = if real && product.is_negative() {
                        if plus == two_a.is_positive() {
                            SignSet::POSITIVE
                        } else {
                            SignSet::NEGATIVE
                        }
                    } else {
                        sign_of_q(&centre_q)
                    };
                    out.push(RationalPole {
                        factor: factor.clone(),
                        cofactor: cofactor.clone(),
                        mult,
                        place,
                        re_sign,
                        root,
                    });
                }
            }
            _ => {
                let balls = crate::poly::roots::certified_roots(factor, POLE_PREC);
                if balls.len() != deg
                    || balls.iter().filter(|b| b.real).count()
                        != crate::poly::sturm::SturmChain::new(factor).count_real_roots()
                {
                    return None;
                }
                let mut places = Vec::with_capacity(deg);
                for ball in &balls {
                    let radius = ball.radius.as_ref()?;
                    places.push(if ball.real {
                        PolePlace::Real
                    } else if ball.value.1.abs().cmp(radius).unwrap_or(-1) > 0 {
                        if crate::base::bigfloat::is_strictly_positive(&ball.value.1) {
                            PolePlace::Upper
                        } else {
                            PolePlace::Lower
                        }
                    } else {
                        return None;
                    });
                }
                let re_signs = ball_re_signs(factor, &balls);
                if let Some(explicit) = explicit_roots(arena, factor, bound, &places, &re_signs) {
                    for (root, place, re_sign) in explicit {
                        out.push(RationalPole {
                            factor: factor.clone(),
                            cofactor: cofactor.clone(),
                            mult,
                            place,
                            re_sign,
                            root,
                        });
                    }
                    continue;
                }
                let g_expr = crate::poly::polybridge::poly_to_expr(arena, factor, bound);
                for (k, (&place, &re_sign)) in places.iter().zip(&re_signs).enumerate() {
                    let idx = arena.int(i64::try_from(k).ok()?);
                    let root = arena.intern(ExprNode::RootOf(g_expr, bound, idx));
                    out.push(RationalPole {
                        factor: factor.clone(),
                        cofactor: cofactor.clone(),
                        mult,
                        place,
                        re_sign,
                        root,
                    });
                }
            }
        }
    }
    Some(out)
}

/// Largest irreducible factor whose roots [`explicit_roots`] asks `solve`
/// for (Cardano, Ferrari).
const MAX_EXPLICIT_ROOT_DEGREE: usize = 4;

/// Largest root (nodes written out) used explicitly; others stay `RootOf`.
const MAX_EXPLICIT_ROOT_SIZE: usize = 48;

/// The sign of the real constant `e` (a real or an imaginary part of a
/// root): from certified digits, or 0 when proved
/// ([`calculus_util::hidden_zero`]); `None` otherwise.
fn certified_part_sign(arena: &mut Arena, e: ExprId) -> Option<SignSet> {
    let e = eval::eval(arena, e);
    if arena.is_zero_structural(e) {
        return Some(SignSet::ZERO);
    }
    let v = crate::transforms::evalf::evalf_verdict(
        arena,
        e,
        20,
        crate::transforms::evalf::ZeroSearch::Cap,
    )
    .ok()?;
    if let Some(z) = v.proven() {
        return Some(if z.0.is_zero() && z.1.is_zero() {
            SignSet::ZERO
        } else if !z.1.is_zero() {
            return None;
        } else if crate::base::bigfloat::is_strictly_positive(&z.0) {
            SignSet::POSITIVE
        } else {
            SignSet::NEGATIVE
        });
    }
    calculus_util::hidden_zero(arena, e)
        .proven()
        .filter(|t| t.is_true())
        .map(|_| SignSet::ZERO)
}

/// The roots of the irreducible `g` (degree 3 or 4) in radicals with
/// their places, when `solve` writes every one of them in at most
/// [`MAX_EXPLICIT_ROOT_SIZE`] nodes (`x⁴ + 1`: `±√2/2 ± i√2/2`).  Each root
/// is placed by the certified signs of its imaginary and real parts
/// ([`certified_part_sign`]; a part that is neither certified nor proved 0
/// refuses), and the places and signs must agree in number with those of
/// the disks, `certified` and `certified_re` ([`ball_re_signs`]).  Up to
/// 0.50 a root was placed by its `f64` value (`|Im| ≤ 10⁻¹²|ρ|`: real).
fn explicit_roots(
    arena: &mut Arena,
    g: &Poly,
    bound: ExprId,
    certified: &[PolePlace],
    certified_re: &[SignSet],
) -> Option<Vec<(ExprId, PolePlace, SignSet)>> {
    let deg = g.degree()?;
    if deg > MAX_EXPLICIT_ROOT_DEGREE || certified_re.iter().any(|s| !s.is_single()) {
        return None;
    }
    let g_expr = crate::poly::polybridge::poly_to_expr(arena, g, bound);
    let roots = match crate::transforms::solve::solve_classified_real(arena, g_expr, bound)
        .points_if_complete()
    {
        Some(roots) if roots.len() == deg => roots,
        _ => return None,
    };
    let mut out = Vec::with_capacity(deg);
    for r in roots {
        let ids = walk::post_order_ids(arena, r);
        if ids.len() > MAX_EXPLICIT_ROOT_SIZE
            || ids.iter().any(|&id| {
                matches!(
                    arena.node(id),
                    ExprNode::RootOf(..) | ExprNode::Symbol(_) | ExprNode::RootSum(..)
                )
            })
        {
            return None;
        }
        let im = arena.im(r);
        let place = match certified_part_sign(arena, im)? {
            SignSet::ZERO => PolePlace::Real,
            SignSet::POSITIVE => PolePlace::Upper,
            _ => PolePlace::Lower,
        };
        let re = arena.re(r);
        let re_sign = certified_part_sign(arena, re)?;
        out.push((r, place, re_sign));
    }
    let count =
        |v: &mut dyn Iterator<Item = PolePlace>, p: PolePlace| v.filter(|&q| q == p).count();
    for p in [PolePlace::Upper, PolePlace::Lower, PolePlace::Real] {
        if count(&mut out.iter().map(|e| e.1), p) != count(&mut certified.iter().copied(), p) {
            return None;
        }
    }
    for s in [SignSet::NEGATIVE, SignSet::ZERO, SignSet::POSITIVE] {
        if out.iter().filter(|e| e.2 == s).count()
            != certified_re.iter().filter(|&&c| c == s).count()
        {
            return None;
        }
    }
    Some(out)
}

/// `a·b` in `ℚ[t]/(g)`.
fn field_mul(a: &Poly, b: &Poly, g: &Poly) -> Poly {
    a.mul(b).rem(g)
}

/// `a⁻¹` in `ℚ[t]/(g)` (`g` irreducible); `None` for `a ≡ 0`.
fn field_inv(a: &Poly, g: &Poly) -> Option<Poly> {
    let e = Poly::extended_gcd(a, g);
    if e.gcd.degree() != Some(0) {
        return None;
    }
    let c = e.gcd.coeff(0);
    if c.is_zero() {
        return None;
    }
    Some(e.x.scale(&(Q::one() / c)).rem(g))
}

/// The Taylor coefficients `p⁽ʲ⁾(t)/j!` (`j < count`) of `p(t + h)` in
/// `h`, reduced modulo `g`.
fn shifted_coeffs(p: &Poly, g: &Poly, count: usize) -> Vec<Poly> {
    let mut out = Vec::with_capacity(count);
    let mut der = p.clone();
    let mut fact = Q::one();
    for j in 0..count {
        if j > 0 {
            der = der.derivative();
            fact *= Q::from_integer(BigInt::from(j));
        }
        out.push(der.scale(&(Q::one() / &fact)).rem(g));
    }
    out
}

/// Truncated product of two power series over `ℚ[t]/(g)`.
fn series_mul(a: &[Poly], b: &[Poly], g: &Poly, order: usize) -> Vec<Poly> {
    (0..order)
        .map(|n| {
            let mut acc = Poly::zero();
            for k in 0..=n {
                if let (Some(x), Some(y)) = (a.get(k), b.get(n - k)) {
                    acc = acc.add(&field_mul(x, y, g));
                }
            }
            acc.rem(g)
        })
        .collect()
}

/// Truncated reciprocal of a power series over `ℚ[t]/(g)`; `None` when
/// the constant term is `0`.
fn series_inv(a: &[Poly], g: &Poly, order: usize) -> Option<Vec<Poly>> {
    let inv0 = field_inv(a.first()?, g)?;
    let mut out: Vec<Poly> = vec![inv0.clone()];
    for n in 1..order {
        let mut acc = Poly::zero();
        for k in 1..=n {
            if let Some(x) = a.get(k) {
                acc = acc.add(&field_mul(x, &out[n - k], g));
            }
        }
        out.push(field_mul(&acc.neg(), &inv0, g));
    }
    Some(out)
}

/// The principal-part coefficients at a pole `ρ` of multiplicity `m` of
/// `N/D`, `D = g^m·E`: the Taylor coefficients `A₀, …, A_{m−1}` of
/// `A(h) = N(ρ + h)/(E(ρ + h)·G(h)^m)` with `g(ρ + h) = h·G(h)`, so that
/// `N/D = Σ Aₗ h^{l−m} + O(1)`.  Each `Aₗ` is an element of `ℚ(ρ) =
/// ℚ[t]/(g)`, written as a polynomial in `t` of degree `< deg g`.
fn principal_part(numer: &Poly, pole: &RationalPole) -> Option<Vec<Poly>> {
    let g = &pole.factor;
    let m = pole.mult;
    let n_s = shifted_coeffs(numer, g, m);
    let e_s = shifted_coeffs(&pole.cofactor, g, m);
    let g_s: Vec<Poly> = shifted_coeffs(g, g, m + 1).into_iter().skip(1).collect();
    let mut denom = e_s;
    for _ in 0..m {
        denom = series_mul(&denom, &g_s, g, m);
    }
    let inv = series_inv(&denom, g, m)?;
    Some(series_mul(&n_s, &inv, g, m))
}

/// `Σ Res_{z=ρ} [N(z)/D(z) · W(z)]` over the poles `ρ` of `N/D` whose
/// place `select` accepts, for an analytic weight `W` (an expression in
/// `z`, analytic at every selected pole; `e^{ikz}` for Fourier integrals,
/// `(−z)^{s−1}` for Mellin's, `ln²(−z)` for a keyhole).
///
/// At a pole of multiplicity `m`, `Res = Σₗ Aₗ(ρ)·W^{(m−1−l)}(ρ)/(m−1−l)!`
/// with the exact principal-part coefficients `Aₗ ∈ ℚ(ρ)`
/// ([`principal_part`]) — at a simple pole `N(ρ)/D′(ρ)·W(ρ)`, reduced to a
/// polynomial in `ρ` of degree below that of its minimal polynomial (`1/(2i)
/// = −i/2`).  `None` when a pole cannot be placed ([`rational_poles`]).
pub(crate) fn rational_residue_sum(
    arena: &mut Arena,
    r: &RationalQ,
    weight: ExprId,
    z: ExprId,
    select: impl Fn(PolePlace) -> bool,
) -> Option<ExprId> {
    let bound = pole_symbol(arena, weight);
    rational_residue_sum_with(arena, r, z, bound, |_, pole| {
        Some(select(pole.place).then_some(weight))
    })
}

/// What [`rational_residue_sum_with`] tells its weight callback about a
/// pole.
pub(crate) struct PoleView {
    /// Where the pole lies relative to the real axis (certified).
    pub(crate) place: PolePlace,
    /// The sign of its real part: one atom (exact or certified), or
    /// `UNKNOWN`.  Up to 0.50 the callback got a 16-digit value instead,
    /// and the Laplace transform decided `Re ρ ≤ 0` by `Re ρ ≤ 10⁻⁹|ρ|`.
    pub(crate) re_sign: SignSet,
}

/// [`rational_residue_sum`] with a weight chosen per pole: `weight_for`
/// returns `Some(Some(W))` to add `Res_ρ[N/D·W]`, `Some(None)` to skip the
/// pole and `None` to give up.  `bound` names the variable of the `RootOf`
/// nodes ([`pole_symbol`]).
pub(crate) fn rational_residue_sum_with(
    arena: &mut Arena,
    r: &RationalQ,
    z: ExprId,
    bound: ExprId,
    mut weight_for: impl FnMut(&mut Arena, &PoleView) -> Option<Option<ExprId>>,
) -> Option<ExprId> {
    let poles = rational_poles(arena, &r.denom, bound)?;
    // W^{(j)}/j! as expressions in z, per weight.
    let mut derivatives: Vec<(ExprId, Vec<ExprId>)> = Vec::new();
    let mut terms: Vec<ExprId> = Vec::new();
    for pole in &poles {
        let view = PoleView {
            place: pole.place,
            re_sign: pole.re_sign,
        };
        let Some(weight) = weight_for(arena, &view)? else {
            continue;
        };
        let m = pole.mult;
        let slot = match derivatives.iter().position(|(w, _)| *w == weight) {
            Some(k) => k,
            None => {
                derivatives.push((weight, vec![eval::eval(arena, weight)]));
                derivatives.len() - 1
            }
        };
        while derivatives[slot].1.len() < m {
            let j = derivatives[slot].1.len();
            // d/dz of (W^{(j−1)}/(j−1)!) divided by j gives W^{(j)}/j!.
            let prev = derivatives[slot].1[j - 1];
            let d = diff::diff(arena, prev, z);
            let jj = arena.int(i64::try_from(j).ok()?);
            let d = arena.div(d, jj);
            let d = eval::eval(arena, d);
            derivatives[slot].1.push(d);
        }
        let a = principal_part(&r.numer, pole)?;
        for (l, al) in a.iter().enumerate() {
            if al.is_zero() {
                continue;
            }
            let coeff = crate::poly::polybridge::poly_to_expr(arena, al, pole.root);
            let w = subs::subs(arena, derivatives[slot].1[m - 1 - l], z, pole.root);
            terms.push(arena.mul(&[coeff, w]));
        }
    }
    let sum = arena.add(&terms);
    Some(eval::eval(arena, sum))
}

/// The bound variable of the `RootOf` nodes of a residue sum: `r`, or a
/// name not free in `weight`.
pub(crate) fn pole_symbol(arena: &mut Arena, weight: ExprId) -> ExprId {
    let free: Vec<String> = walk::free_symbols(arena, weight)
        .into_iter()
        .filter_map(|s| match arena.node(s) {
            ExprNode::Symbol(sid) => Some(arena.symbol_name(*sid).to_string()),
            _ => None,
        })
        .collect();
    let name = ["r", "_r", "__r"]
        .into_iter()
        .find(|n| !free.iter().any(|f| f == n))
        .unwrap_or("__r");
    arena.symbol(name)
}

/// `∫_{−∞}^{∞} N(x)/D(x)·e^{ikx} dx` (principal value at simple real poles
/// where the caller knows it to be the value) for a real `k` of known sign
/// `k_sign` (a single atom: `NEGATIVE`, `ZERO` or `POSITIVE`; up to 0.49 an
/// `Option<bool>` whose `None` meant `k = 0`), by Jordan's lemma:
/// `2πi Σ_{Im ρ > 0} Res + πi Σ_{ρ real} Res` for `k > 0`, `−2πi Σ_{Im ρ < 0}
/// Res − πi Σ_{ρ real} Res` for `k < 0` (`k = 0` closes upwards and needs
/// `deg D ≥ deg N + 2`).  The integral converges for `deg D ≥ deg N + 1`
/// when `k ≠ 0` and `D` has no real roots; real poles are the caller's
/// business (half residues give the principal value, which is the value
/// only where the integrand is bounded, as for `sin(kx)/x`).  `x` is the
/// symbol of `N/D`, used as the variable of `e^{ikx}`.
pub(crate) fn fourier_rational(
    arena: &mut Arena,
    r: &RationalQ,
    k: ExprId,
    k_sign: SignSet,
    x: ExprId,
) -> Option<ExprId> {
    let gap = r.degree_gap();
    if !k_sign.is_single() || k_sign == SignSet::NONREAL {
        return None;
    }
    if gap < 1 || (k_sign == SignSet::ZERO && gap < 2) {
        return None;
    }
    let i = arena.i_unit();
    let ikx = arena.mul(&[i, k, x]);
    let weight = arena.exp(ikx);
    let weight = eval::eval(arena, weight);
    // `k = 0` closes upwards too.
    let upward = k_sign != SignSet::NEGATIVE;
    let off_axis = rational_residue_sum(arena, r, weight, x, |p| {
        p == if upward {
            PolePlace::Upper
        } else {
            PolePlace::Lower
        }
    })?;
    let on_axis = rational_residue_sum(arena, r, weight, x, |p| p == PolePlace::Real)?;
    let pi = arena.pi();
    let sign = if upward { arena.one() } else { arena.neg_one() };
    let two = arena.int(2);
    let full = arena.mul(&[two, off_axis]);
    let s = arena.add(&[full, on_axis]);
    let v = arena.mul(&[sign, pi, i, s]);
    Some(eval::eval(arena, v))
}

/// `v / n!`
fn divide_by_factorial(arena: &mut Arena, v: ExprId, n: usize) -> ExprId {
    let mut fact = BigInt::one();
    for k in 2..=n {
        fact *= BigInt::from(k);
    }
    let f = arena.big_int(fact);
    arena.div(v, f)
}

/// Is the coefficient `c` (free of the variable) 0, on exact or certified
/// evidence only?
///
/// - `True`: 0 after `eval` or expansion; a constant proved 0 by
///   [`calculus_util::hidden_zero`] or, failing that, by the exact tests
///   of `eval` ([`eval::constant_zero_decision`]: `atan(1/2) + atan(1/3) −
///   π/4`), as it is, factor by factor (a product is 0 exactly when a
///   factor is, a positive power when its base is: `2c`, `c²` for such a
///   `c`), or multiplied out; a constant with parameters shown 0 for all
///   their values by
///   [`calculus_util::exactly_zero`] or [`calculus_util::zero_by_identity`].
/// - `False`: a constant proved nonzero by either; a constant with
///   parameters certified nonzero at a sample point that respects their
///   assumptions
///   ([`nonzero_at_a_sample`](crate::simplify::identically_zero::nonzero_at_a_sample)):
///   not identically 0, so nonzero for generic values of the parameters,
///   the convention of every parametric residue.
/// - `Unknown` otherwise: a constant 0 to the precision reached but not
///   proved, one without a value, one with an unevaluated part.  Callers
///   refuse: taken for 0, a coefficient of the denominator raises the order
///   of the pole, taken for nonzero it lowers it, and the residue is wrong
///   either way.
///
/// Up to 0.50 a constant counted as 0 when its 20-digit decimal was below
/// `10⁻¹⁸`: the simple pole of `1/(z(z + e⁻⁵⁰))` at `0` was taken for a
/// double pole and its residue, `e⁵⁰`, came out `0`.
fn coefficient_is_zero(arena: &mut Arena, c: ExprId) -> Truth {
    let c = eval::eval(arena, c);
    if arena.is_zero_structural(c) {
        return Truth::True;
    }
    let expanded = crate::transforms::expand::expand(arena, c);
    let expanded = eval::eval(arena, expanded);
    if arena.is_zero_structural(expanded) {
        return Truth::True;
    }
    if walk::free_symbols(arena, c).is_empty() {
        let t = constant_is_zero(arena, c);
        if !matches!(t, Truth::Unknown) {
            return t;
        }
        // Factor by factor, when the value is finite (0 to the precision
        // reached, so no factor is infinite).
        if calculus_util::hidden_zero(arena, c).value.is_true() {
            let factors = match arena.node(c) {
                ExprNode::Mul(fs) => fs.to_vec(),
                _ => vec![c],
            };
            let mut all_nonzero = true;
            for f in factors {
                let base = match *arena.node(f) {
                    ExprNode::Pow(b, e) if arena.as_num(e).is_some_and(Signed::is_positive) => b,
                    _ => f,
                };
                match constant_is_zero(arena, base) {
                    Truth::True => return Truth::True,
                    Truth::False => {}
                    Truth::Unknown => all_nonzero = false,
                }
            }
            if all_nonzero {
                return Truth::False;
            }
        }
        return if expanded == c {
            Truth::Unknown
        } else {
            constant_is_zero(arena, expanded)
        };
    }
    if crate::simplify::identically_zero::nonzero_at_a_sample(arena, expanded) {
        return Truth::False;
    }
    let rewritten = calculus_util::exact_rewrites(arena, expanded);
    if calculus_util::exactly_zero(arena, expanded, rewritten)
        || calculus_util::zero_by_identity(arena, expanded, rewritten)
    {
        return Truth::True;
    }
    Truth::Unknown
}

/// Is the symbol-free constant `v` 0: [`calculus_util::hidden_zero`] on
/// exact or certified evidence, else [`eval::constant_zero_decision`].
fn constant_is_zero(arena: &mut Arena, v: ExprId) -> Truth {
    let t = calculus_util::hidden_zero(arena, v).truth_at_least(Evidence::Certified);
    if matches!(t, Truth::Unknown) {
        Truth::from(eval::constant_zero_decision(arena, v))
    } else {
        t
    }
}

/// What is known about a constant `v` (the value of a subexpression at the
/// expansion point).
struct PointValue {
    /// Proved nonzero ([`coefficient_is_zero`] is `False`; with free
    /// parameters: nonzero for generic values of them).
    nonzero: bool,
    /// The complex value, when `v` is a finite number.
    value: Option<num_complex::Complex64>,
}

fn point_value(arena: &mut Arena, v: ExprId) -> PointValue {
    let v = eval::eval(arena, v);
    if arena.is_zero_structural(v) {
        return PointValue {
            nonzero: false,
            value: Some(num_complex::Complex64::new(0.0, 0.0)),
        };
    }
    if !walk::free_symbols(arena, v).is_empty() {
        // Symbolic parameter: generic position, as for every other
        // parameter-dependent residue.
        return PointValue {
            nonzero: coefficient_is_zero(arena, v).is_false(),
            value: None,
        };
    }
    match crate::transforms::evalf::evalf_complex64(arena, v) {
        Ok(z) if z.re.is_finite() && z.im.is_finite() => PointValue {
            nonzero: coefficient_is_zero(arena, v).is_false(),
            value: Some(z),
        },
        _ => PointValue {
            nonzero: false,
            value: None,
        },
    }
}

/// Is `e` provably analytic (holomorphic) in a neighbourhood of `t = 0`?
///
/// Every subexpression depending on `t` is checked with an explicit stack
/// (no recursion over the tree): sums, products and non-negative integer
/// powers of analytic functions are analytic, as are the entire functions
/// (`exp`, `sin`, `cos`, `sinh`, `cosh`, `erf`, `erfc`) of analytic
/// arguments.  Negative powers need a nonzero base at 0; fractional powers,
/// logarithms and the inverse functions additionally need the argument at 0
/// off their branch cuts; `tan`/`tanh`/`Γ`/`ψ` need to avoid their poles.
/// Anything else (`|·|`, `re`, `conj`, `sign`, `H`, `floor`, …) is not
/// certified.  With symbolic parameters the value at 0 is taken in generic
/// position (nonzero when certified nonzero at a sample point,
/// [`coefficient_is_zero`]).
pub(crate) fn analytic_at_zero(arena: &mut Arena, e: ExprId, t: ExprId) -> bool {
    use num_complex::Complex64;
    let zero = arena.zero();
    let mut stack = vec![e];
    let mut seen = rustc_hash::FxHashSet::default();
    // Value of a child at t = 0.
    let at0 = |arena: &mut Arena, c: ExprId| -> PointValue {
        let v = subs::subs(arena, c, t, zero);
        point_value(arena, v)
    };
    // Off the negative real axis (principal branch cut of ln / powers).
    let off_negative_axis = |z: Option<Complex64>| match z {
        Some(z) => z.re > 0.0 || z.im.abs() > 1e-12 * z.norm(),
        None => true,
    };
    while let Some(id) = stack.pop() {
        if !seen.insert(id) || !walk::contains(arena, id, t) {
            continue;
        }
        let node = arena.node(id).clone();
        let ok = match node {
            ExprNode::Symbol(_) => true,
            ExprNode::Add(_) | ExprNode::Mul(_) | ExprNode::Neg(_) => true,
            ExprNode::Exp(_)
            | ExprNode::Sin(_)
            | ExprNode::Cos(_)
            | ExprNode::Sinh(_)
            | ExprNode::Cosh(_)
            | ExprNode::Erf(_)
            | ExprNode::Erfc(_) => true,
            ExprNode::Pow(b, x) => {
                let nonneg_int = arena
                    .as_num(x)
                    .is_some_and(|q| q.is_integer() && !num_traits::Signed::is_negative(q));
                let neg_int = arena
                    .as_num(x)
                    .is_some_and(|q| q.is_integer() && num_traits::Signed::is_negative(q));
                if nonneg_int {
                    true
                } else if !walk::contains(arena, b, t) {
                    // c^x(t) = exp(x(t)·ln c): entire in x for c ≠ 0.
                    at0(arena, b).nonzero
                } else {
                    let pv = at0(arena, b);
                    pv.nonzero && (neg_int || off_negative_axis(pv.value))
                }
            }
            ExprNode::Ln(a) => {
                let pv = at0(arena, a);
                pv.nonzero && off_negative_axis(pv.value)
            }
            ExprNode::Tan(a) => {
                let c = arena.cos(a);
                at0(arena, c).nonzero
            }
            ExprNode::Tanh(a) => {
                let c = arena.cosh(a);
                at0(arena, c).nonzero
            }
            ExprNode::Asin(a) | ExprNode::Acos(a) | ExprNode::Atanh(a) => {
                // cuts: real |x| ≥ 1
                match at0(arena, a).value {
                    Some(z) => z.im.abs() > 1e-12 * z.norm() || z.re.abs() < 1.0,
                    None => true,
                }
            }
            ExprNode::Atan(a) | ExprNode::Asinh(a) => {
                // cuts: imaginary |y| ≥ 1
                match at0(arena, a).value {
                    Some(z) => z.re.abs() > 1e-12 * z.norm() || z.im.abs() < 1.0,
                    None => true,
                }
            }
            ExprNode::Acosh(a) => {
                // cut: real x ≤ 1
                match at0(arena, a).value {
                    Some(z) => z.im.abs() > 1e-12 * z.norm() || z.re > 1.0,
                    None => true,
                }
            }
            ExprNode::Gamma(a) | ExprNode::Digamma(a) => match at0(arena, a).value {
                Some(z) => {
                    !(z.im.abs() <= 1e-12 * z.norm().max(1.0)
                        && z.re <= 0.5
                        && (z.re - z.re.round()).abs() < 1e-12)
                }
                None => true,
            },
            _ => false,
        };
        if !ok {
            return false;
        }
        arena.node(id).clone().for_each_child(|c| stack.push(c));
    }
    true
}

/// Is `v` a finite value free of `t`, infinities and formal nodes?
fn is_finite_constant(arena: &mut Arena, v: ExprId, t: ExprId) -> bool {
    if walk::contains(arena, v, t) || walk::has_unevaluated(arena, v) {
        return false;
    }
    let mut stack = vec![v];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Infinity
            | ExprNode::NegInfinity
            | ExprNode::ComplexInfinity
            | ExprNode::NaN => {
                return false;
            }
            ExprNode::Ln(inner) if arena.is_zero_structural(*inner) => return false,
            ExprNode::Pow(base, exp)
                if arena.is_zero_structural(*base)
                    && arena
                        .as_num(*exp)
                        .is_some_and(num_traits::Signed::is_negative) =>
            {
                return false;
            }
            node => node.for_each_child(|c| stack.push(c)),
        }
    }
    if walk::free_symbols(arena, v).is_empty() && arena.as_num(v).is_none() {
        // Constant: make sure it evaluates to a finite (possibly complex) number.
        return match crate::transforms::evalf::evalf(arena, v, 16) {
            Ok(s) => !s.contains("inf") && !s.contains("nan") && !s.contains("NaN"),
            Err(_) => false,
        };
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn show(a: &Arena, id: ExprId) -> String {
        a.display(id).to_string()
    }

    #[test]
    fn simple_pole() {
        let mut a = Arena::new();
        let z = a.symbol("z");
        let f = a.pow(z, a.neg_one());
        let zero = a.zero();
        let r = residue(&mut a, f, z, zero).unwrap();
        assert_eq!(show(&a, r), "1");
    }

    #[test]
    fn double_pole_exp() {
        let mut a = Arena::new();
        let z = a.symbol("z");
        let m2 = a.int(-2);
        let zm2 = a.pow(z, m2);
        let ez = a.exp(z);
        let f = a.mul(&[zm2, ez]);
        let zero = a.zero();
        let r = residue(&mut a, f, z, zero).unwrap();
        assert_eq!(show(&a, r), "1");
    }

    #[test]
    fn triple_pole_zero_residue() {
        let mut a = Arena::new();
        let z = a.symbol("z");
        let one = a.one();
        let zm1 = a.sub(z, one);
        let m3 = a.int(-3);
        let f = a.pow(zm1, m3);
        let r = residue(&mut a, f, z, one).unwrap();
        assert_eq!(show(&a, r), "0");
    }

    #[test]
    fn cos_over_z_cubed() {
        let mut a = Arena::new();
        let z = a.symbol("z");
        let c = a.cos(z);
        let m3 = a.int(-3);
        let zm3 = a.pow(z, m3);
        let f = a.mul(&[c, zm3]);
        let zero = a.zero();
        let r = residue(&mut a, f, z, zero).unwrap();
        assert_eq!(show(&a, r), "-1/2");
    }

    #[test]
    fn double_pole_at_i() {
        let mut a = Arena::new();
        let z = a.symbol("z");
        let two = a.int(2);
        let z2 = a.pow(z, two);
        let one = a.one();
        let d = a.add(&[z2, one]);
        let m2 = a.int(-2);
        let f = a.pow(d, m2);
        let i = a.i_unit();
        let r = residue(&mut a, f, z, i).unwrap();
        // −i/4
        let s = show(&a, r);
        assert!(s == "-1/4*I" || s == "-1/4*i", "{s}");
    }

    #[test]
    fn analytic_point_is_zero() {
        let mut a = Arena::new();
        let z = a.symbol("z");
        let two = a.int(2);
        let f = a.pow(z, two);
        let zero = a.zero();
        let r = residue(&mut a, f, z, zero).unwrap();
        assert_eq!(show(&a, r), "0");
    }

    #[test]
    fn essential_singularity_by_the_residue_theorem() {
        // Res(e^{1/z}, 0) = 1, the coefficient of 1/z in Σ z^{-n}/n! (this
        // was refused; before that it was 0).
        let mut a = Arena::new();
        let z = a.symbol("z");
        let inv = a.pow(z, a.neg_one());
        let f = a.exp(inv);
        let zero = a.zero();
        let r = residue(&mut a, f, z, zero).unwrap();
        assert_eq!(show(&a, r), "1");
        // e^z·e^{1/z}: ∞ is essential too (the residue is I₁(2)): refused.
        let ez = a.exp(z);
        let g = a.mul(&[ez, f]);
        assert!(residue(&mut a, g, z, zero).is_err());
    }

    #[test]
    fn transcendental_denominator() {
        let mut a = Arena::new();
        let z = a.symbol("z");
        let s = a.sin(z);
        let f = a.pow(s, a.neg_one());
        let zero = a.zero();
        let r = residue(&mut a, f, z, zero).unwrap();
        assert_eq!(show(&a, r), "1");
    }

    #[test]
    fn at_infinity() {
        let mut a = Arena::new();
        let z = a.symbol("z");
        let f = a.pow(z, a.neg_one());
        let r = residue_at_infinity(&mut a, f, z).unwrap();
        assert_eq!(show(&a, r), "-1");
    }
}

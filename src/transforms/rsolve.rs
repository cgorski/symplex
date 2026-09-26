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
/// The roots of one irreducible factor of the characteristic polynomial.
struct RootGroup {
    /// The irreducible factor.
    factor: crate::poly::Poly,
    /// The factor raised to its multiplicity.
    power: crate::poly::Poly,
    mult: usize,
    roots: Vec<Ex>,
}

/// All roots with multiplicities, flattened.
fn flat_roots(groups: &[RootGroup]) -> Vec<(Ex, usize)> {
    groups
        .iter()
        .flat_map(|g| g.roots.iter().map(move |r| (r.clone(), g.mult)))
        .collect()
}

fn characteristic_roots(coeffs: &[Ex], n: &Ex) -> Result<Vec<RootGroup>, SymplexError> {
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
    let mut groups: Vec<RootGroup> = Vec::new();
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
        let mut roots: Vec<Ex> = Vec::new();
        if degree >= 3 {
            for k in 0..degree {
                let root = {
                    let mut inner = ctx.inner.write();
                    let idx = inner.arena.int(k as i64);
                    inner
                        .arena
                        .intern(ExprNode::RootOf(f_expr, r.raw_id(), idx))
                };
                roots.push(r.wrap(root));
            }
            groups.push(RootGroup {
                power: factor.pow(mult),
                factor,
                mult,
                roots,
            });
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
            roots.push(root);
        }
        groups.push(RootGroup {
            power: factor.pow(mult),
            factor,
            mult,
            roots,
        });
    }
    Ok(groups)
}

/// Power sums `p₀, …, p_{count−1}` of the roots of `f` (`pₘ = Σ ρᵐ`), by
/// Newton's identities and then the recurrence of `f`.
fn power_sums(f: &crate::poly::Poly, count: usize) -> Option<Vec<Q>> {
    use num_traits::Zero;
    let d = f.degree()?;
    let lead = f.leading_coeff()?.clone();
    // monic aⱼ (coefficient of xʲ), a_d = 1
    let a: Vec<Q> = (0..=d).map(|j| f.coeff(j) / &lead).collect();
    let mut p: Vec<Q> = Vec::with_capacity(count);
    for m in 0..count {
        let v = if m == 0 {
            Q::from_integer(num_bigint::BigInt::from(d))
        } else if m <= d {
            // p_m + a_{d−1} p_{m−1} + … + a_{d−m+1} p_1 + m·a_{d−m} = 0
            let mut acc = Q::from_integer(num_bigint::BigInt::from(m)) * &a[d - m];
            for i in 1..m {
                acc += &a[d - i] * &p[m - i];
            }
            -acc
        } else {
            let mut acc = Q::zero();
            for i in 1..=d {
                acc += &a[d - i] * &p[m - i];
            }
            -acc
        };
        p.push(v);
    }
    Some(p)
}

/// Exact solution of a square rational linear system (`None` if singular).
fn solve_rational(mut m: Vec<Vec<Q>>, mut rhs: Vec<Q>) -> Option<Vec<Q>> {
    use num_traits::Zero;
    let n = m.len();
    for col in 0..n {
        let p = (col..n).find(|&i| !m[i][col].is_zero())?;
        m.swap(col, p);
        rhs.swap(col, p);
        let pivot_row = m[col].clone();
        let pivot_rhs = rhs[col].clone();
        for (i, (row, r)) in m.iter_mut().zip(rhs.iter_mut()).enumerate() {
            if i != col && !row[col].is_zero() {
                let f = &row[col] / &pivot_row[col];
                for (x, p) in row.iter_mut().zip(&pivot_row).skip(col) {
                    *x -= p * &f;
                }
                *r -= &pivot_rhs * &f;
            }
        }
    }
    Some((0..n).map(|i| &rhs[i] / &m[i][i]).collect())
}

/// Fit the component of an irreducible factor `F` of degree `d ≥ 3`
/// (roots `RootOf(F, k)`) with multiplicity `m` without symbolic
/// elimination over the roots: the constants of conjugate roots are
/// conjugate, `Cⱼ(ρ) = Σ_{l<d} cⱼₗ ρˡ` with rational `cⱼₗ`, so summing the
/// basis over the roots turns every datum into power sums `pₘ = Σ ρᵐ`:
///
/// * sequences: `Σ_ρ nʲ ρˡ ρⁿ = nʲ p_{n+l}`;
/// * functions: `dᵏ/dtᵏ Σ_ρ ρˡ tʲ e^{ρt}` at 0 `= k!/(k−j)! p_{k−j+l}`.
///
/// This is a rational `dm × dm` system (nonsingular: the `nʲρⁿ` are
/// independent and `Cⱼ` has degree `< d`).  Returns the constant for each
/// member basis element.  (`linsolve` over `RootOf` powers did not
/// terminate: `(r³ + r² + r − 1)²` hung.)
fn fit_root_of_component(
    g: &RootGroup,
    members: &[usize],
    basis: &[BasisSeq],
    data: &[Q],
    continuous: bool,
    ctx: &crate::api::context::Context,
) -> Option<Vec<Ex>> {
    use num_traits::Zero;
    let d = g.factor.degree()?;
    let m = g.mult;
    let dim = d * m;
    let p = power_sums(&g.factor, 2 * dim + d + 1)?;
    // unknown index (j, l) → j·d + l
    let mut mat = vec![vec![Q::zero(); dim]; dim];
    for (k, row) in mat.iter_mut().enumerate() {
        for j in 0..m {
            for l in 0..d {
                row[j * d + l] = if continuous {
                    if k < j {
                        Q::zero()
                    } else {
                        let mut falling = num_bigint::BigInt::from(1);
                        for i in (k - j + 1)..=k {
                            falling *= num_bigint::BigInt::from(i);
                        }
                        Q::from_integer(falling) * &p[k - j + l]
                    }
                } else {
                    let kj = num_bigint::BigInt::from(k).pow(j as u32);
                    Q::from_integer(kj) * &p[k + l]
                };
            }
        }
    }
    let c = solve_rational(mat, data[..dim].to_vec())?;
    // constant of member (root ρ, power j) = Σ_l c_{j,l} ρ^l
    let mut out = Vec::with_capacity(members.len());
    for &i in members {
        let BasisKind::Power { root, j } = &basis[i].kind else {
            return None;
        };
        let mut acc = ctx.zero();
        for l in 0..d {
            let cl = &c[*j * d + l];
            if cl.is_zero() {
                continue;
            }
            let term = &ctx.from_ratio(cl.clone()) * &root.powi(l as i64);
            acc = &acc + &term;
        }
        out.push(acc);
    }
    Some(out)
}

/// One homogeneous basis sequence, with what is needed to evaluate it
/// exactly at an integer `n = k` (for fitting initial values).
enum BasisKind {
    /// `n^j·rⁿ` for a real root `r` (rational, radical or `RootOf`).
    Power { root: Ex, j: usize },
    /// `KroneckerDelta(n, i)`: the root `0` of multiplicity `m` (leading
    /// coefficients `c₀ = … = c_{m−1} = 0`) contributes the sequences
    /// supported on `n < m` — the recurrence never constrains `a(0..m)`.
    Delta(usize),
    /// `n^j·Re((α + iβ)ⁿ) = n^j·ρⁿ·cos(nθ)`, `β > 0`.
    Cos { re: Ex, im: Ex, j: usize },
    /// `n^j·Im((α + iβ)ⁿ) = n^j·ρⁿ·sin(nθ)`, `β > 0`.
    Sin { re: Ex, im: Ex, j: usize },
}

struct BasisSeq {
    expr: Ex,
    kind: BasisKind,
    /// Index of the [`RootGroup`] the sequence belongs to.
    group: usize,
    /// A function of a continuous variable (`tʲe^{rt}`, the `k`-th datum
    /// is the `k`-th derivative at `0`) rather than a sequence (`nʲrⁿ`, the
    /// `k`-th datum is the value at `n = k`).
    continuous: bool,
}

/// `xᵏ` by repeated multiplication with expansion, so that powers of
/// quadratic radicals stay reduced (`((1+√5)/2)³ = 2 + √5`).
fn exact_pow(x: &Ex, k: usize) -> Ex {
    let ctx = x.context();
    let mut acc = ctx.one();
    for _ in 0..k {
        acc = (&acc * x).expand().eval();
    }
    acc
}

/// `(re + i·im)ᵏ` as `(Re, Im)`, expanded exactly (de Moivre without
/// angles: `cos(k·θ)` of an angle such as `acos(√5/5)` does not reduce).
fn exact_complex_pow(re: &Ex, im: &Ex, k: usize) -> (Ex, Ex) {
    let ctx = re.context();
    let (mut x, mut y) = (ctx.one(), ctx.zero());
    for _ in 0..k {
        let nx = (&(&x * re) - &(&y * im)).expand().eval();
        let ny = (&(&x * im) + &(&y * re)).expand().eval();
        x = nx;
        y = ny;
    }
    (x, y)
}

impl BasisSeq {
    /// The exact `k`-th datum: the value at `n = k` of a sequence, the
    /// `k`-th derivative at `t = 0` of a function.
    fn value_at(&self, k: usize, ctx: &crate::api::context::Context) -> Ex {
        if self.continuous {
            return self.derivative_at_zero(k, ctx);
        }
        let k_pow = |j: usize| -> Ex {
            let mut acc = ctx.one();
            for _ in 0..j {
                acc = (&acc * k as i64).eval();
            }
            acc
        };
        match &self.kind {
            BasisKind::Delta(i) => {
                if *i == k {
                    ctx.one()
                } else {
                    ctx.zero()
                }
            }
            BasisKind::Power { root, j } => {
                let rk = if contains_root_of(root) {
                    root.powi(k as i64).eval()
                } else {
                    exact_pow(root, k)
                };
                (&k_pow(*j) * &rk).eval()
            }
            BasisKind::Cos { re, im, j } => {
                let (x, _) = exact_complex_pow(re, im, k);
                (&k_pow(*j) * &x).eval()
            }
            BasisKind::Sin { re, im, j } => {
                let (_, y) = exact_complex_pow(re, im, k);
                (&k_pow(*j) * &y).eval()
            }
        }
    }

    /// `dᵏ/dtᵏ [tʲ e^{rt}]` at `t = 0` is `k!/(k−j)!·r^{k−j}` (`0` for
    /// `k < j`); the complex pairs take real and imaginary parts.
    fn derivative_at_zero(&self, k: usize, ctx: &crate::api::context::Context) -> Ex {
        let j = match &self.kind {
            BasisKind::Power { j, .. } | BasisKind::Cos { j, .. } | BasisKind::Sin { j, .. } => *j,
            BasisKind::Delta(_) => return ctx.zero(),
        };
        if k < j {
            return ctx.zero();
        }
        // k!/(k − j)!
        let mut falling = num_bigint::BigInt::from(1);
        for i in (k - j + 1)..=k {
            falling *= num_bigint::BigInt::from(i);
        }
        let falling = ctx.from_ratio(Q::from_integer(falling));
        let v = match &self.kind {
            BasisKind::Power { root, .. } => {
                if contains_root_of(root) {
                    root.powi((k - j) as i64).eval()
                } else {
                    exact_pow(root, k - j)
                }
            }
            BasisKind::Cos { re, im, .. } => exact_complex_pow(re, im, k - j).0,
            BasisKind::Sin { re, im, .. } => exact_complex_pow(re, im, k - j).1,
            BasisKind::Delta(_) => ctx.zero(),
        };
        (&falling * &v).eval()
    }
}

fn contains_root_of(e: &Ex) -> bool {
    let inner = e.inner.read();
    let mut stack = vec![e.raw_id()];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        let node = inner.arena.node(id);
        if matches!(node, ExprNode::RootOf(..)) {
            return true;
        }
        node.for_each_child(|c| stack.push(c));
    }
    false
}

/// The angle `θ ∈ (0, π)` of `α + iβ` (`β > 0`) from `cos θ = α/ρ`.
///
/// For a root of a rational quadratic, `cos²θ = α²/(α² + β²)` is rational;
/// when it is `0`, `1/4`, `1/2` or `3/4` the angle is a rational multiple
/// of `π` (the root is `ρ` times a root of unity) and is returned as such —
/// `eval` does not reduce `atan2(√3/2, 1/2)` or `acos(√2/2)`.  Otherwise
/// `acos(α/ρ)`.
fn pair_angle(re: &Ex, im: &Ex, rho: &Ex) -> Ex {
    let ctx = re.context();
    let cos2 = (&re.powi(2) / &(&re.powi(2) + &im.powi(2))).expand().eval();
    if let (Some(c2), Some(a)) = (as_rational(&cos2), as_rational(&re.eval())) {
        let neg = num_traits::Signed::is_negative(&a);
        let table: [(i64, i64, i64, i64); 4] = [
            // (cos² num, den, θ/π num, den) for α ≥ 0
            (0, 1, 1, 2),
            (1, 4, 1, 3),
            (1, 2, 1, 4),
            (3, 4, 1, 6),
        ];
        for (cn, cd, tn, td) in table {
            if c2 == Q::new(cn.into(), cd.into()) {
                let frac = if neg && cn != 0 {
                    ctx.rational(td - tn, td)
                } else {
                    ctx.rational(tn, td)
                };
                return (&frac * &ctx.pi()).eval();
            }
        }
    }
    (re / rho).acos().eval().simplify()
}

/// Homogeneous basis sequences for the given characteristic roots.
///
/// Real root `r` of multiplicity `m` → `n^j·rⁿ` (j < m); the root `0` →
/// `KroneckerDelta(n, j)` (j < m).  A complex pair `α ± βi` →
/// `n^j·ρⁿ·cos(nθ)`, `n^j·ρⁿ·sin(nθ)` with `ρ = |α+βi|` and
/// `θ ∈ (0, π)` (see [`pair_angle`]).
///
/// With `continuous`, the basis of the ODE `Σ cₖ f⁽ᵏ⁾ = 0` instead:
/// `tʲe^{rt}` and `tʲe^{αt}cos(βt)`, `tʲe^{αt}sin(βt)`.
fn homogeneous_basis(groups: &[RootGroup], n: &Ex, continuous: bool) -> Vec<BasisSeq> {
    let mut basis = Vec::new();
    for (gi, g) in groups.iter().enumerate() {
        let roots: Vec<(Ex, usize)> = g.roots.iter().map(|r| (r.clone(), g.mult)).collect();
        group_basis(&roots, n, gi, continuous, &mut basis);
    }
    basis
}

/// [`homogeneous_basis`] for the roots of one irreducible factor.
fn group_basis(
    roots: &[(Ex, usize)],
    n: &Ex,
    group: usize,
    continuous: bool,
    basis: &mut Vec<BasisSeq>,
) {
    let ctx = n.context();
    let i_unit = ctx.i_unit();
    let mut used = vec![false; roots.len()];
    for i in 0..roots.len() {
        if used[i] {
            continue;
        }
        used[i] = true;
        let (root, mult) = &roots[i];
        if !root.contains(&i_unit) {
            for j in 0..*mult {
                if root.is_zero_structural() && !continuous {
                    basis.push(BasisSeq {
                        expr: n.kronecker_delta(&ctx.int(j as i64)),
                        kind: BasisKind::Delta(j),
                        group,
                        continuous,
                    });
                    continue;
                }
                let term = if continuous {
                    if root.is_zero_structural() {
                        n_pow(n, j)
                    } else {
                        &n_pow(n, j) * &(root * n).exp()
                    }
                } else if root.is_one_structural() {
                    n_pow(n, j)
                } else {
                    &n_pow(n, j) * &root.pow(n)
                };
                basis.push(BasisSeq {
                    expr: term.eval(),
                    kind: BasisKind::Power {
                        root: root.clone(),
                        j,
                    },
                    group,
                    continuous,
                });
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
        // β > 0 so that sin(nθ) = Im((α + iβ)ⁿ)/ρⁿ with θ ∈ (0, π).
        let im = im.powi(2).sqrt().eval().simplify();
        let (envelope_base, angle) = if continuous {
            // e^{αt}, βt
            let env = if re.is_zero_structural() {
                None
            } else {
                Some((&re * n).exp())
            };
            (env, im.clone())
        } else {
            let rho = (&re.powi(2) + &im.powi(2)).sqrt().eval().simplify();
            let env = if rho.is_one_structural() {
                None
            } else {
                Some(rho.pow(n))
            };
            (env, pair_angle(&re, &im, &rho))
        };
        for j in 0..*mult {
            let envelope = match &envelope_base {
                None => n_pow(n, j),
                Some(e) => &n_pow(n, j) * e,
            };
            let n_theta = n * &angle;
            basis.push(BasisSeq {
                expr: (&envelope * &n_theta.cos()).eval(),
                kind: BasisKind::Cos {
                    re: re.clone(),
                    im: im.clone(),
                    j,
                },
                group,
                continuous,
            });
            basis.push(BasisSeq {
                expr: (&envelope * &n_theta.sin()).eval(),
                kind: BasisKind::Sin {
                    re: re.clone(),
                    im: im.clone(),
                    j,
                },
                group,
                continuous,
            });
        }
    }
}

/// Fit the constants one irreducible factor at a time.
///
/// The homogeneous part `h = a − yₚ` of the solution lies in
/// `ker χ(E) = ⊕ᵢ ker Fᵢ^{mᵢ}(E)` (`E` the shift, `χ = Π Fᵢ^{mᵢ}` over ℚ).
/// With `x·Gᵢ + y·Fᵢ^{mᵢ} = 1` for `Gᵢ = χ/Fᵢ^{mᵢ}`, the operator `Wᵢ(E)`,
/// `Wᵢ = x·Gᵢ mod χ`, projects `h` onto its `Fᵢ` component, so the first
/// values of every component are *rational* (from the rational values of
/// `h`, extended by the recurrence).  Each component is then fitted by its
/// own basis — a system of size `mᵢ·deg Fᵢ` over a single number field
/// instead of one system of size `k` over all of them together (which
/// swelled without bound: order 8, roots `(1±√5)/2` and `e^{±2πi/3}` twice).
///
/// The same projection works for the ODE `Σ cₖ f⁽ᵏ⁾ = 0` with `E`
/// replaced by `d/dt` and values by derivatives at `0` (`initial` then
/// holds `f(0), f′(0), …`).
///
/// `initial` holds the `order` rational data of `h`.  Returns `None` if a
/// component system is not uniquely solvable; the caller then fits all
/// constants at once.
fn fit_by_components(
    coeffs: &[Q],
    groups: &[RootGroup],
    basis: &[BasisSeq],
    constants: &[Ex],
    initial: &[Q],
    ctx: &crate::api::context::Context,
) -> Option<Vec<(Ex, Ex)>> {
    use num_traits::Zero;
    let order = coeffs.len() - 1;
    if initial.len() != order {
        return None;
    }
    // h(0..2·order): the initial data, extended by the recurrence (for the
    // ODE: the derivatives at 0, extended by the equation).
    let mut h: Vec<Q> = initial.to_vec();
    let lead = coeffs[order].clone();
    if lead.is_zero() {
        return None;
    }
    for m in order..2 * order {
        let mut acc = Q::zero();
        for (j, c) in coeffs.iter().enumerate().take(order) {
            acc += c * &h[m - order + j];
        }
        h.push(-acc / &lead);
    }
    let chi = crate::poly::Poly::from_coeffs(coeffs.to_vec());
    let mut pairs: Vec<(Ex, Ex)> = Vec::new();
    for (gi, g) in groups.iter().enumerate() {
        let (cofactor, rem) = chi.div_rem(&g.power);
        if !rem.is_zero() {
            return None;
        }
        let eg = crate::poly::Poly::extended_gcd(&cofactor, &g.power);
        if eg.gcd.degree() != Some(0) {
            return None;
        }
        let w = eg.x.mul(&cofactor).rem(&chi);
        let dim = g.power.degree()?;
        let members: Vec<usize> = (0..basis.len()).filter(|&i| basis[i].group == gi).collect();
        if members.len() != dim {
            return None;
        }
        let unknowns: Vec<Ex> = members.iter().map(|&i| constants[i].clone()).collect();
        // the component's data hᵢ(0..dim)
        let mut hi: Vec<Q> = Vec::with_capacity(dim);
        for k in 0..dim {
            let mut hk = Q::zero();
            for (j, wj) in w.coeffs().iter().enumerate() {
                hk += wj * h.get(k + j)?;
            }
            hi.push(hk);
        }
        if g.factor.degree().is_some_and(|d| d >= 3) {
            let continuous = basis[members[0]].continuous;
            let values = fit_root_of_component(g, &members, basis, &hi, continuous, ctx)?;
            pairs.extend(unknowns.iter().cloned().zip(values));
            continue;
        }
        let mut eqs: Vec<Ex> = Vec::with_capacity(dim);
        for (k, hk) in hi.into_iter().enumerate() {
            let mut acc = -ctx.from_ratio(hk);
            for &i in &members {
                let bk = basis[i].value_at(k, ctx);
                if !bk.is_zero_structural() {
                    acc = &acc + &(&constants[i] * &bk);
                }
            }
            eqs.push(acc.eval());
        }
        match linsolve(&eqs, &unknowns).ok()? {
            LinearSolution::Unique(sol) => pairs.extend(sol),
            _ => return None,
        }
    }
    Some(pairs)
}

/// [`fit_by_components`] with only the first `ics.len() < k` initial
/// values: the missing values become free parameters, named `C1, C2, …` in
/// the result (`Cᵢ = a(len + i − 1)`).
#[allow(clippy::too_many_arguments)]
fn fit_partial_by_components(
    coeffs: &[Q],
    groups: &[RootGroup],
    basis: &[BasisSeq],
    constants: &[Ex],
    general: &Ex,
    particular: &Ex,
    ics: &[Ex],
    n: &Ex,
) -> Option<Ex> {
    use num_traits::Zero;
    let ctx = n.context();
    let order = coeffs.len() - 1;
    let known = ics.len();
    // h(k) = a(k) − yₚ(k) for the known values; the missing ones are free.
    let mut base: Vec<Q> = Vec::with_capacity(order);
    for (k, v) in ics.iter().enumerate() {
        base.push(as_rational(
            &(v - &particular.subs_i64(n, k as i64).eval()).eval(),
        )?);
    }
    // a(k) = p_k is free for k ≥ known, so h(k) = p_k − yₚ(k).
    let mut yp_tail: Vec<Q> = Vec::new();
    for k in known..order {
        yp_tail.push(as_rational(&particular.subs_i64(n, k as i64).eval())?);
    }
    let mut data0 = base.clone();
    data0.extend(yp_tail.iter().map(|y| -y.clone()));
    let fit0 = fit_by_components(coeffs, groups, basis, constants, &data0, &ctx)?;
    let params: Vec<Ex> = (0..order - known)
        .map(|i| ctx.symbol(&format!("__rsolve_P{i}")))
        .collect();
    let mut value: Vec<Ex> = constants
        .iter()
        .map(|c| fit0.iter().find(|(k, _)| k == c).map(|(_, v)| v.clone()))
        .collect::<Option<Vec<_>>>()?;
    for (i, p) in params.iter().enumerate() {
        let mut unit = vec![Q::zero(); order];
        unit[known + i] = Q::from_integer(1.into());
        let fit = fit_by_components(coeffs, groups, basis, constants, &unit, &ctx)?;
        for (slot, c) in value.iter_mut().zip(constants) {
            let vi = fit.iter().find(|(k, _)| k == c).map(|(_, v)| v.clone())?;
            if !vi.is_zero_structural() {
                *slot = &*slot + &(p * &vi);
            }
        }
    }
    // Substitute all basis constants at once (they share the C-names).
    let pairs: Vec<(&Ex, &Ex)> = constants.iter().zip(value.iter()).collect();
    let mut sol = general.subs_map(&pairs);
    for (i, p) in params.iter().enumerate() {
        sol = sol.subs(p, &constant(&ctx, i + 1));
    }
    swell_check(&[&sol]).ok()?;
    let sol = sol.eval();
    swell_check(&[&sol]).ok()?;
    Some(sol.simplify())
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

    let groups = characteristic_roots(&coeffs, n)?;
    let roots = flat_roots(&groups);
    let basis = homogeneous_basis(&groups, n, false);
    let constants: Vec<Ex> = (1..=basis.len()).map(|k| constant(&ctx, k)).collect();
    let mut general = ctx.zero();
    for (c, b) in constants.iter().zip(&basis) {
        general = &general + &(c * &b.expr);
    }
    let mut particular = ctx.zero();

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
                particular = &particular + &yp;
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

    let rat_coeffs: Option<Vec<Q>> = coeffs.iter().map(as_rational).collect();
    // h(k) = a(k) − yₚ(k), rational when the forcing bases are.
    let initial: Option<Vec<Q>> = ics
        .iter()
        .enumerate()
        .map(|(k, v)| as_rational(&(v - &particular.subs_i64(n, k as i64).eval()).eval()))
        .collect();
    if let (Some(rc), Some(initial)) = (rat_coeffs, initial)
        && let Some(pairs) = fit_by_components(&rc, &groups, &basis, &constants, &initial, &ctx)
    {
        let mut sol = general.clone();
        for (c, v) in &pairs {
            sol = sol.subs(c, v);
        }
        swell_check(&[&sol])?;
        let sol = sol.eval();
        swell_check(&[&sol])?;
        return Ok(sol.simplify());
    }

    // Fewer initial values than the order over several number fields or
    // RootOf roots: the global elimination below does not terminate there
    // (order 7 with a cubic and a Gaussian quadratic factor hung).  Fit by
    // components instead, with the missing values a(len), …, a(k−1) as the
    // free constants C1, C2, … (linearity: one rational fit per unit datum).
    let quadratic_fields = groups
        .iter()
        .filter(|g| g.factor.degree() == Some(2))
        .count();
    let risky = groups
        .iter()
        .any(|g| g.factor.degree().is_some_and(|d| d >= 3))
        || quadratic_fields >= 2;
    if risky
        && ics.len() < coeffs.len() - 1
        && let Some(rc) = coeffs.iter().map(as_rational).collect::<Option<Vec<Q>>>()
        && let Some(sol) = fit_partial_by_components(
            &rc,
            &groups,
            &basis,
            &constants,
            &general,
            &particular,
            ics,
            n,
        )
    {
        return Ok(sol);
    }

    // Fit initial values a(0..len) all at once (fewer initial values than
    // the order, or irrational values).  The equations are built from the
    // exact values of the basis sequences (powers of the roots expanded,
    // complex pairs by de Moivre) rather than by substituting into
    // `general`: `cos(k·acos(√5/5))` does not reduce, and symbolic
    // elimination over such entries does not terminate in practice
    // (order-6 recurrences hung).
    let eqs: Vec<Ex> = ics
        .iter()
        .enumerate()
        .map(|(k, v)| {
            let mut acc = (particular.subs_i64(n, k as i64).eval() - v).eval();
            for (c, b) in constants.iter().zip(&basis) {
                let bk = b.value_at(k, &ctx);
                if !bk.is_zero_structural() {
                    acc = &acc + &(c * &bk);
                }
            }
            acc.eval()
        })
        .collect();
    let fitted =
        crate::api::expr_solve_ext::fit_constants(&general, &constants, &eqs, "rsolve_linear")?;
    swell_check(&[&fitted])?;
    Ok(fitted)
}

/// Solve a homogeneous linear problem with rational constant coefficients
/// `coeffs = [c₀, …, c_k]` from `k` rational initial data:
///
/// * `continuous == false`: the sequence with `Σ cⱼ a(n+j) = 0` and
///   `a(i) = initial[i]`;
/// * `continuous == true`: the function with `Σ cⱼ f⁽ʲ⁾(t) = 0` and
///   `f⁽ⁱ⁾(0) = initial[i]`.
///
/// The closed form is in `var`.  Used by the inverse Laplace and Z
/// transforms of rational functions (whose initial data come from the
/// expansion at infinity).
///
/// # Errors
///
/// `InvalidArgument` for a zero leading coefficient or the wrong number of
/// initial data; `ComputationFailed` if the roots are not available or a
/// component system is singular.
pub(crate) fn solve_constant_coefficient_ivp(
    coeffs: &[Q],
    initial: &[Q],
    var: &Ex,
    continuous: bool,
) -> Result<Ex, SymplexError> {
    let op = if continuous {
        "linear ODE initial value problem"
    } else {
        "linear recurrence initial value problem"
    };
    let order = coeffs.len().saturating_sub(1);
    if order == 0 || num_traits::Zero::is_zero(&coeffs[order]) || initial.len() != order {
        return Err(SymplexError::InvalidArgument {
            operation: "solve_constant_coefficient_ivp",
            reason: format!("{op}: need a nonzero leading coefficient and one datum per order"),
        });
    }
    let ctx = var.context();
    let coeff_ex: Vec<Ex> = coeffs.iter().map(|c| ctx.from_ratio(c.clone())).collect();
    let groups = characteristic_roots(&coeff_ex, var)?;
    let basis = homogeneous_basis(&groups, var, continuous);
    let constants: Vec<Ex> = (1..=basis.len())
        .map(|k| ctx.symbol(&format!("__ivp_C{k}")))
        .collect();
    let pairs =
        fit_by_components(coeffs, &groups, &basis, &constants, initial, &ctx).ok_or_else(|| {
            SymplexError::ComputationFailed {
                operation: "solve_constant_coefficient_ivp",
                reason: format!("{op}: could not fit the initial data"),
            }
        })?;
    let mut sol = ctx.zero();
    for (c, b) in constants.iter().zip(&basis) {
        let v = pairs
            .iter()
            .find(|(k, _)| k == c)
            .map(|(_, v)| v.clone())
            .ok_or_else(|| SymplexError::ComputationFailed {
                operation: "solve_constant_coefficient_ivp",
                reason: format!("{op}: a constant was not determined"),
            })?;
        if !v.is_zero_structural() {
            sol = &sol + &(&v * &b.expr);
        }
    }
    swell_check(&[&sol])?;
    let sol = sol.eval();
    swell_check(&[&sol])?;
    Ok(sol.simplify())
}

/// Coefficients `e₀, e₁, …, e_{count−1}` of `num/den` in powers of `1/x`
/// (the expansion at infinity; requires `deg num ≤ deg den`).
pub(crate) fn series_at_infinity(
    num: &crate::poly::Poly,
    den: &crate::poly::Poly,
    count: usize,
) -> Option<Vec<Q>> {
    use num_traits::Zero;
    let d = den.degree()?;
    if num.degree().is_some_and(|k| k > d) {
        return None;
    }
    // With w = 1/x: num/den = Ñ(w)/D̃(w), Ñ_m = num_{d−m}, D̃_j = den_{d−j}.
    let at =
        |p: &crate::poly::Poly, m: usize| -> Q { if m > d { Q::zero() } else { p.coeff(d - m) } };
    let d0 = at(den, 0);
    if d0.is_zero() {
        return None;
    }
    let mut e: Vec<Q> = Vec::with_capacity(count);
    for m in 0..count {
        let mut acc = at(num, m);
        for j in 1..=m.min(d) {
            acc -= at(den, j) * &e[m - j];
        }
        e.push(acc / &d0);
    }
    Some(e)
}

/// The exact inverse transform of the rational function `num/den` (rational
/// coefficients), built in a scratch context and transferred into `arena`
/// as an expression in `var`:
///
/// * `continuous` (Laplace, `deg num < deg den`): `f(t)` with
///   `den(d/dt) f = 0` and `f⁽ᵏ⁾(0⁺)` the coefficient of `s^{−k−1}` at
///   infinity;
/// * otherwise (Z, `deg num ≤ deg den`): `x[n]` with `x[n]` the coefficient
///   of `z^{−n}`; it satisfies `Σⱼ denⱼ x[m + j] = 0` for `m ≥ 1`, i.e.
///   the recurrence with coefficients `[0, den₀, …, den_d]` (the extra root
///   `0` contributes `KroneckerDelta(n, 0)`).
///
/// Repeated, complex and non-monic factors are all handled by
/// [`solve_constant_coefficient_ivp`].  `None` if the shape does not apply
/// or the roots are not available.
pub(crate) fn rational_inverse_into(
    arena: &mut crate::base::arena::Arena,
    num: &crate::poly::Poly,
    den: &crate::poly::Poly,
    var: crate::base::node::ExprId,
    continuous: bool,
) -> Option<crate::base::node::ExprId> {
    let d = den.degree()?;
    if d == 0 || num.is_zero() {
        return None;
    }
    let (coeffs, data) = if continuous {
        if num.degree()? >= d {
            return None;
        }
        let e = series_at_infinity(num, den, d + 1)?;
        (den.coeffs().to_vec(), e[1..].to_vec())
    } else {
        let e = series_at_infinity(num, den, d + 1)?;
        let mut c = vec![<Q as num_traits::Zero>::zero()];
        c.extend(den.coeffs().iter().cloned());
        (c, e)
    };
    let scratch = crate::api::context::Context::new();
    let v = scratch.symbol("__rational_inverse_var");
    let sol = solve_constant_coefficient_ivp(&coeffs, &data, &v, continuous).ok()?;
    if sol.has_unevaluated() {
        return None;
    }
    let id = {
        let inner = scratch.inner.read();
        let saved = arena.last_compact_size;
        let mut map = rustc_hash::FxHashMap::default();
        let id =
            crate::base::compact::transfer_subtree(&inner.arena, arena, sol.raw_id(), &mut map);
        arena.last_compact_size = saved;
        id
    };
    let tmp = arena.symbol("__rational_inverse_var");
    let id = crate::transforms::subs::subs(arena, id, tmp, var);
    Some(crate::transforms::eval::eval(arena, id))
}

/// The first index `k₀ ≥ 0` at which the rational function `p(n)` has a
/// zero (`(k₀, false)`) or a pole (`(k₀, true)`), if any.  Only the
/// rational roots of the numerator and denominator are candidates.
fn first_index_singularity(p: &Ex, n: &Ex) -> Option<(i64, bool)> {
    let (num, den) = p.eval().as_numer_denom();
    let mut best: Option<(i64, bool)> = None;
    for (part, is_pole) in [(num, false), (den, true)] {
        if !part.contains(n) || !part.is_polynomial(n) {
            continue;
        }
        let coeffs = {
            let mut inner = part.inner.write();
            crate::transforms::solve::symbolic_poly_coeffs(
                &mut inner.arena,
                part.raw_id(),
                n.raw_id(),
            )
        };
        let Some(coeffs) = coeffs else { continue };
        let rat: Option<Vec<Q>> = coeffs.iter().map(|&c| as_rational(&part.wrap(c))).collect();
        let Some(rat) = rat else { continue };
        let poly = crate::poly::Poly::from_coeffs(rat);
        let Some(lead) = poly.leading_coeff().cloned() else {
            continue;
        };
        if poly.degree() == Some(1) {
            let root = -poly.coeff(0) / &lead;
            if root.is_integer()
                && !num_traits::Signed::is_negative(&root)
                && let Ok(k0) = i64::try_from(root.to_integer())
                && best.is_none_or(|(b, _)| k0 < b)
            {
                best = Some((k0, is_pole));
            }
            continue;
        }
        // Non-negative integer roots: test k = 0, 1, … up to the Cauchy
        // bound on the roots (p of degree ≥ 2 only reaches the formal
        // `Product`, which stays correct past a zero anyway).
        let mut bound = Q::from_integer(1.into());
        for c in poly.coeffs() {
            let r = num_traits::Signed::abs(&(c / &lead));
            bound += r;
        }
        let bound = bound.to_integer();
        let bound: i64 = bound.try_into().unwrap_or(i64::MAX).min(1_000_000);
        let mut kk = 0i64;
        while kk <= bound {
            if best.is_some_and(|(b, _)| b <= kk) {
                break;
            }
            if num_traits::Zero::is_zero(&poly.eval(&Q::from_integer(kk.into()))) {
                best = Some((kk, is_pole));
                break;
            }
            kk += 1;
        }
    }
    best
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
    let start = match a0 {
        Some(v) => {
            let _ = n.checked_id(v);
            v.clone()
        }
        None => constant(&ctx, 1),
    };
    let q_eval = q.eval();
    // A zero or pole of p at an index k₀ ≥ 0 breaks the product formula:
    // P(n) = Π_{k<n} p(k) vanishes (or is undefined) from n = k₀ + 1 on, so
    // the closed forms Γ(n + β/α)/Γ(β/α) and Σ q(k)/P(k+1) are meaningless
    // (p = n gave 3·Γ(n)/Γ(0)).
    if let Some((k0, is_pole)) = first_index_singularity(p, n) {
        if is_pole {
            return Err(SymplexError::InvalidArgument {
                operation: "rsolve_first_order",
                reason: format!(
                    "p(n) = {p} has a pole at n = {k0}: a({}) is undefined",
                    k0 + 1
                ),
            });
        }
        if !q_eval.is_zero_structural() {
            return Err(SymplexError::ComputationFailed {
                operation: "rsolve_first_order",
                reason: format!(
                    "p(n) = {p} vanishes at n = {k0}; the product formula does not apply"
                ),
            });
        }
        // a(n) = a(0)·P(n) is supported on n ≤ k₀: a finite sum of deltas.
        let mut value = start.clone();
        let mut total = ctx.zero();
        for i in 0..=k0 {
            total = &total + &(&value * &n.kronecker_delta(&ctx.int(i)));
            value = (&value * &p.subs_i64(n, i)).eval();
        }
        return Ok(total.eval());
    }
    let k = ctx.symbol("__k_rsolve");
    let p_k = p.subs(n, &k);
    let big_p = product_closed_form(&p_k, &k, n);
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

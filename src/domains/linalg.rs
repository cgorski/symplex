//! Linear system solving via symbolic Gaussian elimination.
//!
//! [`linsolve_symbolic`] computes a reduced row-echelon form.  It handles
//! symbolic coefficients, under- and over-determined systems, and reports
//! free variables and inconsistency.  It is the backend of the public
//! `linsolve` API (and of `Context::solve_system`).  Rational coefficients
//! are eliminated on [`QMatrix`]; rational functions of the parameters
//! over square roots of rationals and `i` fraction-free over
//! `K[params]` ([`Tower`]); anything else on expressions.
//!
//! [`Tower`] also provides the exact zero test of rational functions over
//! `ℚ(radicals, i)` used by the pivots of `Matrix::rref`, `rank`,
//! `nullspace`, `lu` and the eigenvector eliminations.

use num_bigint::BigInt;
use num_traits::{One, Signed, ToPrimitive, Zero};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::api::expr::{Ex, SimplifyOpts};
use crate::base::arena::Arena;
use crate::base::errors::SymplexError;
use crate::base::node::{ExprId, ExprNode};
use crate::domains::exact_matrix::QMatrix;
use crate::domains::linprog::Q;
use crate::domains::matrix::{RF_POINTS, RF_TERM_BUDGET, rf_sample_point};
use crate::poly::multipoly::{GrevLex, MultiPoly};

// ═══════════════════════════════════════════════════════════════════════════
// Symbolic RREF solver
// ═══════════════════════════════════════════════════════════════════════════

/// Result of the symbolic RREF solver.
#[derive(Debug, Clone)]
pub(crate) struct SymbolicLinearResult {
    /// One value per unknown, in input order.  Free unknowns map to
    /// themselves; pivot unknowns are expressed in terms of the free ones.
    pub values: Vec<Ex>,
    /// Indices (into the unknowns) of the free variables.
    pub free: Vec<usize>,
    /// `true` if a row `0 = c` with `c ≠ 0` was found.
    pub inconsistent: bool,
}

/// Decompose `eq` (an expression equal to zero) into a linear row over
/// `vars`: returns `(coefficients, rhs)` with `Σ coeffᵢ·varᵢ = rhs`.
///
/// Returns `None` if `eq` is not linear in the unknowns (a term contains
/// two unknowns, a power of an unknown, or an unknown inside a function).
pub(crate) fn extract_linear_row(
    arena: &mut Arena,
    eq: ExprId,
    vars: &[ExprId],
) -> Option<(Vec<ExprId>, ExprId)> {
    let expanded = crate::transforms::expand::expand(arena, eq);
    let expanded = crate::transforms::eval::eval(arena, expanded);
    let terms: Vec<ExprId> = match arena.node(expanded).clone() {
        ExprNode::Add(children) => children.to_vec(),
        _ => vec![expanded],
    };

    let mut coeff_parts: Vec<Vec<ExprId>> = vec![Vec::new(); vars.len()];
    let mut const_parts: Vec<ExprId> = Vec::new();

    for term in terms {
        let mut which: Option<usize> = None;
        for (i, &v) in vars.iter().enumerate() {
            if crate::base::walk::contains(arena, term, v) {
                if which.is_some() {
                    return None; // two unknowns in one term → nonlinear
                }
                which = Some(i);
            }
        }
        match which {
            None => const_parts.push(term),
            Some(i) => {
                let coeffs = crate::transforms::solve::symbolic_poly_coeffs(arena, term, vars[i])?;
                // Must be exactly c·var (degree 1, zero constant part).
                if coeffs.len() != 2 || !arena.is_zero_structural(coeffs[0]) {
                    return None;
                }
                coeff_parts[i].push(coeffs[1]);
            }
        }
    }

    let mut coeffs = Vec::with_capacity(vars.len());
    for parts in coeff_parts {
        let c = match parts.len() {
            0 => arena.zero,
            1 => parts[0],
            _ => arena.add(&parts),
        };
        coeffs.push(crate::transforms::eval::eval(arena, c));
    }
    let constant = match const_parts.len() {
        0 => arena.zero,
        1 => const_parts[0],
        _ => arena.add(&const_parts),
    };
    let rhs = arena.neg(constant);
    let rhs = crate::transforms::eval::eval(arena, rhs);
    Some((coeffs, rhs))
}

/// Final clean-up of a solved value: simplify, and combine over a common
/// denominator when that is shorter.
fn tidy(e: &Ex) -> Ex {
    let s = e.eval().simplify();
    // Combine over a common denominator and expand the numerator so that
    // cancellations like -a*b + b*(a+1) → b are found.
    let (num, den) = s.together().as_numer_denom();
    let num = num.expand().eval();
    let t = if den.is_one_structural() {
        num
    } else {
        (&num / &den).eval()
    };
    // Prefer the single-fraction form unless it is clearly larger
    // (`count_ops` is DAG-based, so a shared denominator makes the split
    // form look deceptively small; allow one extra node for the fraction).
    if t.count_ops() <= s.count_ops() + 1 {
        t
    } else {
        s
    }
}

/// Light-weight normalisation used between elimination steps.
fn normalize(e: &Ex) -> Ex {
    let e1 = e.eval();
    if e1.is_zero_structural() || is_number(&e1) {
        return e1;
    }
    let e2 = e1.simplify_with(&SimplifyOpts::single_pass());
    if e2.count_ops() <= e1.count_ops() {
        e2
    } else {
        e1
    }
}

/// `true` if `e` is a numeric literal.
fn is_number(e: &Ex) -> bool {
    e.inner.read().arena.as_num(e.raw_id()).is_some()
}

/// `true` if `e` is a **nonzero** numeric literal.
fn is_nonzero_number(e: &Ex) -> bool {
    e.inner
        .read()
        .arena
        .as_num(e.raw_id())
        .is_some_and(|r| !r.is_zero())
}

/// Decide whether an (already normalised) entry is zero.
///
/// Structural zero → `true`; numeric literal → by value; a rational
/// function over `ℚ(radicals, i)` of the symbols → exactly
/// ([`rational_function_is_zero`](crate::domains::matrix::rational_function_is_zero),
/// which before 0.31 did not take `√2`: a zero mixing `√2` and a symbol
/// was then "nonzero" and a consistent system "Inconsistent"); otherwise a
/// full `simplify` pass, then the certified test for constants.  An entry
/// not recognised as zero is a pivot: the generic solution.
fn entry_is_zero(e: &Ex) -> bool {
    if e.is_zero_structural() {
        return true;
    }
    if is_number(e) {
        return false;
    }
    // A rational function of the symbols is decided exactly.
    if let Some(b) = crate::domains::matrix::rational_function_is_zero(e) {
        return b;
    }
    let s = e.simplify();
    if s.is_zero_structural() {
        return true;
    }
    // Purely numeric (no free symbols) but not folded: the certified zero
    // test (before 0.30, `|f64 value| < 10⁻¹²` counted as zero).
    if s.free_symbols().is_empty() && s.eval_complex64().is_ok() {
        let mut inner = s.inner.write();
        return crate::poly::algebraic::is_zero_checked(&mut inner.arena, s.raw_id()) == Some(true);
    }
    false
}

/// Symbolic reduced-row-echelon solve of `rows · x = rhs`.
///
/// `rows[i]` holds the coefficients of equation `i`; `rhs[i]` its
/// right-hand side.  `n_vars` is the number of unknowns; `unknowns` are the
/// symbols the free variables should be expressed with.
///
/// Pivot choice prefers nonzero numeric literals; a symbolic pivot is used
/// only when no numeric one is available and is then **assumed nonzero**
/// (generic solution).
pub(crate) fn rref_solve(
    rows: Vec<Vec<Ex>>,
    rhs: Vec<Ex>,
    unknowns: &[Ex],
) -> Result<SymbolicLinearResult, SymplexError> {
    let n_vars = unknowns.len();
    let m = rows.len();
    if m == 0 || n_vars == 0 {
        return Err(SymplexError::InvalidArgument {
            operation: "linsolve",
            reason: "need at least one equation and one unknown".into(),
        });
    }
    // Augmented matrix.
    let mut mat: Vec<Vec<Ex>> = rows
        .into_iter()
        .zip(rhs)
        .map(|(mut r, b)| {
            r.push(b);
            r
        })
        .collect();
    let ncols = n_vars + 1;
    for r in &mat {
        if r.len() != ncols {
            return Err(SymplexError::InvalidArgument {
                operation: "linsolve",
                reason: "row length does not match number of unknowns".into(),
            });
        }
    }

    if let Some(result) = rref_solve_rational(&mat, n_vars, unknowns) {
        return Ok(result);
    }
    if let Some(result) = rref_solve_algebraic(&mat, n_vars, unknowns) {
        return Ok(result);
    }

    let mut pivots: Vec<usize> = Vec::new();
    let mut pivot_row = 0usize;

    for col in 0..n_vars {
        if pivot_row >= m {
            break;
        }
        // Candidate pivots: prefer numeric nonzero, then simplest symbolic.
        let mut numeric: Option<usize> = None;
        let mut symbolic: Option<(usize, usize)> = None; // (row, ops)
        for (r, row) in mat.iter_mut().enumerate().skip(pivot_row) {
            let e = normalize(&row[col]);
            row[col] = e.clone();
            if is_nonzero_number(&e) {
                numeric = Some(r);
                break;
            }
            if !entry_is_zero(&e) {
                let ops = e.count_ops();
                if symbolic.is_none_or(|(_, o)| ops < o) {
                    symbolic = Some((r, ops));
                }
            } else {
                row[col] = e.context().zero();
            }
        }
        let found = match (numeric, symbolic) {
            (Some(r), _) => r,
            (None, Some((r, _))) => r,
            (None, None) => continue,
        };
        if found != pivot_row {
            mat.swap(pivot_row, found);
        }
        // Scale pivot row.
        let pv = mat[pivot_row][col].clone();
        for (j, entry) in mat[pivot_row].iter_mut().enumerate() {
            if j == col {
                *entry = pv.context().one();
            } else {
                let v = &*entry / &pv;
                *entry = normalize(&v);
            }
        }
        // Eliminate in all other rows.
        let pivot_vals = mat[pivot_row].clone();
        for (i, row) in mat.iter_mut().enumerate() {
            if i == pivot_row {
                continue;
            }
            let factor = normalize(&row[col]);
            if entry_is_zero(&factor) {
                row[col] = factor.context().zero();
                continue;
            }
            for (j, entry) in row.iter_mut().enumerate() {
                if j == col {
                    *entry = factor.context().zero();
                } else {
                    let t = &factor * &pivot_vals[j];
                    let v = &*entry - &t;
                    *entry = normalize(&v);
                }
            }
        }
        pivots.push(col);
        pivot_row += 1;
    }

    // Inconsistency check: rows with zero coefficients but nonzero rhs.
    for r in &mat {
        let all_zero = r[..n_vars].iter().all(entry_is_zero);
        if all_zero && !entry_is_zero(&r[n_vars]) {
            return Ok(SymbolicLinearResult {
                values: Vec::new(),
                free: Vec::new(),
                inconsistent: true,
            });
        }
    }

    let pivot_set: std::collections::HashSet<usize> = pivots.iter().copied().collect();
    let free: Vec<usize> = (0..n_vars).filter(|c| !pivot_set.contains(c)).collect();

    let mut values: Vec<Ex> = unknowns.to_vec();
    for (r, &pc) in pivots.iter().enumerate() {
        let mut v = mat[r][n_vars].clone();
        for &f in &free {
            if !entry_is_zero(&mat[r][f]) {
                let t = &mat[r][f] * &unknowns[f];
                v = &v - &t;
            }
        }
        values[pc] = tidy(&v);
    }

    Ok(SymbolicLinearResult {
        values,
        free,
        inconsistent: false,
    })
}

/// Exact fast path of [`rref_solve`] for an augmented matrix whose entries
/// are all rational literals: fraction-free elimination on a [`QMatrix`]
/// instead of expression arithmetic.  Returns `None` if any entry is
/// symbolic.
fn rref_solve_rational(
    mat: &[Vec<Ex>],
    n_vars: usize,
    unknowns: &[Ex],
) -> Option<SymbolicLinearResult> {
    let rows: Vec<Vec<Q>> = mat
        .iter()
        .map(|r| r.iter().map(Ex::as_rational).collect())
        .collect::<Option<_>>()?;
    let aug = QMatrix::new(rows).ok()?;
    let (r, pivots) = aug.rref_limited(n_vars);
    let rank = pivots.len();
    // Rows below the rank are zero in the coefficient part; a nonzero
    // right-hand side there means `0 = c`.
    if (rank..aug.nrows()).any(|i| !r[(i, n_vars)].is_zero()) {
        return Some(SymbolicLinearResult {
            values: Vec::new(),
            free: Vec::new(),
            inconsistent: true,
        });
    }
    let mut is_pivot = vec![false; n_vars];
    for &c in &pivots {
        is_pivot[c] = true;
    }
    let free: Vec<usize> = (0..n_vars).filter(|&c| !is_pivot[c]).collect();
    let ctx = unknowns.first()?.context();
    let mut values: Vec<Ex> = unknowns.to_vec();
    for (i, &pc) in pivots.iter().enumerate() {
        let mut v = ctx.from_ratio(r[(i, n_vars)].clone());
        for &f in &free {
            let c = &r[(i, f)];
            if !c.is_zero() {
                v -= ctx.from_ratio(c.clone()) * &unknowns[f];
            }
        }
        // Same clean-up as the symbolic path, so parametric solutions print
        // identically whichever route produced them.
        values[pc] = if free.is_empty() { v } else { tidy(&v) };
    }
    Some(SymbolicLinearResult {
        values,
        free,
        inconsistent: false,
    })
}

/// Solve the linear system `eqs = 0` for `vars` symbolically.
///
/// See [`rref_solve`] for the pivoting policy.  Returns
/// [`SymplexError::InvalidArgument`] if an equation is not linear in the
/// unknowns or the input is empty.
pub(crate) fn linsolve_symbolic(
    eqs: &[Ex],
    vars: &[Ex],
) -> Result<SymbolicLinearResult, SymplexError> {
    if eqs.is_empty() || vars.is_empty() {
        return Err(SymplexError::InvalidArgument {
            operation: "linsolve",
            reason: "need at least one equation and one unknown".into(),
        });
    }
    let first = &eqs[0];
    let var_ids: Vec<ExprId> = vars.iter().map(|v| first.checked_id(v)).collect();
    let eq_ids: Vec<ExprId> = eqs.iter().map(|e| first.checked_id(e)).collect();

    let mut rows: Vec<Vec<ExprId>> = Vec::with_capacity(eqs.len());
    let mut rhs: Vec<ExprId> = Vec::with_capacity(eqs.len());
    {
        let mut guard = first.inner.write();
        let arena = &mut guard.arena;
        for (k, &eq) in eq_ids.iter().enumerate() {
            match extract_linear_row(arena, eq, &var_ids) {
                Some((coeffs, b)) => {
                    rows.push(coeffs);
                    rhs.push(b);
                }
                None => {
                    let shown = arena.display(eq).to_string();
                    drop(guard);
                    return Err(SymplexError::InvalidArgument {
                        operation: "linsolve",
                        reason: format!("equation {k} is not linear in the unknowns: {shown}"),
                    });
                }
            }
        }
    }
    let rows: Vec<Vec<Ex>> = rows
        .into_iter()
        .map(|r| r.into_iter().map(|id| first.wrap(id)).collect())
        .collect();
    let rhs: Vec<Ex> = rhs.into_iter().map(|id| first.wrap(id)).collect();
    rref_solve(rows, rhs, vars)
}

// ═══════════════════════════════════════════════════════════════════════════
// Rational functions over ℚ(radicals, i)
// ═══════════════════════════════════════════════════════════════════════════

type Mp = MultiPoly<GrevLex>;

/// Largest common index `L` of the radicals a [`Tower`] takes.
const MAX_ROOT_INDEX: u32 = 12;

/// Largest integer exponent (and radical numerator) a [`Tower`] reads.
const MAX_POWER: u64 = 64;

/// Radicands are factored by trial division below this bound; a cofactor
/// below its square is prime.
const TRIAL_DIVISION_BOUND: u64 = 1 << 16;

/// Largest entry (in terms of numerator or denominator) of the exact
/// elimination of [`rref_solve_algebraic`].
const KFRAC_TERM_BUDGET: usize = 20_000;

/// The field of constants of expressions built by `+`, `·` and integer
/// powers from rationals, symbols (the parameters), radicals `q^(s/t)` of
/// rationals, the golden ratio and `i`:
/// `K = ℚ(p₁^(1/L), …, pₖ^(1/L))` or `K(i)`, with `pⱼ` the primes of the
/// radicands, `L` the lcm of the root indices and `uⱼ = pⱼ^(1/L)` the
/// positive real root.  A radical `q^(s/t)` of a positive `q` is a rational
/// multiple of a monomial `∏ uⱼ^(eⱼ)` with `0 ≤ eⱼ < L`, and
/// `(−q)^(s/2) = q^(s/2)·i^s` (principal branch).
///
/// An element of `K[params]` is a polynomial over ℚ in the parameters and
/// the generators (`uⱼ`, `i`), kept **reduced** (`uⱼ`-degree `< L`,
/// `i`-degree `< 2`, by `uⱼ^L = pⱼ` and `i² = −1`).  The reduced form is
/// canonical: the monomials `∏ uⱼ^(eⱼ)`, `0 ≤ eⱼ < L`, are linearly
/// independent over ℚ (Besicovitch, *On the linear independence of
/// fractional powers of integers*, J. London Math. Soc. 15 (1940) 3–6), so
/// they are a basis of the real field `ℚ(u₁, …, uₖ)`, and `{1, i}` is a
/// basis of `K(i)` over it.  So an element is zero iff its reduced
/// polynomial is, and `K[params]` is an integral domain in which a product
/// of nonzero reduced polynomials reduces to a nonzero one.
pub(crate) struct Tower {
    /// The parameters: polynomial variables `0..syms.len()`.
    syms: Vec<ExprId>,
    /// The radicand primes: generator `uⱼ` is variable `syms.len() + j`.
    primes: Vec<BigInt>,
    /// The common root index (1 when there is no radical).
    l: u32,
    /// Is `i` a generator (the last variable)?
    has_i: bool,
    /// Each radical node (and the golden ratio) as a reduced polynomial.
    atoms: FxHashMap<ExprId, Mp>,
    /// A nested square root `w = √D` with `D ∈ K` a constant (square-root
    /// towers only): the last variable, reduced by `w² = D`.  With it the
    /// reduced form is not canonical when `D` is a square in `K`; the zero
    /// test decides `P + Q·w` by [`Tower::decide`].
    ext: Option<Mp>,
}

/// `N/D` over `K[params]`; `None` if a node is not field arithmetic or a
/// polynomial exceeds the budget, `Some(None)` on a division by zero.
type FracResult = Option<Option<(Mp, Mp)>>;

/// The prime factorisation of a positive integer below `2⁶⁴` by trial
/// division, or `None` when a cofactor is left that the trial division
/// bound does not prove prime.
fn small_factorization(n: &BigInt) -> Option<Vec<(BigInt, u32)>> {
    let mut m = n.to_u64()?;
    if m == 0 {
        return None;
    }
    let mut out = Vec::new();
    let mut p = 2u64;
    while p * p <= m && p < TRIAL_DIVISION_BOUND {
        if m % p == 0 {
            let mut e = 0u32;
            while m % p == 0 {
                m /= p;
                e += 1;
            }
            out.push((BigInt::from(p), e));
        }
        p += if p == 2 { 1 } else { 2 };
    }
    if m > 1 {
        if p * p <= m {
            return None;
        }
        out.push((BigInt::from(m), 1));
    }
    Some(out)
}

fn gcd_u32(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

impl Tower {
    /// The tower of the expressions `roots`; `None` if a node is not
    /// field arithmetic over `ℚ(radicals, i)(symbols)` (a function, `π`,
    /// a radical of a symbol, a root of a negative number of index > 2, a
    /// radicand that is not factored by trial division, `L > 12`).
    fn scan(arena: &Arena, roots: &[ExprId]) -> Option<Tower> {
        let mut syms: Vec<ExprId> = Vec::new();
        // (node, radicand, s, t) for `radicand^(s/t)`
        let mut radicals: Vec<(ExprId, Q, i64, u32)> = Vec::new();
        let mut golden: Vec<ExprId> = Vec::new();
        // (node, s) for `base^(s/2)` with a non-rational base, one base only
        let mut nested: Vec<(ExprId, i64)> = Vec::new();
        let mut nested_base: Option<ExprId> = None;
        let mut has_i = false;
        let mut seen: FxHashSet<ExprId> = FxHashSet::default();
        let mut stack: Vec<ExprId> = roots.to_vec();
        while let Some(id) = stack.pop() {
            if !seen.insert(id) {
                continue;
            }
            match arena.node(id) {
                ExprNode::Num(_) => {}
                ExprNode::Symbol(_) => syms.push(id),
                ExprNode::ImaginaryUnit => has_i = true,
                ExprNode::GoldenRatio => golden.push(id),
                ExprNode::Add(ch) | ExprNode::Mul(ch) => stack.extend(ch.iter().copied()),
                ExprNode::Neg(b) => stack.push(*b),
                ExprNode::Pow(b, x) => {
                    let x = arena.as_num(*x)?;
                    if x.is_integer() {
                        if x.to_integer().to_i64()?.unsigned_abs() > MAX_POWER {
                            return None;
                        }
                        stack.push(*b);
                    } else if let Some(q) = arena.as_num(*b) {
                        let q = q.clone();
                        let s = x.numer().to_i64()?;
                        let t = x.denom().to_u32()?;
                        if q.is_zero() || s.unsigned_abs() > MAX_POWER {
                            return None;
                        }
                        radicals.push((id, q, s, t));
                    } else {
                        let s = x.numer().to_i64()?;
                        if x.denom().to_u32()? != 2
                            || s.unsigned_abs() > MAX_POWER
                            || nested_base.is_some_and(|nb| nb != *b)
                        {
                            return None;
                        }
                        nested_base = Some(*b);
                        nested.push((id, s));
                        stack.push(*b);
                    }
                }
                _ => return None,
            }
        }
        let mut l: u32 = if golden.is_empty() { 1 } else { 2 };
        for (_, q, _, t) in &radicals {
            if q.is_negative() {
                if *t != 2 {
                    return None;
                }
                has_i = true;
            }
            l = l / gcd_u32(l, *t) * *t;
            if l > MAX_ROOT_INDEX {
                return None;
            }
        }
        // Factor the radicands (exponent of each prime in |q|).
        let mut primes: Vec<BigInt> = Vec::new();
        let mut factored: Vec<Vec<(BigInt, i64)>> = Vec::with_capacity(radicals.len());
        for (_, q, _, _) in &radicals {
            let mut f: Vec<(BigInt, i64)> = Vec::new();
            for (p, e) in small_factorization(&q.numer().abs())? {
                f.push((p, i64::from(e)));
            }
            for (p, e) in small_factorization(q.denom())? {
                f.push((p, -i64::from(e)));
            }
            for (p, _) in &f {
                if !primes.contains(p) {
                    primes.push(p.clone());
                }
            }
            factored.push(f);
        }
        let five = BigInt::from(5);
        if !golden.is_empty() && !primes.contains(&five) {
            primes.push(five.clone());
        }
        primes.sort();
        syms.sort();
        let ns = syms.len();
        let np = primes.len();
        if nested_base.is_some() && l > 2 {
            return None;
        }
        let nv = ns + np + usize::from(has_i) + usize::from(nested_base.is_some());
        let li = i64::from(l);
        let mut atoms: FxHashMap<ExprId, Mp> = FxHashMap::default();
        for ((id, q, s, t), f) in radicals.iter().zip(&factored) {
            // |q|^(s/t) = ∏ p^(k·s/t) = ∏ u_p^(k·s·L/t)
            let mut c = Q::one();
            let mut exps = vec![0u32; nv];
            for (p, k) in f {
                let j = primes.binary_search(p).ok()?;
                let e = k.checked_mul(*s)?.checked_mul(li / i64::from(*t))?;
                let whole = u32::try_from(e.div_euclid(li).unsigned_abs()).ok()?;
                let pw = Q::from_integer(p.pow(whole));
                if e.div_euclid(li) >= 0 {
                    c *= pw;
                } else {
                    c /= pw;
                }
                exps[ns + j] += u32::try_from(e.rem_euclid(li)).ok()?;
            }
            if q.is_negative() {
                // (−|q|)^(s/2) = |q|^(s/2)·i^s with s odd.
                if s.rem_euclid(4) == 3 {
                    c = -c;
                }
                exps[ns + np] = 1;
            }
            let m = Mp::from_terms(nv, vec![(exps, c)])?;
            atoms.insert(*id, m);
        }
        let tower = Tower {
            syms,
            primes,
            l,
            has_i,
            atoms,
            ext: None,
        };
        // The atoms may carry exponents ≥ L after the merge (a prime of
        // several radicands never does, but reduce for safety).
        let atoms: FxHashMap<ExprId, Mp> = tower
            .atoms
            .iter()
            .map(|(id, m)| (*id, tower.reduce(m.clone())))
            .collect();
        let mut tower = Tower { atoms, ..tower };
        if !golden.is_empty() {
            // φ = 1/2 + √5/2
            let j = tower.primes.binary_search(&five).ok()?;
            let mut e = vec![0u32; nv];
            e[ns + j] = l / 2;
            let half = Q::new(BigInt::from(1), BigInt::from(2));
            let phi = Mp::from_terms(nv, vec![(vec![0u32; nv], half.clone()), (e, half)])?;
            for g in golden {
                tower.atoms.insert(g, phi.clone());
            }
        }
        if let Some(b) = nested_base {
            // D = value of the radicand, a constant of K (the placeholder
            // keeps the variable count while it is evaluated).
            tower.ext = Some(Mp::zero(nv));
            let (n, d) = tower.eval_frac(arena, b, None, RF_TERM_BUDGET)??;
            if !tower.in_k(&n) || !tower.in_k(&d) {
                return None;
            }
            let dd = tower.mul_r(&n, &tower.inv_k(&d)?);
            let w = Mp::var(nv, nv - 1);
            let dinv = if dd.is_zero() { None } else { tower.inv_k(&dd) };
            let mut vals: Vec<(ExprId, Mp)> = Vec::with_capacity(nested.len());
            for (id, s) in nested {
                // b^(s/2) = (√b)^s = w·D^((s−1)/2), s odd.
                let v = if dd.is_zero() {
                    if s < 0 {
                        return None;
                    }
                    Mp::zero(nv)
                } else {
                    let k = (s - 1).div_euclid(2);
                    let base = if k >= 0 { dd.clone() } else { dinv.clone()? };
                    tower.mul_r(&w, &tower.pow_r(&base, k.unsigned_abs()))
                };
                vals.push((id, v));
            }
            tower.ext = Some(dd);
            for (id, v) in vals {
                tower.atoms.insert(id, v);
            }
        }
        Some(tower)
    }

    fn nvars(&self) -> usize {
        self.syms.len()
            + self.primes.len()
            + usize::from(self.has_i)
            + usize::from(self.ext.is_some())
    }

    /// The zero decision for a reduced numerator `n`.  Without a nested
    /// root the reduced form is canonical: zero iff `n = 0`.  With `w = √D`,
    /// `n = P + Q·w` (`P`, `Q` free of `w`) is zero iff `Q = 0` and `P = 0`,
    /// or `P² = Q²·D` (so `r = −P/Q` is a square root of `D`, a constant of
    /// `K`) and `r` is the principal one (`Re r > 0`, or `Re r = 0 < Im r`),
    /// decided exactly by [`real_sign`](Self::real_sign).  `None` if a sign
    /// is not decided.
    fn decide(&self, n: &Mp) -> Option<bool> {
        let Some(dd) = &self.ext else {
            return Some(n.is_zero());
        };
        let nv = self.nvars();
        let wv = nv - 1;
        let split = |k: u32| -> Option<Mp> {
            let terms: Vec<(Vec<u32>, Q)> = n
                .terms()
                .filter(|(e, _)| e[wv] == k)
                .map(|(e, c)| {
                    let mut e = e.to_vec();
                    e[wv] = 0;
                    (e, c.clone())
                })
                .collect();
            Mp::from_terms(nv, terms)
        };
        let (p, q) = (split(0)?, split(1)?);
        if q.is_zero() {
            return Some(p.is_zero());
        }
        let r = self.mul_r(&p, &p).sub(&self.mul_r(&self.mul_r(&q, &q), dd));
        if !r.is_zero() {
            return Some(false);
        }
        // r = −P/Q is constant: read it at a point where Q ≠ 0.
        let ns = self.syms.len();
        let at = |f: &Mp, k: u64| -> Mp {
            let mut g = f.clone();
            for i in 0..ns {
                g = g.substitute(i, &rf_sample_point(i, k));
            }
            g
        };
        let (pk, qk) = (0..RF_POINTS)
            .map(|k| (at(&p, k), at(&q, k)))
            .find(|(_, qk)| !qk.is_zero())?;
        let root = self.mul_r(&pk.neg(), &self.inv_k(&qk)?);
        let iv = self.has_i.then(|| ns + self.primes.len());
        let part = |k: u32| -> Option<Mp> {
            let terms: Vec<(Vec<u32>, Q)> = root
                .terms()
                .filter(|(e, _)| iv.map_or(k == 0, |v| e[v] == k))
                .map(|(e, c)| {
                    let mut e = e.to_vec();
                    if let Some(v) = iv {
                        e[v] = 0;
                    }
                    (e, c.clone())
                })
                .collect();
            Mp::from_terms(nv, terms)
        };
        let re = self.real_sign(&part(0)?)?;
        let im = self.real_sign(&part(1)?)?;
        Some(re > 0 || (re == 0 && im > 0))
    }

    /// The sign of a real constant of `K` (a reduced polynomial in the
    /// square roots `uⱼ = √pⱼ`): interval evaluation with rational
    /// enclosures `⌊√(p·4ᵏ)⌋/2ᵏ ≤ uⱼ < (⌊√(p·4ᵏ)⌋ + 1)/2ᵏ`, refined until
    /// the interval excludes 0 (the canonical form of a nonzero element is
    /// nonzero, so this ends); `None` past 4096 bits.
    fn real_sign(&self, p: &Mp) -> Option<i8> {
        if p.is_zero() {
            return Some(0);
        }
        let ns = self.syms.len();
        let np = self.primes.len();
        let mut bits: u32 = 64;
        while bits <= 4096 {
            let scale = BigInt::from(1) << bits;
            let enclosures: Vec<(Q, Q)> = self
                .primes
                .iter()
                .map(|pr| {
                    let s = (pr * &scale * &scale).sqrt();
                    (
                        Q::new(s.clone(), scale.clone()),
                        Q::new(s + 1, scale.clone()),
                    )
                })
                .collect();
            let (mut lo, mut hi) = (Q::zero(), Q::zero());
            for (e, c) in p.terms() {
                let (mut mlo, mut mhi) = (Q::one(), Q::one());
                for j in 0..np {
                    for _ in 0..e[ns + j] {
                        mlo *= &enclosures[j].0;
                        mhi *= &enclosures[j].1;
                    }
                }
                if c.is_negative() {
                    lo += c * &mhi;
                    hi += c * &mlo;
                } else {
                    lo += c * &mlo;
                    hi += c * &mhi;
                }
            }
            if lo.is_positive() {
                return Some(1);
            }
            if hi.is_negative() {
                return Some(-1);
            }
            bits *= 2;
        }
        None
    }

    /// Does the tower have an algebraic generator (a radical or `i`)?
    fn has_generators(&self) -> bool {
        !self.primes.is_empty() || self.has_i
    }

    /// The reduced form of `p` (`uⱼ^L → pⱼ`, `i² → −1`, `w² → D`).
    fn reduce(&self, p: Mp) -> Mp {
        let p = self.reduce_generators(p);
        let Some(dd) = &self.ext else {
            return p;
        };
        let wv = self.nvars() - 1;
        if !p.terms().any(|(e, _)| e[wv] >= 2) {
            return p;
        }
        // c·m·w^e → c·m·w^(e mod 2)·D^(e div 2); D is free of w.
        let mut out = Mp::zero(p.num_vars());
        for (e, c) in p.terms() {
            let mut e = e.to_vec();
            let k = e[wv] / 2;
            e[wv] %= 2;
            let Some(t) = Mp::from_terms(p.num_vars(), vec![(e, c.clone())]) else {
                continue;
            };
            let t = if k > 0 {
                self.reduce_generators(t.mul(&self.pow_r(dd, u64::from(k))))
            } else {
                t
            };
            out = out.add(&t);
        }
        out
    }

    /// [`reduce`](Self::reduce) for the radicals and `i`.
    fn reduce_generators(&self, p: Mp) -> Mp {
        let ns = self.syms.len();
        let np = self.primes.len();
        let l = self.l;
        let over = |e: &[u32]| (0..np).any(|j| e[ns + j] >= l) || (self.has_i && e[ns + np] >= 2);
        if !p.terms().any(|(e, _)| over(e)) {
            return p;
        }
        let terms: Vec<(Vec<u32>, Q)> = p
            .terms()
            .map(|(e, c)| {
                let mut e = e.to_vec();
                let mut c = c.clone();
                for j in 0..np {
                    let k = e[ns + j] / l;
                    if k > 0 {
                        c *= Q::from_integer(self.primes[j].pow(k));
                        e[ns + j] %= l;
                    }
                }
                if self.has_i {
                    if (e[ns + np] / 2) % 2 == 1 {
                        c = -c;
                    }
                    e[ns + np] %= 2;
                }
                (e, c)
            })
            .collect();
        Mp::from_terms(p.num_vars(), terms).unwrap_or(p)
    }

    fn mul_r(&self, a: &Mp, b: &Mp) -> Mp {
        self.reduce(a.mul(b))
    }

    fn pow_r(&self, a: &Mp, n: u64) -> Mp {
        let mut acc = Mp::from_int(a.num_vars(), 1);
        let mut base = a.clone();
        let mut k = n;
        while k > 0 {
            if k & 1 == 1 {
                acc = self.mul_r(&acc, &base);
            }
            k >>= 1;
            if k > 0 {
                base = self.mul_r(&base, &base);
            }
        }
        acc
    }

    /// `root` as `N/D` over `K[params]` (both reduced), the parameters kept
    /// as variables (`point = None`) or set to the `k`-th sample point of
    /// [`rf_sample_point`].
    fn eval_frac(
        &self,
        arena: &Arena,
        root: ExprId,
        point: Option<u64>,
        budget: usize,
    ) -> FracResult {
        let nv = self.nvars();
        let one = Mp::from_int(nv, 1);
        let sym_index: FxHashMap<ExprId, usize> =
            self.syms.iter().enumerate().map(|(i, &s)| (s, i)).collect();
        let mut val: FxHashMap<ExprId, (Mp, Mp)> = FxHashMap::default();
        let mut stack = vec![(root, false)];
        while let Some((id, ready)) = stack.pop() {
            if val.contains_key(&id) {
                continue;
            }
            let node = arena.node(id);
            let atom = self.atoms.get(&id);
            let children: Vec<ExprId> = match node {
                _ if atom.is_some() => Vec::new(),
                ExprNode::Add(ch) | ExprNode::Mul(ch) => ch.to_vec(),
                ExprNode::Pow(b, _) | ExprNode::Neg(b) => vec![*b],
                _ => Vec::new(),
            };
            if !ready && !children.is_empty() {
                stack.push((id, true));
                stack.extend(children.into_iter().map(|c| (c, false)));
                continue;
            }
            let r = if let Some(a) = atom {
                (a.clone(), one.clone())
            } else {
                match node {
                    ExprNode::Num(_) => (Mp::constant(nv, arena.as_num(id)?.clone()), one.clone()),
                    ExprNode::Symbol(_) => {
                        let i = *sym_index.get(&id)?;
                        let v = match point {
                            Some(k) => Mp::constant(nv, rf_sample_point(i, k)),
                            None => Mp::var(nv, i),
                        };
                        (v, one.clone())
                    }
                    ExprNode::ImaginaryUnit if self.has_i => (
                        Mp::var(nv, self.syms.len() + self.primes.len()),
                        one.clone(),
                    ),
                    ExprNode::Neg(b) => {
                        let (n, d) = val.get(b)?;
                        (n.neg(), d.clone())
                    }
                    ExprNode::Add(ch) => {
                        let (mut n, mut d) = (Mp::zero(nv), one.clone());
                        for c in ch {
                            let (cn, cd) = val.get(c)?;
                            if *cd == d {
                                n = n.add(cn);
                            } else {
                                n = self.mul_r(&n, cd).add(&self.mul_r(cn, &d));
                                d = self.mul_r(&d, cd);
                            }
                            if n.num_terms().max(d.num_terms()) > budget {
                                return None;
                            }
                        }
                        (n, d)
                    }
                    ExprNode::Mul(ch) => {
                        let (mut n, mut d) = (one.clone(), one.clone());
                        for c in ch {
                            let (cn, cd) = val.get(c)?;
                            n = self.mul_r(&n, cn);
                            d = self.mul_r(&d, cd);
                            if n.num_terms().max(d.num_terms()) > budget {
                                return None;
                            }
                        }
                        (n, d)
                    }
                    ExprNode::Pow(b, x) => {
                        let k = arena.as_num(*x)?.to_integer().to_i64()?;
                        let (bn, bd) = val.get(b)?;
                        if k >= 0 {
                            (
                                self.pow_r(bn, k.unsigned_abs()),
                                self.pow_r(bd, k.unsigned_abs()),
                            )
                        } else {
                            if bn.is_zero() {
                                return Some(None);
                            }
                            (
                                self.pow_r(bd, k.unsigned_abs()),
                                self.pow_r(bn, k.unsigned_abs()),
                            )
                        }
                    }
                    _ => return None,
                }
            };
            if r.0.num_terms().max(r.1.num_terms()) > budget {
                return None;
            }
            val.insert(id, r);
        }
        Some(val.remove(&root))
    }

    /// Exact zero test of `root` in `K(params)`: evaluated at the sample
    /// points of the parameters first (a nonzero value proves `≢ 0`); at a
    /// zero value the numerator is computed as a reduced polynomial, and
    /// only when that exceeds [`RF_TERM_BUDGET`] terms is the answer taken
    /// from the values at all the points (Schwartz–Zippel), as in
    /// [`rational_function_is_zero`](crate::domains::matrix::rational_function_is_zero).
    fn is_zero(&self, arena: &Arena, root: ExprId) -> Option<bool> {
        if self.syms.is_empty() || self.ext.is_some() {
            // (With a nested root a nonzero value at a point proves
            // nothing: the representation is not canonical.)
            let (n, _) = self.eval_frac(arena, root, None, RF_TERM_BUDGET)??;
            return self.decide(&n);
        }
        let mut zero_at_all = true;
        for k in 0..RF_POINTS {
            match self.eval_frac(arena, root, Some(k), RF_TERM_BUDGET)? {
                Some((n, _)) if !n.is_zero() => return Some(false),
                Some(_) => {}
                None => zero_at_all = false,
            }
            if let Some(Some((n, _))) = self.eval_frac(arena, root, None, RF_TERM_BUDGET) {
                return Some(n.is_zero());
            }
        }
        zero_at_all.then_some(true)
    }

    // ── The field K(params) for square roots (L ≤ 2) ────────────────────────

    /// The conjugate of `p` flipping the sign of the generator variable `v`
    /// (an automorphism of `K` when `L ≤ 2`).
    fn conjugate(p: &Mp, v: usize) -> Mp {
        let terms: Vec<(Vec<u32>, Q)> = p
            .terms()
            .map(|(e, c)| {
                (
                    e.to_vec(),
                    if e[v] % 2 == 1 { -c.clone() } else { c.clone() },
                )
            })
            .collect();
        Mp::from_terms(p.num_vars(), terms).unwrap_or_else(|| p.clone())
    }

    /// The normal form of `n/d` (`d ≠ 0`, both reduced): the denominator
    /// rationalised into `ℚ[params]` (multiplied by its conjugates), the
    /// common factor cancelled (a factor of `d` divides `n` in `K[params]`
    /// iff it divides every generator coefficient of `n`, so the gcd over
    /// `ℚ[params, generators]` is the right one), and `d` made primitive over
    /// ℤ with a positive leading coefficient.  Unique for each element of
    /// `K(params)`.  `None` if `d = 0` or a polynomial exceeds the budget.
    fn normalize(&self, n: Mp, d: Mp) -> Option<KFrac> {
        if d.is_zero() {
            return None;
        }
        let nv = self.nvars();
        let (mut n, mut d) = (n, d);
        for v in self.syms.len()..nv {
            if d.degree_in(v) > 0 {
                let c = Self::conjugate(&d, v);
                n = self.mul_r(&n, &c);
                d = self.mul_r(&d, &c);
            }
        }
        if n.is_zero() {
            return Some(KFrac::zero(nv));
        }
        if d.as_constant().is_none() {
            // gcd(d, N_m for every generator monomial m), stopping at 1, in
            // the ring of the parameters alone (all of these are free of
            // the generators).
            let ns = self.syms.len();
            let mut g = self.project(&d, ns)?;
            for comp in self.generator_components(&n) {
                g = Mp::gcd(&g, &self.project(&comp, ns)?);
                if g.as_constant().is_some() {
                    break;
                }
            }
            if g.as_constant().is_none() {
                let g = self.embed(&g, nv)?;
                n = n.div_exact(&g)?;
                d = d.div_exact(&g)?;
            }
        }
        let dp = d.primitive_part_q();
        let (dlc, plc) = (d.leading_coeff()?.clone(), dp.leading_coeff()?.clone());
        // d = (dlc/plc)·dp; keep the positive-leading primitive part.
        let mut scale = plc.clone() / dlc;
        let mut dp = dp;
        if plc.is_negative() {
            dp = dp.neg();
            scale = -scale;
        }
        let n = n.scale(&scale);
        if n.num_terms().max(dp.num_terms()) > KFRAC_TERM_BUDGET {
            return None;
        }
        Some(KFrac { num: n, den: dp })
    }

    /// The coefficients of `p` at each generator monomial, as polynomials
    /// in the parameters (the generator exponents set to 0).
    fn generator_components(&self, p: &Mp) -> Vec<Mp> {
        let ns = self.syms.len();
        let mut groups: std::collections::BTreeMap<Vec<u32>, Vec<(Vec<u32>, Q)>> =
            std::collections::BTreeMap::new();
        for (e, c) in p.terms() {
            let mut pe = e.to_vec();
            for k in pe.iter_mut().skip(ns) {
                *k = 0;
            }
            groups
                .entry(e[ns..].to_vec())
                .or_default()
                .push((pe, c.clone()));
        }
        groups
            .into_values()
            .filter_map(|t| Mp::from_terms(p.num_vars(), t))
            .collect()
    }

    /// `p` (free of the generators) in the ring of the first `ns`
    /// variables.
    fn project(&self, p: &Mp, ns: usize) -> Option<Mp> {
        Mp::from_terms(
            ns,
            p.terms()
                .map(|(e, c)| (e[..ns].to_vec(), c.clone()))
                .collect(),
        )
    }

    /// `p` over the first variables back in the ring of `nv` variables.
    fn embed(&self, p: &Mp, nv: usize) -> Option<Mp> {
        Mp::from_terms(
            nv,
            p.terms()
                .map(|(e, c)| {
                    let mut e = e.to_vec();
                    e.resize(nv, 0);
                    (e, c.clone())
                })
                .collect(),
        )
    }

    /// Is `p` free of the parameters (an element of `K`)?
    fn in_k(&self, p: &Mp) -> bool {
        let ns = self.syms.len();
        p.terms().all(|(e, _)| e[..ns].iter().all(|&k| k == 0))
    }

    /// The inverse in `K` of a nonzero element `c` of `K` (square-root
    /// towers): `c·c′ = N(c)` is rational for `c′` the product of the
    /// conjugates, so `1/c = c′/N(c)`.
    fn inv_k(&self, c: &Mp) -> Option<Mp> {
        let nv = self.nvars();
        let mut prod = c.clone();
        let mut cof = Mp::from_int(nv, 1);
        for v in self.syms.len()..nv {
            if prod.degree_in(v) > 0 {
                let conj = Self::conjugate(&prod, v);
                cof = self.mul_r(&cof, &conj);
                prod = self.mul_r(&prod, &conj);
            }
        }
        let norm = prod.as_constant()?;
        if norm.is_zero() {
            return None;
        }
        Some(cof.scale(&(Q::one() / norm)))
    }

    /// `num/den` (`den ≠ 0`, both reduced) for output: the polynomial
    /// quotient when `den` divides `num` in `K[params]`, otherwise the
    /// fraction with the gcd over `ℚ[params, generators]` cancelled and the
    /// denominator primitive with a positive leading coefficient.  The
    /// denominator is not rationalised (a rationalised denominator has
    /// `2^k` times the degree in `k` generators; SymPy's `linsolve` also
    /// leaves algebraic numbers in denominators).
    fn quotient(&self, num: &Mp, den: &Mp) -> Option<KFrac> {
        let nv = self.nvars();
        if num.is_zero() {
            return Some(KFrac::zero(nv));
        }
        if let Some(q) = self.div_exact_k(num, den) {
            return Some(KFrac {
                num: q,
                den: Mp::from_int(nv, 1),
            });
        }
        let (mut n, mut d) = (num.clone(), den.clone());
        let g = Mp::gcd(&n, &d);
        if g.as_constant().is_none() {
            n = n.div_exact(&g)?;
            d = d.div_exact(&g)?;
        }
        let dp = d.primitive_part_q();
        let (dlc, plc) = (d.leading_coeff()?.clone(), dp.leading_coeff()?.clone());
        let mut scale = plc.clone() / dlc;
        let mut dp = dp;
        if plc.is_negative() {
            dp = dp.neg();
            scale = -scale;
        }
        Some(KFrac {
            num: n.scale(&scale),
            den: dp,
        })
    }

    /// `a/b` in `K[params]` when `b` divides `a` (square-root towers), by
    /// division with respect to the leading parameter monomial of `b`,
    /// whose coefficient is inverted in `K`; `None` if the division leaves
    /// a remainder or a polynomial exceeds the budget.
    fn div_exact_k(&self, a: &Mp, b: &Mp) -> Option<Mp> {
        let ns = self.syms.len();
        let nv = self.nvars();
        if b.is_zero() {
            return None;
        }
        if a.is_zero() {
            return Some(Mp::zero(nv));
        }
        // Leading parameter exponent (graded, then lexicographic) and its
        // coefficient in K.
        let lead = |p: &Mp| -> Option<(Vec<u32>, Mp)> {
            let key = |e: &[u32]| {
                let s: u32 = e[..ns].iter().sum();
                (s, e[..ns].to_vec())
            };
            let (_, best) = p.terms().map(|(e, _)| key(e)).max()?;
            let terms: Vec<(Vec<u32>, Q)> = p
                .terms()
                .filter(|(e, _)| e[..ns] == best[..])
                .map(|(e, c)| {
                    let mut e = e.to_vec();
                    for k in e.iter_mut().take(ns) {
                        *k = 0;
                    }
                    (e, c.clone())
                })
                .collect();
            Some((best, Mp::from_terms(nv, terms)?))
        };
        // Degree filter: a quotient needs deg_v a ≥ deg_v b in every
        // parameter.
        if (0..ns).any(|v| a.degree_in(v) < b.degree_in(v)) {
            return None;
        }
        let (lb_exp, lb_coef) = lead(b)?;
        let lb_inv = self.inv_k(&lb_coef)?;
        let mut r = a.clone();
        let mut q = Mp::zero(nv);
        while !r.is_zero() {
            let (lr_exp, lr_coef) = lead(&r)?;
            if lr_exp.iter().zip(&lb_exp).any(|(x, y)| x < y) {
                return None;
            }
            let mut shift = vec![0u32; nv];
            for k in 0..ns {
                shift[k] = lr_exp[k] - lb_exp[k];
            }
            let t = self
                .mul_r(&lr_coef, &lb_inv)
                .mul_monomial(&Q::one(), &shift);
            r = r.sub(&self.mul_r(&t, b));
            q = q.add(&t);
            if r.num_terms().max(q.num_terms()) > KFRAC_TERM_BUDGET {
                return None;
            }
        }
        Some(q)
    }

    /// `f` as an expression: `(Σ P_m(params)·m) / D` over the generator
    /// monomials `m` (products of `√pⱼ` and `i`).  `handle` is any
    /// expression of the context the parameters live in.
    fn frac_to_ex(&self, f: &KFrac, handle: &Ex) -> Ex {
        let ctx = handle.context();
        let ns = self.syms.len();
        let syms: Vec<Ex> = self.syms.iter().map(|&s| handle.wrap(s)).collect();
        // One n-ary sum (and product per term): adding term by term would
        // re-canonicalise the partial sum each time.
        let poly_ex = |terms: &[(Vec<u32>, Q)]| -> Ex {
            let items = terms.iter().map(|(e, c)| {
                let mut factors = vec![ctx.from_ratio(c.clone())];
                for (k, &p) in e.iter().enumerate() {
                    if p > 0 {
                        factors.push(syms[k].powi(i64::from(p)));
                    }
                }
                Ex::product_of(&ctx, factors)
            });
            Ex::sum_of(&ctx, items)
        };
        // Σ over generator monomials m of P_m(params)·m.
        let to_ex = |p: &Mp| -> Ex {
            let mut groups: std::collections::BTreeMap<Vec<u32>, Vec<(Vec<u32>, Q)>> =
                std::collections::BTreeMap::new();
            for (e, c) in p.terms() {
                groups
                    .entry(e[ns..].to_vec())
                    .or_default()
                    .push((e[..ns].to_vec(), c.clone()));
            }
            let mut parts: Vec<Ex> = Vec::with_capacity(groups.len());
            for (g, terms) in &groups {
                let mut factors = vec![poly_ex(terms)];
                for (j, p) in self.primes.iter().enumerate() {
                    if g[j] > 0 {
                        let root = ctx
                            .from_ratio(Q::from_integer(p.clone()))
                            .sqrt()
                            .powi(i64::from(g[j]));
                        factors.push(root);
                    }
                }
                if self.has_i && g[self.primes.len()] > 0 {
                    factors.push(ctx.i_unit());
                }
                parts.push(Ex::product_of(&ctx, factors));
            }
            Ex::sum_of(&ctx, parts)
        };
        let num = to_ex(&f.num);
        let den = to_ex(&f.den);
        if den.is_one_structural() {
            num
        } else {
            &num / &den
        }
    }
}

/// An element `N/D` of `K(params)` (square-root towers only): in the normal
/// form of [`Tower::normalize`], or as produced by [`Tower::quotient`].
#[derive(Clone, Debug, PartialEq)]
struct KFrac {
    num: Mp,
    den: Mp,
}

impl KFrac {
    fn zero(nv: usize) -> KFrac {
        KFrac {
            num: Mp::zero(nv),
            den: Mp::from_int(nv, 1),
        }
    }
}

/// A Gaussian rational `re + im·i`.
pub(crate) struct GaussianRational {
    pub re: Q,
    pub im: Q,
}

/// `e` as a Gaussian rational, when `e` is field arithmetic on rationals
/// and `i` only.
pub(crate) fn gaussian_rational_parts(e: &Ex) -> Option<GaussianRational> {
    let inner = e.inner.read();
    let arena = &inner.arena;
    let root = e.raw_id();
    if let Some(q) = arena.as_num(root) {
        return Some(GaussianRational {
            re: q.clone(),
            im: Q::zero(),
        });
    }
    let tower = Tower::scan(arena, &[root])?;
    if !tower.syms.is_empty() || !tower.primes.is_empty() || !tower.has_i || tower.ext.is_some() {
        return None;
    }
    let (n, d) = tower.eval_frac(arena, root, None, RF_TERM_BUDGET)??;
    let f = tower.normalize(n, d)?;
    let den = f.den.as_constant()?;
    let part = |k: u32| f.num.coeff(&[k]).cloned().unwrap_or_else(Q::zero) / &den;
    Some(GaussianRational {
        re: part(0),
        im: part(1),
    })
}

/// The normal form of a constant `e` of `ℚ(√p₁, …, i)` (square roots of
/// rationals, `i`): `Σ cₘ·m` over products `m` of the square roots and
/// `i`, with rational `cₘ` (the denominator rationalised).  `None` for
/// anything else.  `8/(1 + i) + 4i − 5` becomes `−1`: evaluated as it
/// stands, such an imaginary part cancels only numerically, and `√` of it
/// sits on the branch cut with the sign of rounding noise (`evalf`
/// refuses).
pub(crate) fn algebraic_constant_normal_form(e: &Ex) -> Option<Ex> {
    let (tower, frac) = {
        let inner = e.inner.read();
        let arena = &inner.arena;
        let root = e.raw_id();
        let tower = Tower::scan(arena, &[root])?;
        if !tower.syms.is_empty() || !tower.has_generators() || tower.l > 2 || tower.ext.is_some() {
            return None;
        }
        let (n, d) = tower.eval_frac(arena, root, None, RF_TERM_BUDGET)??;
        let f = tower.normalize(n, d)?;
        (tower, f)
    };
    Some(tower.frac_to_ex(&frac, e))
}

/// Exact zero test of `e` as a rational function of its free symbols over
/// `ℚ(radicals of rationals, golden ratio, i)` (see [`Tower`]).  `None`
/// when `e` has another node (a function, `π`, a radical of a symbol, …),
/// no algebraic constant (the plain rational-function test applies), or a
/// division by zero.
pub(crate) fn algebraic_function_is_zero(e: &Ex) -> Option<bool> {
    let inner = e.inner.read();
    let arena = &inner.arena;
    let root = e.raw_id();
    let tower = Tower::scan(arena, &[root])?;
    if !tower.has_generators() {
        return None;
    }
    tower.is_zero(arena, root)
}

/// Exact path of [`rref_solve`] for an augmented matrix whose entries are
/// rational functions of the parameters over a field of square roots and
/// `i` (a [`Tower`] with `L ≤ 2` and at least one generator): Gauss–Jordan
/// elimination on normal forms in `K(params)` ([`Tower::normalize`]), so
/// every zero decision is exact and no entry swells beyond its reduced
/// form.  The reduced row-echelon form over the field `K(params)` is
/// unique, so the answer does not depend on the pivot order; a pivot that
/// vanishes only at special parameter values is nonzero here (the generic
/// solution, as SymPy's `linsolve`).  Returns `None` for any other entry
/// (the expression path then runs) or when an entry exceeds the budget.
fn rref_solve_algebraic(
    mat: &[Vec<Ex>],
    n_vars: usize,
    unknowns: &[Ex],
) -> Option<SymbolicLinearResult> {
    let handle = mat.first()?.first()?.clone();
    let (tower, mut a) = {
        let inner = handle.inner.read();
        let arena = &inner.arena;
        let ids: Vec<Vec<ExprId>> = mat
            .iter()
            .map(|r| r.iter().map(Ex::raw_id).collect())
            .collect();
        let all: Vec<ExprId> = ids.iter().flatten().copied().collect();
        let tower = Tower::scan(arena, &all)?;
        if !tower.has_generators() || tower.l > 2 || tower.ext.is_some() {
            return None;
        }
        // Each row over a common denominator in ℚ[params]: a matrix over
        // the ring K[params].
        let mut a: Vec<Vec<Mp>> = Vec::with_capacity(ids.len());
        for row in &ids {
            let mut fr = Vec::with_capacity(row.len());
            for &id in row {
                let (n, d) = tower.eval_frac(arena, id, None, KFRAC_TERM_BUDGET)??;
                fr.push(tower.normalize(n, d)?);
            }
            let mut lcm = Mp::from_int(tower.nvars(), 1);
            for f in &fr {
                if f.den != lcm && f.den.as_constant().is_none() {
                    lcm = Mp::lcm(&lcm, &f.den);
                }
            }
            let mut out = Vec::with_capacity(fr.len());
            for f in fr {
                let cof = if f.den == lcm {
                    Mp::from_int(tower.nvars(), 1)
                } else {
                    lcm.div_exact(&f.den)?
                };
                out.push(f.num.mul(&cof));
            }
            a.push(out);
        }
        (tower, a)
    };
    let m = a.len();
    let ncols = n_vars + 1;
    let nv = tower.nvars();
    // Fraction-free Gauss–Jordan (FFGJ, Nakos, Turner & Williams,
    // *Fraction-free algorithms for linear and polynomial equations*, ACM
    // SIGSAM Bull. 31 (1997); as SymPy's `ddm_irref_den`): every entry is
    // a minor of the input, the division by the previous pivot is exact,
    // and at the end every pivot entry is the common denominator.
    let mut pivots: Vec<usize> = Vec::new();
    let mut pr = 0usize;
    let mut prev = Mp::from_int(nv, 1);
    for col in 0..n_vars {
        if pr >= m {
            break;
        }
        // Prefer a pivot in K, then the smallest.
        let Some(r) = (pr..m)
            .filter(|&r| !a[r][col].is_zero())
            .min_by_key(|&r| (!tower.in_k(&a[r][col]), a[r][col].num_terms()))
        else {
            continue;
        };
        a.swap(pr, r);
        let p = a[pr][col].clone();
        let pivot_row = a[pr].clone();
        let prev_inv = if tower.in_k(&prev) {
            Some(tower.inv_k(&prev)?)
        } else {
            None
        };
        for (i, row) in a.iter_mut().enumerate() {
            if i == pr {
                continue;
            }
            let f = row[col].clone();
            for j in 0..ncols {
                if j == col {
                    continue;
                }
                let t = tower.mul_r(&p, &row[j]);
                let t = if f.is_zero() {
                    t
                } else {
                    t.sub(&tower.mul_r(&f, &pivot_row[j]))
                };
                row[j] = match &prev_inv {
                    Some(inv) => tower.mul_r(&t, inv),
                    None => tower.div_exact_k(&t, &prev)?,
                };
                if row[j].num_terms() > KFRAC_TERM_BUDGET {
                    return None;
                }
            }
            row[col] = Mp::zero(nv);
        }
        prev = p;
        pivots.push(col);
        pr += 1;
    }
    // Below the rank the coefficient part is zero: a nonzero right-hand
    // side is `0 = c`.
    if a[pr..].iter().any(|row| !row[n_vars].is_zero()) {
        return Some(SymbolicLinearResult {
            values: Vec::new(),
            free: Vec::new(),
            inconsistent: true,
        });
    }
    let mut is_pivot = vec![false; n_vars];
    for &c in &pivots {
        is_pivot[c] = true;
    }
    let free: Vec<usize> = (0..n_vars).filter(|&c| !is_pivot[c]).collect();
    let mut values: Vec<Ex> = unknowns.to_vec();
    for (r, &pc) in pivots.iter().enumerate() {
        // Row r reads  D·x_pc + Σ_f a[r][f]·x_f = a[r][n], D = a[r][pc].
        let den = a[r][pc].clone();
        let quotient = |num: &Mp| -> Option<KFrac> { tower.quotient(num, &den) };
        let mut v = tower.frac_to_ex(&quotient(&a[r][n_vars])?, &handle);
        for &f in &free {
            if !a[r][f].is_zero() {
                let c = quotient(&a[r][f])?;
                v = &v - &(&tower.frac_to_ex(&c, &handle) * &unknowns[f]);
            }
        }
        // Each coefficient is in normal form already; `tidy` (a full
        // `simplify`) of the sum took seconds for entries of a few hundred
        // terms.
        values[pc] = v;
    }
    Some(SymbolicLinearResult {
        values,
        free,
        inconsistent: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::context::Context;

    fn show(e: &Ex) -> String {
        format!("{e}")
    }

    #[test]
    fn extract_linear_row_reads_coefficients_and_constant() {
        let ctx = Context::new();
        let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
        // 2x - 3y + 5 = 0   →   coeffs [2, -3], rhs -5
        let eq = &x * 2 - &y * 3 + 5;
        let (coeffs, rhs) = ctx.with_arena_mut(|a| {
            extract_linear_row(a, eq.raw_id(), &[x.raw_id(), y.raw_id()])
                .map(|(c, r)| {
                    (
                        c.iter()
                            .map(|&i| a.display(i).to_string())
                            .collect::<Vec<_>>(),
                        a.display(r).to_string(),
                    )
                })
                .expect("linear")
        });
        assert_eq!(coeffs, ["2", "-3"]);
        assert_eq!(rhs, "-5");
    }

    #[test]
    fn extract_linear_row_rejects_products_powers_and_functions() {
        let ctx = Context::new();
        let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
        for bad in [&x * &y, x.powi(2), x.sin(), &x / &y] {
            let none = ctx.with_arena_mut(|a| {
                extract_linear_row(a, bad.raw_id(), &[x.raw_id(), y.raw_id()]).is_none()
            });
            assert!(none, "{bad} should not be linear");
        }
    }

    #[test]
    fn unique_solution() {
        let ctx = Context::new();
        let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
        let r = linsolve_symbolic(&[&x + &y - 3, &x - &y - 1], &[x, y]).unwrap();
        assert!(!r.inconsistent);
        assert!(r.free.is_empty());
        assert_eq!(r.values.iter().map(show).collect::<Vec<_>>(), ["2", "1"]);
    }

    #[test]
    fn inconsistent_system_is_flagged() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let r = linsolve_symbolic(&[&x - 1, &x - 2], &[x]).unwrap();
        assert!(r.inconsistent);
    }

    #[test]
    fn underdetermined_system_reports_free_variable() {
        let ctx = Context::new();
        let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
        let r = linsolve_symbolic(&[&x + &y - 5], &[x.clone(), y.clone()]).unwrap();
        assert!(!r.inconsistent);
        assert_eq!(r.free.len(), 1);
        let f = r.free[0];
        // The free unknown maps to itself; the other is expressed through it.
        assert_eq!(show(&r.values[f]), show([&x, &y][f]));
        let pinned = 1 - f;
        let residual = (&x + &y - 5)
            .subs([&x, &y][pinned], &r.values[pinned])
            .simplify();
        assert_eq!(show(&residual), "0");
    }

    #[test]
    fn symbolic_coefficients_and_rational_answer() {
        let ctx = Context::new();
        let (x, a) = (ctx.symbol("x"), ctx.symbol("a"));
        // 3x = 1  →  1/3 ;  a·x = 1 → 1/a (a assumed nonzero as pivot)
        let r = linsolve_symbolic(&[&x * 3 - 1], std::slice::from_ref(&x)).unwrap();
        assert_eq!(show(&r.values[0]), "1/3");
        let r = linsolve_symbolic(&[&a * &x - 1], std::slice::from_ref(&x)).unwrap();
        let at2 = r.values[0].subs(&a, &ctx.int(2)).eval();
        assert_eq!(show(&at2), "1/2");
    }

    #[test]
    fn empty_input_is_invalid_argument() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        assert!(matches!(
            linsolve_symbolic(&[], std::slice::from_ref(&x)),
            Err(SymplexError::InvalidArgument { .. })
        ));
        assert!(matches!(
            linsolve_symbolic(&[&x - 1], &[]),
            Err(SymplexError::InvalidArgument { .. })
        ));
    }
}

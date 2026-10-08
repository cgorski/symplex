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
        Self::scan_opts(arena, roots, false)
    }

    /// [`scan`](Self::scan); with `conjugates` a node `conjugate(s)` of a
    /// symbol `s` is a parameter of its own (the Hermitian products of
    /// [`hermitian_gram_schmidt`]: `s` and `s̄` are algebraically
    /// independent, so a rational function of both is zero iff it is zero
    /// as a function of two independent variables).
    fn scan_opts(arena: &Arena, roots: &[ExprId], conjugates: bool) -> Option<Tower> {
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
                ExprNode::Conjugate(b)
                    if conjugates && matches!(arena.node(*b), ExprNode::Symbol(_)) =>
                {
                    syms.push(id);
                }
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
                    // (`conjugate(s)` is in `syms` only for a tower built by
                    // `scan_opts(…, true)`; elsewhere the lookup fails.)
                    ExprNode::Symbol(_) | ExprNode::Conjugate(_) => {
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

    /// The complex conjugate of a reduced polynomial of a square-root tower
    /// without a nested root: the parameters permuted by `perm` (`s ↔ s̄`
    /// for a parameter `s` that is not real, a real one fixed), the square
    /// roots of the positive primes (and `φ`) fixed, `i ↦ −i`.
    fn conj_poly(&self, p: &Mp, perm: &[usize]) -> Option<Mp> {
        let ns = self.syms.len();
        let iv = self.has_i.then(|| ns + self.primes.len());
        let terms: Vec<(Vec<u32>, Q)> = p
            .terms()
            .map(|(e, c)| {
                let mut f = e.to_vec();
                for (k, &to) in perm.iter().enumerate() {
                    f[to] = e[k];
                }
                let flip = iv.is_some_and(|v| e[v] % 2 == 1);
                (f, if flip { -c.clone() } else { c.clone() })
            })
            .collect();
        Mp::from_terms(p.num_vars(), terms)
    }

    /// `⟨x, y⟩ = Σ x̄ₖ·yₖ` of polynomial vectors, `xbar` the conjugates of `x`.
    fn hdot(&self, xbar: &[Mp], y: &[Mp]) -> Option<Mp> {
        let mut acc = Mp::zero(self.nvars());
        for (a, b) in xbar.iter().zip(y) {
            acc = acc.add(&self.mul_r(a, b));
            if acc.num_terms() > KFRAC_TERM_BUDGET {
                return None;
            }
        }
        Some(acc)
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

// ── Polynomials over the constants of a square-root tower ───────────────────

/// A univariate polynomial over the field `K` of a [`Tower`] without
/// parameters: reduced coefficients, ascending, no trailing zeros.
type KPoly = Vec<Mp>;

fn kp_trim(mut p: KPoly) -> KPoly {
    while p.last().is_some_and(Mp::is_zero) {
        p.pop();
    }
    p
}

fn kp_sub(a: &[Mp], b: &[Mp], nv: usize) -> KPoly {
    let zero = Mp::zero(nv);
    let terms = (0..a.len().max(b.len()))
        .map(|k| a.get(k).unwrap_or(&zero).sub(b.get(k).unwrap_or(&zero)))
        .collect();
    kp_trim(terms)
}

fn kp_deriv(p: &[Mp]) -> KPoly {
    let terms = p
        .iter()
        .enumerate()
        .skip(1)
        .map(|(k, c)| c.scale(&Q::from_integer(BigInt::from(k))))
        .collect();
    kp_trim(terms)
}

impl Tower {
    fn kp_monic(&self, p: &[Mp]) -> Option<KPoly> {
        let inv = self.inv_k(p.last()?)?;
        Some(p.iter().map(|c| self.mul_r(c, &inv)).collect())
    }

    fn kp_mul(&self, a: &[Mp], b: &[Mp]) -> KPoly {
        if a.is_empty() || b.is_empty() {
            return Vec::new();
        }
        let nv = self.nvars();
        let mut out = vec![Mp::zero(nv); a.len() + b.len() - 1];
        for (i, x) in a.iter().enumerate() {
            for (j, y) in b.iter().enumerate() {
                out[i + j] = out[i + j].add(&self.mul_r(x, y));
            }
        }
        kp_trim(out)
    }

    /// Quotient and remainder (`[quotient, remainder]`) of `a` by `b ≠ 0`.
    fn kp_divrem(&self, a: &[Mp], b: &[Mp]) -> Option<[KPoly; 2]> {
        let db = b.len().checked_sub(1)?;
        let inv = self.inv_k(b.last()?)?;
        let mut r: KPoly = a.to_vec();
        let mut q = vec![Mp::zero(self.nvars()); a.len().saturating_sub(db).max(1)];
        while r.len() > db && !r.is_empty() {
            let k = r.len() - 1 - db;
            let f = self.mul_r(r.last()?, &inv);
            for (j, c) in b.iter().enumerate() {
                r[k + j] = r[k + j].sub(&self.mul_r(&f, c));
            }
            q[k] = f;
            r.pop();
            r = kp_trim(r);
        }
        Some([kp_trim(q), r])
    }

    /// The monic gcd (`[]` for two zero polynomials).
    fn kp_gcd(&self, a: &[Mp], b: &[Mp]) -> Option<KPoly> {
        let (mut x, mut y) = (kp_trim(a.to_vec()), kp_trim(b.to_vec()));
        while !y.is_empty() {
            let [_, r] = self.kp_divrem(&x, &y)?;
            x = y;
            y = r;
        }
        if x.is_empty() {
            return Some(x);
        }
        self.kp_monic(&x)
    }

    /// `b/a` for `a | b`.
    fn kp_quo(&self, b: &[Mp], a: &[Mp]) -> Option<KPoly> {
        let [q, r] = self.kp_divrem(b, a)?;
        r.is_empty().then_some(q)
    }

    /// Yun's square-free decomposition of a monic `f`: `(factor, k)` with
    /// `f = ∏ factorᵏ`, the factors monic, square-free and coprime.
    fn kp_squarefree(&self, f: &[Mp]) -> Option<Vec<(KPoly, usize)>> {
        let nv = self.nvars();
        let df = kp_deriv(f);
        let a0 = self.kp_gcd(f, &df)?;
        let mut b = self.kp_quo(f, &a0)?;
        let c = self.kp_quo(&df, &a0)?;
        let mut d = kp_sub(&c, &kp_deriv(&b), nv);
        let mut out = Vec::new();
        let mut k = 1usize;
        while b.len() > 1 {
            let a = self.kp_gcd(&b, &d)?;
            let nb = self.kp_quo(&b, &a)?;
            let nc = self.kp_quo(&d, &a)?;
            if a.len() > 1 {
                out.push((a, k));
            }
            d = kp_sub(&nc, &kp_deriv(&nb), nv);
            b = nb;
            k += 1;
            if k > f.len() {
                return None;
            }
        }
        Some(out)
    }

    /// The factors of a monic square-free `f` over `K` obtained from the
    /// irreducible factors `h` of its norm `N(f) = ∏_σ σ(f) ∈ ℚ[λ]` (`σ`
    /// over the sign changes of the generators): `f = ∏ gcd(f, h)`, since
    /// every root of `f` is a root of exactly one `h`.  A factor may still
    /// be reducible over `K` when `N(f)` is not square-free.  `[f]` when the
    /// norm is too large (degree above 64) or not rational.
    fn kp_split_by_norm(&self, f: &[Mp]) -> Option<Vec<KPoly>> {
        let ns = self.syms.len();
        let gens: Vec<usize> = (ns..self.nvars()).collect();
        let deg = f.len() - 1;
        if deg <= 1 || gens.len() > 6 || deg << gens.len() > 64 {
            return Some(vec![f.to_vec()]);
        }
        let flip = |p: &Mp, mask: usize| -> Option<Mp> {
            let terms: Vec<(Vec<u32>, Q)> = p
                .terms()
                .map(|(e, c)| {
                    let odd = gens
                        .iter()
                        .enumerate()
                        .filter(|(b, v)| mask >> b & 1 == 1 && e[**v] % 2 == 1)
                        .count();
                    (
                        e.to_vec(),
                        if odd % 2 == 1 { -c.clone() } else { c.clone() },
                    )
                })
                .collect();
            Mp::from_terms(p.num_vars(), terms)
        };
        let mut norm: KPoly = vec![Mp::from_int(self.nvars(), 1)];
        for mask in 0..(1usize << gens.len()) {
            let g: KPoly = f.iter().map(|c| flip(c, mask)).collect::<Option<_>>()?;
            norm = self.kp_mul(&norm, &g);
        }
        let mut qc: Vec<Q> = Vec::with_capacity(norm.len());
        for c in &norm {
            match c.as_constant() {
                Some(v) => qc.push(v),
                None if c.is_zero() => qc.push(Q::zero()),
                None => return Some(vec![f.to_vec()]),
            }
        }
        let (_, factors) = crate::poly::dense::Poly::from_coeffs(qc).factor_over_z();
        let mut out = Vec::new();
        let nv = self.nvars();
        for (h, _) in &factors {
            if h.degree().unwrap_or(0) == 0 {
                continue;
            }
            let hk: KPoly = h
                .coeffs()
                .iter()
                .map(|c| Mp::constant(nv, c.clone()))
                .collect();
            let g = self.kp_gcd(f, &hk)?;
            if g.len() > 1 {
                out.push(g);
            }
        }
        let total: usize = out.iter().map(|g| g.len() - 1).sum();
        if total != deg {
            return Some(vec![f.to_vec()]);
        }
        Some(out)
    }

    fn kp_rem(&self, a: &[Mp], g: &[Mp]) -> Option<KPoly> {
        let [_, r] = self.kp_divrem(a, g)?;
        Some(r)
    }

    /// The inverse of `a` modulo `g` (extended Euclid over `K`); `None` when
    /// `gcd(a, g) ≠ 1` (a zero divisor of `K[t]/(g)`).
    fn kp_inv_mod(&self, a: &[Mp], g: &[Mp]) -> Option<KPoly> {
        let nv = self.nvars();
        let (mut r0, mut r1) = (g.to_vec(), self.kp_rem(a, g)?);
        let (mut s0, mut s1): (KPoly, KPoly) = (Vec::new(), vec![Mp::from_int(nv, 1)]);
        while r1.len() > 1 {
            let [q, r2] = self.kp_divrem(&r0, &r1)?;
            let s2 = kp_sub(&s0, &self.kp_mul(&q, &s1), nv);
            (r0, r1) = (r1, r2);
            (s0, s1) = (s1, s2);
        }
        let c = self.inv_k(r1.first()?)?;
        let s: KPoly = s1.iter().map(|x| self.mul_r(x, &c)).collect();
        self.kp_rem(&s, g)
    }

    /// Is the constant `c` of `K` real and negative / positive?  `Some(sign)`
    /// of a real `c`, `None` for a non-real one or an undecided sign.
    fn real_constant_sign(&self, c: &Mp) -> Option<i8> {
        let ns = self.syms.len();
        if self.has_i {
            let iv = ns + self.primes.len();
            if c.terms().any(|(e, _)| e[iv] % 2 == 1) {
                return None;
            }
        }
        self.real_sign(c)
    }
}

/// The roots, with multiplicities, of the polynomial `Σ coeffs[k]·λᵏ`
/// whose coefficients are constants of `K = ℚ(√p₁, …, i)` (not all
/// rational), in radicals: SymPy's `roots` for a polynomial over an
/// algebraic field — the multiplicities from the square-free
/// decomposition over `K`, the factors split by the norm to `ℚ[λ]` (roots
/// in `K` exactly), then the quadratic formula `(−b ± √(b² − 4c))/2` and
/// Cardano's formulas in the form of SymPy's `roots_cubic`.  Every
/// coefficient computation is exact in `K`; the radicands are in its normal
/// form; a quartic factor by SymPy's `roots_quartic` ([`kp_quartic_roots`]).
/// `None` when a coefficient is not such a constant, a factor has degree
/// above 4, or a sign is undecided.
pub(crate) fn algebraic_poly_roots(coeffs: &[Ex]) -> Option<Vec<(Ex, usize)>> {
    let handle = coeffs.first()?.clone();
    let (tower, f) = algebraic_k_poly(coeffs)?;
    let mut out: Vec<(Ex, usize)> = Vec::new();
    for (sf, mult) in tower.kp_squarefree(&f)? {
        for g in tower.kp_split_by_norm(&sf)? {
            for root in kp_radical_roots(&tower, &g, &handle)? {
                out.push((root, mult));
            }
        }
    }
    let total: usize = out.iter().map(|(_, m)| *m).sum();
    (total == f.len() - 1).then_some(out)
}

/// The polynomial `Σ coeffs[k]·λᵏ` over `K = ℚ(√p₁, …, i)`, made monic,
/// with its tower: `None` when a coefficient is not a constant of such a
/// `K` (or all are rational), or the polynomial is constant.
fn algebraic_k_poly(coeffs: &[Ex]) -> Option<(Tower, KPoly)> {
    let handle = coeffs.first()?;
    let (tower, f) = {
        let inner = handle.inner.read();
        let arena = &inner.arena;
        let ids: Vec<ExprId> = coeffs.iter().map(Ex::raw_id).collect();
        let tower = Tower::scan(arena, &ids)?;
        if !tower.syms.is_empty() || !tower.has_generators() || tower.l > 2 || tower.ext.is_some() {
            return None;
        }
        let mut f: KPoly = Vec::with_capacity(ids.len());
        for &id in &ids {
            let (n, d) = tower.eval_frac(arena, id, None, RF_TERM_BUDGET)??;
            f.push(tower.mul_r(&n, &tower.inv_k(&d)?));
        }
        (tower, kp_trim(f))
    };
    if f.len() < 2 {
        return None;
    }
    let f = tower.kp_monic(&f)?;
    Some((tower, f))
}

/// The number of distinct roots of the polynomial `Σ coeffs[k]·λᵏ` whose
/// coefficients are constants of `K = ℚ(√p₁, …, i)`, not all rational:
/// the degree of its square-free part over `K` (exact, from the same
/// square-free decomposition as [`algebraic_poly_roots`]).  `None` when a
/// coefficient is not such a constant.
pub(crate) fn algebraic_poly_distinct_root_count(coeffs: &[Ex]) -> Option<usize> {
    let (tower, f) = algebraic_k_poly(coeffs)?;
    let factors = tower.kp_squarefree(&f)?;
    Some(factors.iter().map(|(g, _)| g.len() - 1).sum())
}

/// Can the exact pivot test ([`algebraic_function_is_zero`]) decide the
/// entries built from `es`: are they rational functions over a square-root
/// tower (possibly with one nested square root)?
pub(crate) fn in_square_root_tower(es: &[Ex]) -> bool {
    let Some(first) = es.first() else {
        return true;
    };
    let inner = first.inner.read();
    let ids: Vec<ExprId> = es.iter().map(Ex::raw_id).collect();
    Tower::scan(&inner.arena, &ids).is_some_and(|t| t.l <= 2)
}

/// A basis of the eigenspace of the eigenvalue `lam` of the matrix `rows`
/// with constant entries in `K = ℚ(√p…, i)` and characteristic polynomial
/// `charpoly` (ascending), exactly: `lam` is a root of one factor `g` of
/// `charpoly` over `K` (square-free parts split by their norms, as in
/// [`algebraic_poly_roots`]), found by a certified evaluation — every
/// other factor has nonzero digits at `lam`, so `lam` is a root of the one
/// that evaluates to zero — and `A − tI` is eliminated over `K[t]/(g)`.
/// Each vector entry is the residue `c₀ + c₁·t + …` at `t = lam` (the form
/// of the rational-matrix eigenvectors over `ℚ[t]/(g)`).  Before 0.32 the
/// elimination ran on the expressions of `A − λI`, whose pivots in a
/// Cardano `λ` no zero test decides (no eigenvector, or no answer).  `None`
/// when an entry is not such a constant, the factor is not identified, or a
/// pivot is a zero divisor of `K[t]/(g)` (`g` reducible over `K`).
pub(crate) fn algebraic_eigenvectors(
    rows: &[Vec<Ex>],
    charpoly: &[Ex],
    lam: &Ex,
) -> Option<Vec<Vec<Ex>>> {
    let n = rows.len();
    let (tower, a, f) = {
        let inner = lam.inner.read();
        let arena = &inner.arena;
        let mut ids: Vec<ExprId> = rows.iter().flatten().map(Ex::raw_id).collect();
        ids.extend(charpoly.iter().map(Ex::raw_id));
        let tower = Tower::scan(arena, &ids)?;
        if !tower.syms.is_empty() || !tower.has_generators() || tower.l > 2 || tower.ext.is_some() {
            return None;
        }
        let kval = |id: ExprId| -> Option<Mp> {
            let (nm, d) = tower.eval_frac(arena, id, None, RF_TERM_BUDGET)??;
            Some(tower.mul_r(&nm, &tower.inv_k(&d)?))
        };
        let a: Vec<Vec<Mp>> = rows
            .iter()
            .map(|r| r.iter().map(|e| kval(e.raw_id())).collect::<Option<_>>())
            .collect::<Option<_>>()?;
        let f: KPoly = charpoly
            .iter()
            .map(|e| kval(e.raw_id()))
            .collect::<Option<_>>()?;
        (tower, a, kp_trim(f))
    };
    let nv = tower.nvars();
    let kx = |c: &Mp| -> Ex {
        let fr = KFrac {
            num: c.clone(),
            den: Mp::from_int(nv, 1),
        };
        tower.frac_to_ex(&fr, lam)
    };
    let lam_pows: Vec<Ex> = (0..f.len()).map(|k| lam.powi(k as i64)).collect();
    let at_lam = |p: &[Mp]| -> Ex {
        let terms = p
            .iter()
            .enumerate()
            .filter(|(_, c)| !c.is_zero())
            .map(|(k, c)| if k == 0 { kx(c) } else { &kx(c) * &lam_pows[k] });
        Ex::sum_of(&lam.context(), terms)
    };
    let f = tower.kp_monic(&f)?;
    let mut gs: Vec<KPoly> = Vec::new();
    for (sf, _) in tower.kp_squarefree(&f)? {
        gs.extend(tower.kp_split_by_norm(&sf)?);
    }
    let mut which: Option<usize> = None;
    for (idx, g) in gs.iter().enumerate() {
        match at_lam(g).eval_decimal(30) {
            Ok(s) if s == "0" => {
                if which.replace(idx).is_some() {
                    return None;
                }
            }
            Ok(_) => {}
            Err(_) => return None,
        }
    }
    let g = gs.swap_remove(which?);
    // A − tI over K[t]/(g), Gauss–Jordan.
    let t: KPoly = vec![Mp::zero(nv), Mp::from_int(nv, 1)];
    let mut m: Vec<Vec<KPoly>> = Vec::with_capacity(n);
    for (i, row) in a.iter().enumerate() {
        let mut r = Vec::with_capacity(n);
        for (j, e) in row.iter().enumerate() {
            let c = kp_trim(vec![e.clone()]);
            let x = if i == j { kp_sub(&c, &t, nv) } else { c };
            r.push(tower.kp_rem(&x, &g)?);
        }
        m.push(r);
    }
    let mut pivots: Vec<usize> = Vec::new();
    let mut pr = 0usize;
    for col in 0..n {
        if pr >= n {
            break;
        }
        let Some(f) = (pr..n).find(|&i| !m[i][col].is_empty()) else {
            continue;
        };
        m.swap(pr, f);
        let inv = tower.kp_inv_mod(&m[pr][col], &g)?;
        let pivot_row: Vec<KPoly> = m[pr]
            .iter()
            .map(|x| tower.kp_rem(&tower.kp_mul(x, &inv), &g))
            .collect::<Option<_>>()?;
        for (i, row) in m.iter_mut().enumerate() {
            if i == pr || row[col].is_empty() {
                continue;
            }
            let fac = row[col].clone();
            for (x, p) in row.iter_mut().zip(&pivot_row) {
                let y = kp_sub(x, &tower.kp_mul(&fac, p), nv);
                *x = tower.kp_rem(&y, &g)?;
            }
        }
        m[pr] = pivot_row;
        pivots.push(col);
        pr += 1;
    }
    let ctx = lam.context();
    let mut out = Vec::new();
    for free in (0..n).filter(|c| !pivots.contains(c)) {
        let mut v = vec![ctx.zero(); n];
        v[free] = ctx.one();
        for (r, &pc) in pivots.iter().enumerate() {
            if !m[r][free].is_empty() {
                v[pc] = -at_lam(&m[r][free]);
            }
        }
        out.push(v);
    }
    (!out.is_empty()).then_some(out)
}

/// The roots of a monic square-free `g` over `K` of degree 1 to 4.
fn kp_radical_roots(tower: &Tower, g: &[Mp], handle: &Ex) -> Option<Vec<Ex>> {
    let ctx = handle.context();
    let nv = tower.nvars();
    let kx = |c: &Mp| -> Ex {
        let f = KFrac {
            num: c.clone(),
            den: Mp::from_int(nv, 1),
        };
        tower.frac_to_ex(&f, handle)
    };
    let qn = |n: i64, d: i64| Q::new(BigInt::from(n), BigInt::from(d));
    let sc = |c: &Mp, n: i64, d: i64| c.scale(&qn(n, d));
    let mul = |a: &Mp, b: &Mp| tower.mul_r(a, b);
    let tidy = |e: Ex| e.eval();
    match g.len() - 1 {
        1 => Some(vec![tidy(-kx(&g[0]))]),
        2 => {
            // (−b ± √(b² − 4c))/2
            let (c, b) = (&g[0], &g[1]);
            let disc = mul(b, b).sub(&sc(c, 4, 1));
            let s = kx(&disc).sqrt();
            let mb = -kx(b);
            let two = ctx.int(2);
            Some(vec![tidy(&(&mb + &s) / &two), tidy(&(&mb - &s) / &two)])
        }
        3 => {
            // SymPy `roots_cubic` for x³ + a·x² + b·x + c.
            let (c, b, a) = (&g[0], &g[1], &g[2]);
            let aa = mul(a, a);
            let p = b.sub(&sc(&aa, 1, 3));
            let q = c.sub(&sc(&mul(a, b), 1, 3)).add(&sc(&mul(&aa, a), 2, 27));
            let aon3 = sc(a, 1, 3);
            let shift = kx(&aon3);
            let coeff = &(&ctx.int(3).sqrt() * &ctx.i_unit()) / &ctx.int(2);
            let half = ctx.rational(-1, 2);
            let omega = [ctx.int(1), &half + &coeff, &half - &coeff];
            if c.is_zero() {
                // x·(x² + a·x + b)
                let quad = [b.clone(), a.clone(), Mp::from_int(nv, 1)];
                let mut r = kp_radical_roots(tower, &quad, handle)?;
                r.insert(1, ctx.zero());
                return Some(r);
            }
            if q.is_zero() {
                // y³ + p·y: y = 0, ±√(−p)
                let s = kx(&p.neg()).sqrt();
                return Some(vec![tidy(&s - &shift), tidy(-&shift), tidy(-&s - &shift)]);
            }
            let u1 = if p.is_zero() {
                if tower.real_constant_sign(&q) == Some(1) {
                    Some(-kx(&q).cbrt())
                } else {
                    Some(kx(&q.neg()).cbrt())
                }
            } else if tower.real_constant_sign(&q) == Some(-1) {
                // −∛(−q/2 + √(q²/4 + p³/27))
                let r = sc(&mul(&q, &q), 1, 4).add(&sc(&mul(&mul(&p, &p), &p), 1, 27));
                let inner = &kx(&sc(&q, -1, 2)) + &kx(&r).sqrt();
                Some(-inner.cbrt())
            } else {
                None
            };
            let Some(u1) = u1 else {
                // −(a + uₖ·C + D₀/(C·uₖ))/3, C = ∛((D₁ + √(D₁² − 4D₀³))/2)
                let d0 = aa.sub(&sc(b, 3, 1));
                let d1 = sc(&mul(&aa, a), 2, 1)
                    .sub(&sc(&mul(a, b), 9, 1))
                    .add(&sc(c, 27, 1));
                let rad = mul(&d1, &d1).sub(&sc(&mul(&mul(&d0, &d0), &d0), 4, 1));
                let cc = (&(&kx(&d1) + &kx(&rad).sqrt()) / &ctx.int(2)).cbrt();
                let (ae, d0e) = (kx(a), kx(&d0));
                let three = ctx.int(3);
                return Some(
                    omega
                        .iter()
                        .map(|uk| {
                            let ukc = uk * &cc;
                            tidy(-(&(&(&ae + &ukc) + &(&d0e / &ukc)) / &three))
                        })
                        .collect(),
                );
            };
            let pon3 = kx(&sc(&p, 1, 3));
            Some(
                omega
                    .iter()
                    .map(|uk| {
                        let u = uk * &u1;
                        if p.is_zero() {
                            tidy(&u - &shift)
                        } else {
                            tidy(&(&-&u + &(&pon3 / &u)) - &shift)
                        }
                    })
                    .collect(),
            )
        }
        4 => kp_quartic_roots(tower, g, handle),
        _ => None,
    }
}

/// The roots of a monic square-free quartic `x⁴ + a·x³ + b·x² + c·x + d`
/// over `K` in radicals, as SymPy's `roots_quartic` (`polys/polyroots.py`)
/// takes its cases, every coefficient computation and every case test exact
/// in `K`:
///
/// * `d = 0`: `0` and the roots of the cubic `x³ + a·x² + b·x + c`;
/// * `(c/a)² = d` (quasi-symmetric): `z = x + m/x`, `m = c/a`, solves
///   `z² + a·z + b − 2m = 0`, then `x² − z·x + m = 0`;
/// * otherwise the depressed `y⁴ + e·y² + f·y + g` (`x = y − a/4`): for
///   `f = 0` the biquadratic `y = ±√((−e ± √(e² − 4g))/2)`; for `g = 0`,
///   `0` and the roots of `y³ + e·y + f`; else Descartes–Euler when the
///   resolvent `64R³ + 32e·R² + (4e² − 16g)·R − f²` has a nonzero rational
///   root (the largest, SymPy's `_roots_quartic_euler`), and Ferrari
///   otherwise: `y = −5e/6 − ∛q` for `p = 0`, else `−5e/6 + u − p/(3u)` with
///   `u = ∛(−q/2 + √(q²/4 + p³/27))`, `p = −e²/12 − g`, `q = −e³/108 + e·g/3
///   − f²/8` (decided exactly here, where SymPy may return a `Piecewise`),
///   and the roots `(s·w − t·√(−(3e + 2y + 2s·f/w)))/2 − a/4`, `w = √(e +
///   2y)`, `s, t = ±1`.
///
/// Each formula is an identity for every choice of the square and cube
/// roots (the resolvent root `y` is a root whatever cube root `u` is, `w`
/// enters `2f/w` with the same sign, and both signs of every other square
/// root are taken), so a square root is taken as [`any_sqrt`] does: on the
/// negative real axis as `i·√(−X)`.
fn kp_quartic_roots(tower: &Tower, g: &[Mp], handle: &Ex) -> Option<Vec<Ex>> {
    let ctx = handle.context();
    let nv = tower.nvars();
    let kx = |c: &Mp| -> Ex {
        let f = KFrac {
            num: c.clone(),
            den: Mp::from_int(nv, 1),
        };
        tower.frac_to_ex(&f, handle)
    };
    let qn = |n: i64, d: i64| Q::new(BigInt::from(n), BigInt::from(d));
    let sc = |c: &Mp, n: i64, d: i64| c.scale(&qn(n, d));
    let mul = |a: &Mp, b: &Mp| tower.mul_r(a, b);
    let tidy = |e: Ex| e.eval();
    let (d, c, b, a) = (&g[0], &g[1], &g[2], &g[3]);
    let one = Mp::from_int(nv, 1);
    let two = ctx.int(2);
    if d.is_zero() {
        let cubic = [c.clone(), b.clone(), a.clone(), one];
        let mut r = kp_radical_roots(tower, &cubic, handle)?;
        r.insert(0, ctx.zero());
        return Some(r);
    }
    if !a.is_zero() {
        let m = mul(c, &tower.inv_k(a)?);
        if mul(&m, &m).sub(d).is_zero() {
            // z² + a·z + (b − 2m) = 0, then x² − z·x + m = 0.
            let disc = mul(a, a).sub(&sc(&b.sub(&sc(&m, 2, 1)), 4, 1));
            let s = kx(&disc).sqrt();
            let ma = -kx(a);
            let four_m = kx(&sc(&m, 4, 1));
            let mut out = Vec::with_capacity(4);
            for z in [&(&ma + &s) / &two, &(&ma - &s) / &two] {
                let r = any_sqrt(&(&(&z * &z) - &four_m));
                out.push(tidy(&(&z + &r) / &two));
                out.push(tidy(&(&z - &r) / &two));
            }
            return Some(out);
        }
    }
    let a2 = mul(a, a);
    let e = b.sub(&sc(&a2, 3, 8));
    let f = c.add(&mul(a, &sc(&a2, 1, 8).sub(&sc(b, 1, 2))));
    let aon4 = sc(a, 1, 4);
    let g0 = d.sub(&mul(
        &aon4,
        &mul(a, &sc(&a2, 3, 64).sub(&sc(b, 1, 4))).add(c),
    ));
    let shift = kx(&aon4);
    if f.is_zero() {
        let disc = mul(&e, &e).sub(&sc(&g0, 4, 1));
        let s = kx(&disc).sqrt();
        let me = -kx(&e);
        let y1 = any_sqrt(&(&(&me + &s) / &two));
        let y2 = any_sqrt(&(&(&me - &s) / &two));
        let ys = [-&y1, -&y2, y1, y2];
        return Some(ys.into_iter().map(|y| tidy(&y - &shift)).collect());
    }
    if g0.is_zero() {
        let cubic = [f, e, Mp::zero(nv), one];
        let mut r = kp_radical_roots(tower, &cubic, handle)?;
        r.insert(0, ctx.zero());
        return Some(r.into_iter().map(|y| tidy(&y - &shift)).collect());
    }
    // Descartes–Euler: R a nonzero rational root of the resolvent, then
    // `±√R ∓ √(A ± B)` with `A = −R − e/2`, `B = −f·√R/(4R)`.
    let resolvent = [
        sc(&mul(&f, &f), -1, 64),
        sc(&mul(&e, &e), 1, 16).sub(&sc(&g0, 1, 4)),
        sc(&e, 1, 2),
        Mp::from_int(nv, 1),
    ];
    // Its rational roots are the common roots of its components over ℚ
    // ([`kp_rational_roots`]); the norm of the resolvent to ℚ[λ] (degree
    // 48 over four generators, factored by Zassenhaus) found the same ones
    // in up to seconds.  Beyond the size at which the norm was not formed
    // only roots of linear square-free factors were seen, and still are.
    let gens = tower.nvars();
    let rational_root = if gens <= 6 && 3usize << gens <= 64 {
        kp_rational_roots(&resolvent)
            .into_iter()
            .filter(|r| !r.is_zero())
            .max()
    } else {
        tower
            .kp_squarefree(&resolvent)?
            .into_iter()
            .filter_map(|(h, _)| tower.kp_split_by_norm(&h))
            .flatten()
            .filter(|h| h.len() == 2)
            .filter_map(|h| h[0].neg().as_constant())
            .filter(|r| !r.is_zero())
            .max()
    };
    if let Some(r) = rational_root {
        let rr = Mp::constant(nv, r);
        let c1 = kx(&rr).sqrt();
        let bb = &(&-kx(&f) * &c1) / &kx(&sc(&rr, 4, 1));
        let aa = kx(&rr.neg().sub(&sc(&e, 1, 2)));
        let c2 = any_sqrt(&(&aa + &bb));
        let c3 = any_sqrt(&(&aa - &bb));
        return Some(vec![
            tidy(&(&c1 - &c2) - &shift),
            tidy(&(&-&c1 - &c3) - &shift),
            tidy(&(&-&c1 + &c3) - &shift),
            tidy(&(&c1 + &c2) - &shift),
        ]);
    }
    // Ferrari.
    let ee = mul(&e, &e);
    let p = sc(&ee, -1, 12).sub(&g0);
    let q = sc(&mul(&ee, &e), -1, 108)
        .add(&sc(&mul(&e, &g0), 1, 3))
        .sub(&sc(&mul(&f, &f), 1, 8));
    let five_e_6 = kx(&sc(&e, -5, 6));
    let y = if p.is_zero() {
        &five_e_6 - &kx(&q).cbrt()
    } else {
        let rad = sc(&mul(&q, &q), 1, 4).add(&sc(&mul(&mul(&p, &p), &p), 1, 27));
        let u = (&kx(&sc(&q, -1, 2)) + &kx(&rad).sqrt()).cbrt();
        &(&five_e_6 + &u) - &(&kx(&p) / &(&ctx.int(3) * &u))
    };
    let w = any_sqrt(&(&kx(&e) + &(&two * &y)));
    let arg1 = &kx(&sc(&e, 3, 1)) + &(&two * &y);
    let arg2 = &(&two * &kx(&f)) / &w;
    let mut out = Vec::with_capacity(4);
    for s in [-1i64, 1] {
        let sw = &ctx.int(s) * &w;
        let root = any_sqrt(&-(&arg1 + &(&ctx.int(s) * &arg2)));
        for t in [-1i64, 1] {
            let x = &(&(&sw - &(&ctx.int(t) * &root)) / &two) - &shift;
            out.push(tidy(x));
        }
    }
    Some(out)
}

/// The rational roots of `p ∈ K[λ]` (a square-root tower without
/// parameters or nested root): with `p = Σₘ pₘ·m` over the reduced monomials
/// `m` of the generators, a basis of `K` over `ℚ`, `p(r) = 0` for a rational
/// `r` exactly when `pₘ(r) = 0` for every `m` — the rational roots of
/// `gcdₘ pₘ ∈ ℚ[λ]`, from its linear factors over `ℤ`.
fn kp_rational_roots(p: &[Mp]) -> Vec<Q> {
    let mut comps: std::collections::BTreeMap<Vec<u32>, Vec<Q>> = std::collections::BTreeMap::new();
    for (k, c) in p.iter().enumerate() {
        for (e, q) in c.terms() {
            let v = comps
                .entry(e.to_vec())
                .or_insert_with(|| vec![Q::zero(); p.len()]);
            v[k] = q.clone();
        }
    }
    let mut g: Option<crate::poly::dense::Poly> = None;
    for cs in comps.into_values() {
        let pm = crate::poly::dense::Poly::from_coeffs(cs);
        g = Some(match g {
            None => pm,
            Some(g0) => crate::poly::dense::Poly::gcd(&g0, &pm),
        });
    }
    let Some(g) = g.filter(|g| g.degree().unwrap_or(0) > 0) else {
        return Vec::new();
    };
    let (_, factors) = g.factor_over_z();
    factors
        .iter()
        .filter_map(|(h, _)| match h.coeffs() {
            [c0, c1] if !c1.is_zero() => Some(-(c0 / c1)),
            _ => None,
        })
        .collect()
}

/// A square root of `x` (either sign will do for the caller): `√x`, or
/// `i·√(−x)` when `x` lies on the negative real axis numerically.  There
/// a radicand built from complex cube roots (Cardano's in the casus
/// irreducibilis: the resolvent root `y` of a real quartic is real, its
/// imaginary part a rounding residue) is real only to within its error, the
/// side of the cut of `√` undecidable, and `evalf` refuses the root — as it
/// refused one root in a hundred of the quartics over `ℚ(√2, √3)` before
/// this choice; `−x` lies on the positive axis, away from the cut.
fn any_sqrt(x: &Ex) -> Ex {
    let on_cut = x
        .eval_complex64()
        .is_ok_and(|z| z.re < 0.0 && z.im.abs() <= 1e-6 * z.re.abs());
    if on_cut {
        &x.context().i_unit() * &(-x).sqrt()
    } else {
        x.sqrt()
    }
}

/// The factors of a Gram–Schmidt orthogonalisation: the columns `q` and
/// the `k×k` upper triangular coefficient matrix `r` (row-major).
pub(crate) struct GramSchmidtFactors {
    pub q: Vec<Vec<Ex>>,
    pub r: Vec<Vec<Ex>>,
}

/// Outcome of [`hermitian_gram_schmidt`].
pub(crate) enum HermitianGs {
    /// The orthogonal (orthonormal when normalising) columns and `R`.
    Factors(GramSchmidtFactors),
    /// Column `j` lies in the span of the columns before it.
    Dependent(usize),
}

/// Gram–Schmidt with the Hermitian inner product `⟨x, y⟩ = Σ x̄ₖ·yₖ` (SymPy's
/// `QRdecomposition`, `u.dot(v, hermitian=True)`), exactly in the field
/// `K(params, conj(params))` of a square-root [`Tower`]: each parameter `s`
/// that is not provably real gets a companion parameter `conjugate(s)`, and
/// conjugation permutes the two, fixes the real radicals and maps `i` to
/// `−i`.  The columns are orthogonalised fraction-free (the integral
/// Gram–Schmidt of de Weger, *Solving exponential Diophantine equations
/// using lattice basis reduction algorithms*, J. Number Theory 26 (1987),
/// §3, in the Hermitian form): with `aⱼ = a′ⱼ/cⱼ` over a common denominator
/// `cⱼ ∈ ℚ[params]`, `Δ₀ = 1` and `Δₗ` the Gram determinant of `a′₀ … a′ₗ₋₁`,
/// `wⱼ = Δⱼ·u′ⱼ` is a polynomial vector reached by
/// `w ← (Δₗ₊₁·w − ⟨wₗ, a′ⱼ⟩·wₗ)/Δₗ` (`l < j`, exact divisions in `K[params]`),
/// and `Δⱼ₊₁ = ⟨wⱼ, wⱼ⟩/Δⱼ`.  Only the outputs are brought to normal form
/// (one gcd each; before 0.32 every projection was `simplify`d, 53 s in a
/// debug build for a 4×4 matrix in one parameter): `uⱼ = wⱼ/(Δⱼ·cⱼ)`,
/// `dⱼ = ‖uⱼ‖² = Δⱼ₊₁/(Δⱼ·cⱼ·c̄ⱼ)`, `⟨uᵢ, aⱼ⟩ = ⟨wᵢ, a′ⱼ⟩/(Δᵢ·c̄ᵢ·cⱼ)`,
/// and `qⱼ = uⱼ/√dⱼ`, `rᵢⱼ = ⟨uᵢ, aⱼ⟩/√dᵢ` (`i < j`), `rⱼⱼ = √dⱼ` (real,
/// positive).  Without `normalize`: `qⱼ = uⱼ`, `rᵢⱼ = ⟨uᵢ, aⱼ⟩/dᵢ`,
/// `rⱼⱼ = 1`.
///
/// `dⱼ = 0` exactly as a rational function (column `j` is dependent,
/// generically) is reported as such.  `None` when an entry is not a
/// rational function over square roots of rationals and `i` (a function, a
/// nested or higher root, a symbol whose conjugate is not `conjugate(s)`),
/// on a division by zero, or when a polynomial exceeds the budget.
pub(crate) fn hermitian_gram_schmidt(cols: &[Vec<Ex>], normalize: bool) -> Option<HermitianGs> {
    let handle = cols.first()?.first()?.clone();
    let ctx = handle.context();
    let mut syms: Vec<Ex> = Vec::new();
    for e in cols.iter().flatten() {
        for s in e.free_symbols() {
            if !syms.contains(&s) {
                syms.push(s);
            }
        }
    }
    // (symbol, conjugate) for the symbols that are not provably real.
    let mut complex: Vec<[Ex; 2]> = Vec::new();
    let mut real: Vec<ExprId> = Vec::new();
    for s in syms {
        if s.is_real() == Some(true) {
            real.push(s.raw_id());
        } else {
            let c = s.conjugate();
            complex.push([s, c]);
        }
    }
    let (tower, a, perm) = {
        let inner = handle.inner.read();
        let arena = &inner.arena;
        for [s, c] in &complex {
            if !matches!(arena.node(c.raw_id()), ExprNode::Conjugate(b) if *b == s.raw_id()) {
                return None;
            }
        }
        let mut ids: Vec<ExprId> = cols.iter().flatten().map(Ex::raw_id).collect();
        ids.extend(complex.iter().map(|[_, c]| c.raw_id()));
        let tower = Tower::scan_opts(arena, &ids, true)?;
        if tower.l > 2 || tower.ext.is_some() {
            return None;
        }
        let ns = tower.syms.len();
        let pos = |id: ExprId| tower.syms.iter().position(|&x| x == id);
        let mut perm: Vec<usize> = (0..ns).collect();
        for [s, c] in &complex {
            let (i, j) = (pos(s.raw_id())?, pos(c.raw_id())?);
            perm[i] = j;
            perm[j] = i;
        }
        // Every parameter fixed by the conjugation must be a real symbol.
        if (0..ns).any(|k| perm[k] == k && !real.contains(&tower.syms[k])) {
            return None;
        }
        let mut a: Vec<Vec<KFrac>> = Vec::with_capacity(cols.len());
        for col in cols {
            let mut fc = Vec::with_capacity(col.len());
            for e in col {
                let (n, d) = tower.eval_frac(arena, e.raw_id(), None, KFRAC_TERM_BUDGET)??;
                fc.push(tower.normalize(n, d)?);
            }
            a.push(fc);
        }
        (tower, a, perm)
    };
    let k = a.len();
    let m = a[0].len();
    let nv = tower.nvars();
    let one_p = Mp::from_int(nv, 1);
    // aⱼ = a′ⱼ/cⱼ with cⱼ = |lⱼ|² real and positive, lⱼ the lcm of the
    // (normalised: 1 or non-constant) denominators of the column.
    let mut ap: Vec<Vec<Mp>> = Vec::with_capacity(k);
    let mut c: Vec<Mp> = Vec::with_capacity(k);
    for col in &a {
        let mut l = one_p.clone();
        for f in col {
            if f.den != l && f.den.as_constant().is_none() {
                l = Mp::lcm(&l, &f.den);
            }
        }
        let lbar = tower.conj_poly(&l, &perm)?;
        let mut out = Vec::with_capacity(m);
        for f in col {
            let cof = if f.den == l {
                lbar.clone()
            } else {
                tower.mul_r(&l.div_exact(&f.den)?, &lbar)
            };
            out.push(tower.mul_r(&f.num, &cof));
        }
        ap.push(out);
        c.push(tower.mul_r(&l, &lbar));
    }
    // Exact division by a nonzero Δ of K[params].
    let div = |x: &Mp, b: &Mp| -> Option<Mp> {
        if tower.in_k(b) {
            Some(tower.mul_r(x, &tower.inv_k(b)?))
        } else {
            tower.div_exact_k(x, b)
        }
    };
    let mut w: Vec<Vec<Mp>> = Vec::with_capacity(k);
    let mut wbar: Vec<Vec<Mp>> = Vec::with_capacity(k);
    let mut delta: Vec<Mp> = vec![one_p.clone()];
    // lam[i][j] = ⟨wᵢ, a′ⱼ⟩
    let mut lam: Vec<Vec<Mp>> = vec![vec![Mp::zero(nv); k]; k];
    for j in 0..k {
        let mut v = ap[j].clone();
        for l in 0..j {
            let lj = tower.hdot(&wbar[l], &ap[j])?;
            for t in 0..m {
                let x = tower
                    .mul_r(&delta[l + 1], &v[t])
                    .sub(&tower.mul_r(&lj, &w[l][t]));
                v[t] = div(&x, &delta[l])?;
            }
            lam[l][j] = lj;
        }
        let vbar: Vec<Mp> = v
            .iter()
            .map(|x| tower.conj_poly(x, &perm))
            .collect::<Option<_>>()?;
        let next = div(&tower.hdot(&vbar, &v)?, &delta[j])?;
        if next.is_zero() {
            return Some(HermitianGs::Dependent(j));
        }
        w.push(v);
        wbar.push(vbar);
        delta.push(next);
    }
    let tidy = |e: Ex| if e.is_constant() { e.eval() } else { e };
    let pe = |p: &Mp| {
        let whole = KFrac {
            num: p.clone(),
            den: one_p.clone(),
        };
        tower.frac_to_ex(&whole, &handle)
    };
    let one = ctx.one();
    let mut r: Vec<Vec<Ex>> = vec![vec![ctx.zero(); k]; k];
    let mut q: Vec<Vec<Ex>> = Vec::with_capacity(k);
    if normalize {
        // uⱼ = wⱼ/(Δⱼ·cⱼ) and dⱼ = Δⱼ₊₁/(Δⱼ·cⱼ²) with Δ, c > 0, so
        // qⱼ = wⱼ/(√Δⱼ·√Δⱼ₊₁), rⱼⱼ = √Δⱼ₊₁/(√Δⱼ·cⱼ) and
        // rᵢⱼ = ⟨uᵢ, aⱼ⟩/√dᵢ = ⟨wᵢ, a′ⱼ⟩/(cⱼ·√Δᵢ·√Δᵢ₊₁): polynomials
        // and square roots of polynomials, no gcd.  (`√(1/n)` of a positive
        // rational folds to `c·√m` under `tidy`.)
        let sq: Vec<Ex> = delta
            .iter()
            .map(|x| {
                if x.as_constant().is_some_and(|v| v.is_one()) {
                    one.clone()
                } else {
                    pe(x).sqrt()
                }
            })
            .collect();
        for j in 0..k {
            let s = &sq[j] * &sq[j + 1];
            q.push(w[j].iter().map(|x| tidy(&pe(x) / &s)).collect());
            let cj = pe(&c[j]);
            for i in 0..j {
                let den = &(&cj * &sq[i]) * &sq[i + 1];
                r[i][j] = tidy(&pe(&lam[i][j]) / &den);
            }
            r[j][j] = tidy(&sq[j + 1] / &(&sq[j] * &cj));
        }
    } else {
        // uⱼ = wⱼ/(Δⱼ·cⱼ), rᵢⱼ = ⟨uᵢ, aⱼ⟩/dᵢ = ⟨wᵢ, a′ⱼ⟩·cᵢ/(cⱼ·Δᵢ₊₁).
        for j in 0..k {
            let den = tower.mul_r(&delta[j], &c[j]);
            q.push(
                w[j].iter()
                    .map(|x| Some(tidy(tower.frac_to_ex(&tower.quotient(x, &den)?, &handle))))
                    .collect::<Option<_>>()?,
            );
            for i in 0..j {
                let num = tower.mul_r(&lam[i][j], &c[i]);
                let f = tower.quotient(&num, &tower.mul_r(&c[j], &delta[i + 1]))?;
                r[i][j] = tidy(tower.frac_to_ex(&f, &handle));
            }
            r[j][j] = one.clone();
        }
    }
    Some(HermitianGs::Factors(GramSchmidtFactors { q, r }))
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

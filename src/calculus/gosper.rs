//! Gosper's algorithm for closed-form hypergeometric summation.
//!
//! Given a hypergeometric term `f(k)` (one where `f(k+1)/f(k)` is a rational
//! function of `k`), this module decides whether the indefinite sum
//! `Σ f(k)` has a closed-form hypergeometric antidifference `g(k)` such that
//! `g(k+1) - g(k) = f(k)`.
//!
//! # Algorithm outline (Gosper 1978)
//!
//! 1. **Normal form.** Factor the ratio `r(k) = p(k)/q(k)` into
//!    `a(k)·c(k+1) / (b(k)·c(k))` with `gcd(a(k), b(k+h)) = 1` for all
//!    non-negative integers `h`.
//!
//! 2. **Certificate.** Solve the recurrence
//!    `a(k)·x(k+1) − b(k−1)·x(k) = c(k)` for a polynomial `x(k)`.
//!
//! 3. **Antidifference.** `g(k) = b(k−1)·x(k)·f(k) / c(k)`.
//!
//! # References
//!
//! - R. W. Gosper, "Decision procedure for indefinite hypergeometric
//!   summation", *PNAS* 75 (1978), 40–42.
//! - M. Petkovsek, H. Wilf, D. Zeilberger, *A = B*, A K Peters, 1996,
//!   Chapter 5.

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Signed, ToPrimitive, Zero};

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};
use crate::base::walk;
use crate::poly::Poly;
use crate::poly::polybridge;
use crate::simplify::combsimp;
use crate::transforms::eval;
use crate::transforms::subs;

/// Largest certificate polynomial degree Gosper's algorithm will attempt.
///
/// The certificate is found by solving a dense `(n+d+1) × (d+1)` rational
/// linear system; degrees beyond this cap are declined (the sum is left
/// unevaluated) so that the algorithm terminates in bounded time.
pub(crate) const MAX_CERTIFICATE_DEGREE: usize = 64;

/// Largest shift `h` examined when computing the dispersion set
/// `{h ≥ 0 : gcd(A(k), B(k+h)) ≠ 1}` in the Gosper normal form.
///
/// A shift `h` in the dispersion set contributes a factor of degree
/// `h·deg d` to `c`, so large shifts also make every later step expensive;
/// sums with such widely separated poles are handled by the partial-fraction
/// telescoping in `calculus::summation` instead.
pub(crate) const MAX_DISPERSION: i64 = 100;

// ═══════════════════════════════════════════════════════════════════════════
// Polynomial shift: p(k) → p(k + n)
// ═══════════════════════════════════════════════════════════════════════════

/// Shift a polynomial by an integer: compute `p(k + n)`.
///
/// Uses Horner's method in the polynomial ring: evaluates `p` at the
/// polynomial `k + n` rather than at a scalar.
#[must_use]
pub(crate) fn poly_shift(p: &Poly, n: i64) -> Poly {
    let deg = match p.degree() {
        Some(d) => d,
        None => return Poly::zero(),
    };

    let shift_poly = Poly::from_coeffs(vec![Ratio::from_integer(BigInt::from(n)), Ratio::one()]);

    let coeffs = p.coeffs();
    let mut result = Poly::constant(coeffs[deg].clone());
    for i in (0..deg).rev() {
        result = &(&result * &shift_poly) + &Poly::constant(coeffs[i].clone());
    }
    result
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper: root-modulus bound
// ═══════════════════════════════════════════════════════════════════════════

/// Cauchy bound `1 + maxᵢ |aᵢ / aₙ|` on the modulus of every (complex) root of
/// `p`, rounded up to an integer.  Returns `0` for constants.
fn root_modulus_bound(p: &Poly) -> Option<BigInt> {
    let deg = p.degree()?;
    if deg == 0 {
        return Some(BigInt::zero());
    }
    let lc = p.leading_coeff()?.abs();
    if lc.is_zero() {
        return None;
    }
    let mut max_ratio = Ratio::<BigInt>::zero();
    for i in 0..deg {
        let ratio = p.coeff(i).abs() / &lc;
        if ratio > max_ratio {
            max_ratio = ratio;
        }
    }
    Some((Ratio::<BigInt>::one() + max_ratio).ceil().to_integer())
}

// ═══════════════════════════════════════════════════════════════════════════
// Gosper normal form
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the Gosper normal form of a ratio `p/q`.
///
/// Given coprime polynomials `p(k)` and `q(k)`, finds `(a, b, c)` where
/// `a = Z·A_monic`, `b = B_monic`, `c` are polynomials satisfying
///
/// ```text
/// p(k)/q(k) = a(k)·c(k+1) / (b(k)·c(k))
/// ```
///
/// with `gcd(A_monic(k), B_monic(k+h)) = 1` for every non-negative integer `h`.
///
/// The dispersion set `{h : gcd(A(k), B(k+h)) ≠ 1}` is found by a direct
/// gcd scan over `0 ≤ h ≤ bound(A) + bound(B)`, where `bound(·)` is the
/// Cauchy root-modulus bound: a common root `r` of `A(k)` and `B(k+h)`
/// satisfies `|r| ≤ bound(A)` and `|r + h| ≤ bound(B)`, so no larger `h`
/// can occur.  The scan is capped at [`MAX_DISPERSION`]; beyond that the
/// normal form may be incomplete, which can only make the certificate step
/// *fail* (the identity `p/q = a·c(k+1)/(b·c(k))` always holds, and a found
/// certificate is verified exactly), never produce a wrong antidifference.
///
/// Returns `None` if the inputs are degenerate (e.g. zero polynomials).
#[must_use]
pub(crate) fn gosper_normal(p: &Poly, q: &Poly) -> Option<(Poly, Poly, Poly)> {
    tracing::debug!("gosper_normal: entering");

    if p.is_zero() || q.is_zero() {
        return None;
    }

    // Ensure p and q are coprime.
    let g = Poly::gcd(p, q);
    let p = if g.is_constant() && g.coeff(0).is_one() {
        p.clone()
    } else {
        p.div(&g)
    };
    let q = if g.is_constant() && g.coeff(0).is_one() {
        q.clone()
    } else {
        q.div(&g)
    };

    let lc_p = p.leading_coeff()?.clone();
    let lc_q = q.leading_coeff()?.clone();
    let z = &lc_p / &lc_q;

    let mut a = p.make_monic(); // A_monic
    let mut b = q.make_monic(); // B_monic
    let mut c = Poly::from_int(1);

    let deg_a = a.degree().unwrap_or(0);
    let deg_b = b.degree().unwrap_or(0);

    if deg_a > 0 && deg_b > 0 {
        let bound = root_modulus_bound(&a)? + root_modulus_bound(&b)?;
        let bound = bound.to_i64().unwrap_or(MAX_DISPERSION).min(MAX_DISPERSION);
        for i in 0..=bound {
            if a.is_constant() || b.is_constant() {
                break;
            }
            let b_shifted_i = poly_shift(&b, i);
            let d = Poly::gcd(&a, &b_shifted_i);
            if d.is_constant() {
                continue; // trivial gcd, nothing to factor out
            }

            a = a.div(&d);
            let d_back = poly_shift(&d, -i);
            b = b.div(&d_back);

            for j in 1..=i {
                let d_shifted = poly_shift(&d, -j);
                c = &c * &d_shifted;
            }
        }
    }

    // Return (Z·A, B, C).
    let za = a.scale(&z);
    tracing::debug!(
        "gosper: normal form deg(A)={}, deg(B)={}, deg(C)={}",
        za.degree().unwrap_or(0),
        b.degree().unwrap_or(0),
        c.degree().unwrap_or(0)
    );
    Some((za, b, c))
}

// ═══════════════════════════════════════════════════════════════════════════
// Linear system solver for undetermined coefficients
// ═══════════════════════════════════════════════════════════════════════════

/// Solve a linear system over ℚ via Gauss–Jordan elimination.
///
/// `mat` is a `nrows × (ncols + 1)` augmented matrix `[A | b]`.
/// Returns a solution vector of length `ncols`, or `None` if inconsistent.
/// Free variables (under-determined systems) are set to zero.
#[allow(clippy::needless_range_loop)]
fn solve_rational_system(
    mat: &mut [Vec<Ratio<BigInt>>],
    nrows: usize,
    ncols: usize,
) -> Option<Vec<Ratio<BigInt>>> {
    let mut pivot_row = 0;
    let mut pivot_cols = Vec::new();

    for col in 0..ncols {
        // Find a non-zero entry in this column at or below pivot_row.
        let mut found = None;
        for row in pivot_row..nrows {
            if !mat[row][col].is_zero() {
                found = Some(row);
                break;
            }
        }
        let Some(pr) = found else { continue };

        mat.swap(pivot_row, pr);

        // Scale so the pivot is 1.
        let scale = mat[pivot_row][col].clone();
        let scale_inv = Ratio::one() / &scale;
        for j in col..=ncols {
            let v = &mat[pivot_row][j] * &scale_inv;
            mat[pivot_row][j] = v;
        }

        // Eliminate this column in all other rows.
        for row in 0..nrows {
            if row == pivot_row {
                continue;
            }
            let factor = mat[row][col].clone();
            if factor.is_zero() {
                continue;
            }
            for j in col..=ncols {
                let sub = &factor * &mat[pivot_row][j];
                mat[row][j] -= sub;
            }
        }

        pivot_cols.push((pivot_row, col));
        pivot_row += 1;
    }

    // Check for inconsistency: a row with all-zero LHS but non-zero RHS.
    for row in pivot_row..nrows {
        if !mat[row][ncols].is_zero() {
            return None;
        }
    }

    // Extract solution (free variables default to zero).
    let mut solution = vec![Ratio::zero(); ncols];
    for &(pr, col) in &pivot_cols {
        solution[col] = mat[pr][ncols].clone();
    }

    Some(solution)
}

// ═══════════════════════════════════════════════════════════════════════════
// Gosper certificate
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the Gosper certificate polynomial `x(k)`.
///
/// Solves `a(k)·x(k+1) − b(k−1)·x(k) = c(k)` for a polynomial `x`,
/// where `(a, b, c)` come from [`gosper_normal`].
///
/// Returns `None` when no polynomial solution exists (the sum is not
/// Gosper-summable).
#[must_use]
#[allow(clippy::needless_range_loop)]
pub(crate) fn gosper_certificate(a: &Poly, b: &Poly, c: &Poly) -> Option<Poly> {
    tracing::debug!("gosper_certificate: entering");

    if c.is_zero() {
        return Some(Poly::zero());
    }

    let b_m1 = poly_shift(b, -1); // b(k − 1)

    let deg_a = a.degree().unwrap_or(0);
    let deg_bm = b_m1.degree().unwrap_or(0);
    let deg_c = c.degree().unwrap_or(0);
    let n = deg_a.max(deg_bm);

    // Compute the degree bound for x(k).
    let d = compute_degree_bound(a, &b_m1, c, deg_a, deg_bm, deg_c, n)?;
    tracing::debug!("gosper: certificate degree bound d={}", d);

    let total_deg = n + d;
    let nrows = total_deg + 1;
    let ncols = d + 1;

    let mut mat = vec![vec![Ratio::<BigInt>::zero(); ncols + 1]; nrows];

    // For each basis monomial u_j · k^j:
    //   column j = coefficients of a(k)·(k+1)^j − b_{-1}(k)·k^j
    for j in 0..ncols {
        let kp1_j = poly_shift(&Poly::monomial(Ratio::one(), j), 1); // (k+1)^j
        let a_term = a * &kp1_j;

        let kj = Poly::monomial(Ratio::one(), j);
        let b_term = &b_m1 * &kj;

        let col_poly = &a_term - &b_term;

        for i in 0..nrows {
            mat[i][j] = col_poly.coeff(i);
        }
    }

    // Right-hand side: coefficients of c(k).
    for i in 0..nrows {
        mat[i][ncols] = c.coeff(i);
    }

    let coeffs = solve_rational_system(&mut mat, nrows, ncols)?;

    // Verify the solution is non-trivially zero only if c is zero.
    let result = Poly::from_coeffs(coeffs);

    // Double-check: a(k)*x(k+1) - b(k-1)*x(k) should equal c(k).
    let x_shifted = poly_shift(&result, 1);
    let lhs = &(a * &x_shifted) - &(&b_m1 * &result);
    if &lhs != c {
        return None;
    }

    Some(result)
}

/// Determine an upper bound on the degree of the certificate polynomial.
///
/// Returns `None` when no polynomial solution can exist.
fn compute_degree_bound(
    a: &Poly,
    b_m1: &Poly,
    c: &Poly,
    deg_a: usize,
    deg_bm: usize,
    deg_c: usize,
    n: usize,
) -> Option<usize> {
    if c.is_zero() {
        return Some(0);
    }

    // Check whether the leading coefficients of a and b(k-1) coincide.
    let lc_a = if a.is_zero() {
        Ratio::zero()
    } else {
        a.coeff(deg_a)
    };
    let lc_b = if b_m1.is_zero() {
        Ratio::zero()
    } else {
        b_m1.coeff(deg_bm)
    };

    let bound = if deg_a == deg_bm && lc_a == lc_b {
        // Leading terms cancel, so `a(k)·x(k+1) − b(k−1)·x(k)` has degree at
        // most `n + d − 1`.  Its `k^{n+d−1}` coefficient is
        // `(lc·d + A − B)·x_d` where `A`, `B` are the next-to-leading
        // coefficients; when `d₀ = (B − A)/lc` is a non-negative integer that
        // coefficient vanishes too and `x` may need degree `d₀`.
        let mut d = deg_c + 1;
        if n >= 1 {
            let next_a = a.coeff(n - 1);
            let next_b = b_m1.coeff(n - 1);
            let d0 = (&next_b - &next_a) / &lc_a;
            if d0.is_integer() && !d0.is_negative() {
                let d0 = d0.to_integer().to_usize()?;
                d = d.max(d0);
            }
        }
        d
    } else if deg_c >= n {
        deg_c - n
    } else {
        // The operator strictly increases degree, but c has lower degree.
        // Try degree 0 anyway — the system will be inconsistent if no solution.
        0
    };
    if bound > MAX_CERTIFICATE_DEGREE {
        tracing::debug!(
            "gosper: certificate degree bound {} exceeds cap {}",
            bound,
            MAX_CERTIFICATE_DEGREE
        );
        return None;
    }
    Some(bound)
}

// ═══════════════════════════════════════════════════════════════════════════
// Hypergeometric ratio detection
// ═══════════════════════════════════════════════════════════════════════════

/// Attempt to compute `f(k+1)/f(k)` as a pair `(numerator, denominator)`
/// of expressions that are polynomial in `k`.
///
/// Returns `None` if `f` is not recognised as hypergeometric.
pub(crate) fn hypergeometric_ratio(
    arena: &mut Arena,
    f: ExprId,
    k: ExprId,
) -> Option<(ExprId, ExprId)> {
    // If f doesn't depend on k it is a constant term; ratio = 1.
    if !walk::free_symbols(arena, f).contains(&k) {
        return Some((arena.one, arena.one));
    }

    // f = k itself: ratio = (k+1)/k.
    if f == k {
        let k1 = arena.add(&[k, arena.one]);
        return Some((k1, k));
    }

    let node = arena.node(f).clone();
    match node {
        // factorial(k) → ratio = k + 1
        ExprNode::Factorial(arg) if arg == k => {
            let k1 = arena.add(&[k, arena.one]);
            Some((k1, arena.one))
        }

        // factorial(expr(k)) where expr is linear in k
        ExprNode::Factorial(arg) => {
            if let Some(arg_poly) = polybridge::expr_to_poly(arena, arg, k)
                && arg_poly.degree() == Some(1)
                && arg_poly.coeff(1).is_one()
            {
                // factorial(k + c): ratio = k + c + 1
                let shifted = poly_shift(&arg_poly, 1);
                let numer_expr = polybridge::poly_to_expr(arena, &shifted, k);
                return Some((numer_expr, arena.one));
            }
            // Fall through to substitution approach
            try_ratio_by_substitution(arena, f, k)
        }

        // binomial(n, k) → ratio = (n − k)/(k + 1)
        ExprNode::Binomial(n, kk) if kk == k && !walk::free_symbols(arena, n).contains(&k) => {
            let n_minus_k = arena.sub(n, k);
            let k1 = arena.add(&[k, arena.one]);
            Some((n_minus_k, k1))
        }

        // pow(base, k) where base is k-free → ratio = base
        ExprNode::Pow(base, exp) if exp == k => {
            if !walk::free_symbols(arena, base).contains(&k) {
                Some((base, arena.one))
            } else {
                None
            }
        }

        // pow(base, exp) where base is k-free and exp is polynomial in k
        ExprNode::Pow(base, exp)
            if !walk::free_symbols(arena, base).contains(&k)
                && walk::free_symbols(arena, exp).contains(&k) =>
        {
            if let Some(exp_poly) = polybridge::expr_to_poly(arena, exp, k) {
                let shifted = poly_shift(&exp_poly, 1);
                let diff = &shifted - &exp_poly;
                if diff.is_constant() {
                    let diff_expr = polybridge::poly_to_expr(arena, &diff, k);
                    let ratio = arena.pow(base, diff_expr);
                    let ratio = eval::eval(arena, ratio);
                    return Some((ratio, arena.one));
                }
            }
            None
        }

        // pow(k, n) where n is k-free integer
        ExprNode::Pow(base, exp) if base == k && !walk::free_symbols(arena, exp).contains(&k) => {
            // Check if n is a negative rational
            let k1 = arena.add(&[k, arena.one]);
            if let Some(r) = arena.as_num(exp) {
                let r = r.clone();
                if r.is_negative() {
                    let pos_exp = arena.neg(exp);
                    let numer = arena.pow(k, pos_exp);
                    let denom = arena.pow(k1, pos_exp);
                    return Some((numer, denom));
                }
            }
            let numer = arena.pow(k1, exp);
            let denom = arena.pow(k, exp);
            Some((numer, denom))
        }

        // pow(expr(k), n) where n is k-free integer, expr is polynomial in k
        ExprNode::Pow(base, exp)
            if !walk::free_symbols(arena, exp).contains(&k)
                && walk::free_symbols(arena, base).contains(&k) =>
        {
            if let Some(r) = arena.as_num(exp) {
                let r = r.clone();
                if r.is_integer() {
                    let n_val: i64 = r.to_integer().to_i64()?;
                    if n_val != 0 && n_val.abs() <= 20 {
                        // (base(k))^n: ratio = (base(k+1)/base(k))^n; for
                        // negative n the two parts swap.
                        if let Some((bp, bq)) = hypergeometric_ratio(arena, base, k) {
                            let abs_exp = arena.int(n_val.abs());
                            let (top, bottom) = if n_val > 0 { (bp, bq) } else { (bq, bp) };
                            let numer = arena.pow(top, abs_exp);
                            let denom = arena.pow(bottom, abs_exp);
                            return Some((numer, denom));
                        }
                    }
                }
            }
            None
        }

        // Neg(inner): same ratio as inner (the −1 factors cancel)
        ExprNode::Neg(inner) => hypergeometric_ratio(arena, inner, k),

        // Mul(factors): product of individual ratios
        ExprNode::Mul(ref children) => {
            let children_vec: Vec<ExprId> = children.iter().copied().collect();
            let mut numer_parts = Vec::new();
            let mut denom_parts = Vec::new();
            for &child in &children_vec {
                let (p, q) = hypergeometric_ratio(arena, child, k)?;
                if p != arena.one {
                    numer_parts.push(p);
                }
                if q != arena.one {
                    denom_parts.push(q);
                }
            }
            let p = match numer_parts.len() {
                0 => arena.one,
                1 => numer_parts[0],
                _ => arena.mul(&numer_parts),
            };
            let q = match denom_parts.len() {
                0 => arena.one,
                1 => denom_parts[0],
                _ => arena.mul(&denom_parts),
            };
            Some((p, q))
        }

        // Add: try to convert to polynomial and compute shift/original
        ExprNode::Add(_) => {
            if let Some(f_poly) = polybridge::expr_to_poly(arena, f, k) {
                let shifted = poly_shift(&f_poly, 1);
                let numer_expr = polybridge::poly_to_expr(arena, &shifted, k);
                let denom_expr = polybridge::poly_to_expr(arena, &f_poly, k);
                Some((numer_expr, denom_expr))
            } else {
                try_ratio_by_substitution(arena, f, k)
            }
        }

        _ => try_ratio_by_substitution(arena, f, k),
    }
}

/// Fallback: compute the ratio by direct substitution k → k+1, division,
/// and combinatorial simplification.
fn try_ratio_by_substitution(arena: &mut Arena, f: ExprId, k: ExprId) -> Option<(ExprId, ExprId)> {
    let k_plus_1 = arena.add(&[k, arena.one]);
    let f_k1 = subs::subs(arena, f, k, k_plus_1);
    let ratio = arena.div(f_k1, f);
    let ratio = combsimp::combsimp(arena, ratio);
    let ratio = eval::eval(arena, ratio);

    let (numer, denom) = polybridge::as_numer_denom(arena, ratio);

    // Verify both parts are polynomial in k.
    // Validate that both numerator and denominator are polynomial in k.
    // The `?` returns None if either fails to convert, rejecting non-rational ratios.
    let _ = polybridge::expr_to_poly(arena, numer, k)?;
    let _ = polybridge::expr_to_poly(arena, denom, k)?;

    Some((numer, denom))
}

/// Decide whether `term` is a hypergeometric term in `k` and, if so, return
/// the ratio `term(k+1)/term(k)` as a rational function of `k`.
///
/// The ratio may contain symbolic parameters other than `k` (e.g. the `n`
/// in `C(n, k)`), but both its numerator and denominator must be polynomial
/// in `k`.  Returns `None` for terms such as `1/k!`-free `sin(k)` or `k^k`.
#[must_use]
pub(crate) fn is_hypergeometric(arena: &mut Arena, term: ExprId, k: ExprId) -> Option<ExprId> {
    if !matches!(arena.node(k), ExprNode::Symbol(_)) {
        return None;
    }
    let (numer, denom) = hypergeometric_ratio(arena, term, k)?;
    let numer = eval::eval(arena, numer);
    let denom = eval::eval(arena, denom);
    let numer_x = arena.expand_expr(numer);
    let denom_x = arena.expand_expr(denom);
    // Both parts must be polynomial in k (symbolic coefficients allowed).
    crate::calculus::summation::sym_poly_in(arena, numer_x, k)?;
    crate::calculus::summation::sym_poly_in(arena, denom_x, k)?;
    if arena.is_zero_structural(denom_x) {
        return None;
    }
    let ratio = arena.div(numer_x, denom_x);
    // Cancel common polynomial factors when everything is rational in k.
    let ratio = if polybridge::expr_to_poly(arena, numer_x, k).is_some()
        && polybridge::expr_to_poly(arena, denom_x, k).is_some()
    {
        polybridge::cancel(arena, ratio, k)
    } else {
        ratio
    };
    Some(eval::eval(arena, ratio))
}

// ═══════════════════════════════════════════════════════════════════════════
// Full Gosper summation
// ═══════════════════════════════════════════════════════════════════════════

/// Attempt to find a closed form for `Σ_{k=lower}^{upper} f(k)`.
///
/// Returns the evaluated definite sum `g(upper+1) − g(lower)` where `g` is
/// the hypergeometric antidifference, or `None` if the sum is not
/// Gosper-summable.
#[must_use]
pub(crate) fn gosper_sum(
    arena: &mut Arena,
    f_expr: ExprId,
    k_var: ExprId,
    lower: ExprId,
    upper: ExprId,
) -> Option<ExprId> {
    tracing::debug!("gosper_sum: entering");

    // Ensure k_var is a symbol.
    if !matches!(arena.node(k_var), ExprNode::Symbol(_)) {
        return None;
    }

    // Step 0: detect the hypergeometric ratio f(k+1)/f(k) = p/q.
    let (numer_expr, denom_expr) = hypergeometric_ratio(arena, f_expr, k_var)?;

    // Convert numerator and denominator to polynomials in k.
    let numer_expr_eval = eval::eval(arena, numer_expr);
    let denom_expr_eval = eval::eval(arena, denom_expr);
    let numer_expanded = arena.expand_expr(numer_expr_eval);
    let denom_expanded = arena.expand_expr(denom_expr_eval);

    let p = polybridge::expr_to_poly(arena, numer_expanded, k_var)?;
    let q = polybridge::expr_to_poly(arena, denom_expanded, k_var)?;

    if q.is_zero() {
        return None;
    }

    // Step 1: Gosper normal form.
    let (a, b, c) = gosper_normal(&p, &q)?;

    // Step 2: find the certificate.
    let x = gosper_certificate(&a, &b, &c)?;

    // Step 3: construct the antidifference.
    //   g(k) = b(k−1) · x(k) · f(k) / c(k)
    let b_m1 = poly_shift(&b, -1);
    let bm1_x = &b_m1 * &x;

    // Cancel common polynomial factors between bm1·x and c.
    let common = Poly::gcd(&bm1_x, &c);
    let numer_poly = if common.is_constant() && common.coeff(0).is_one() {
        bm1_x.clone()
    } else {
        bm1_x.div(&common)
    };
    let denom_poly = if common.is_constant() && common.coeff(0).is_one() {
        c.clone()
    } else {
        c.div(&common)
    };

    let numer_expr2 = polybridge::poly_to_expr(arena, &numer_poly, k_var);
    let denom_expr2 = polybridge::poly_to_expr(arena, &denom_poly, k_var);

    // g(k) = (numer_poly / denom_poly) * f(k)
    let g_k = if denom_poly.is_constant() && denom_poly.coeff(0).is_one() {
        arena.mul(&[numer_expr2, f_expr])
    } else {
        let frac = arena.div(numer_expr2, denom_expr2);
        arena.mul(&[frac, f_expr])
    };

    // Definite sum: g(upper + 1) − g(lower).
    let one = arena.one;
    let upper_plus_1 = arena.add(&[upper, one]);

    let g_upper = subs::subs(arena, g_k, k_var, upper_plus_1);
    let g_lower = subs::subs(arena, g_k, k_var, lower);

    let result = arena.sub(g_upper, g_lower);
    let result = eval::eval(arena, result);

    Some(result)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;
    use num_rational::Ratio;

    fn r(n: i64) -> Ratio<BigInt> {
        Ratio::from_integer(BigInt::from(n))
    }

    fn display(arena: &Arena, id: ExprId) -> String {
        arena.display(id).to_string()
    }

    // ── poly_shift ──────────────────────────────────────────────────

    #[test]
    fn shift_constant() {
        let p = Poly::from_int(5);
        let shifted = poly_shift(&p, 3);
        assert_eq!(shifted, Poly::from_int(5));
    }

    #[test]
    fn shift_zero() {
        let p = Poly::zero();
        let shifted = poly_shift(&p, 10);
        assert!(shifted.is_zero());
    }

    #[test]
    fn shift_x_by_1() {
        // x → x + 1 = 1 + x
        let p = Poly::x();
        let shifted = poly_shift(&p, 1);
        let expected = Poly::from_coeffs(vec![r(1), r(1)]);
        assert_eq!(shifted, expected);
    }

    #[test]
    fn shift_x_by_neg1() {
        // x → x - 1 = -1 + x
        let p = Poly::x();
        let shifted = poly_shift(&p, -1);
        let expected = Poly::from_coeffs(vec![r(-1), r(1)]);
        assert_eq!(shifted, expected);
    }

    #[test]
    fn shift_x_squared_by_1() {
        // x² → (x+1)² = 1 + 2x + x²
        let p = Poly::from_coeffs(vec![r(0), r(0), r(1)]);
        let shifted = poly_shift(&p, 1);
        let expected = Poly::from_coeffs(vec![r(1), r(2), r(1)]);
        assert_eq!(shifted, expected);
    }

    #[test]
    fn shift_quadratic_by_2() {
        // x² + x → (x+2)² + (x+2) = x² + 4x + 4 + x + 2 = x² + 5x + 6
        let p = Poly::from_coeffs(vec![r(0), r(1), r(1)]); // x + x²
        let shifted = poly_shift(&p, 2);
        let expected = Poly::from_coeffs(vec![r(6), r(5), r(1)]);
        assert_eq!(shifted, expected);
    }

    #[test]
    fn shift_polynomial_roundtrip() {
        // shift(shift(p, 3), -3) == p
        let p = Poly::from_coeffs(vec![r(1), r(-2), r(3)]);
        let shifted = poly_shift(&poly_shift(&p, 3), -3);
        assert_eq!(shifted, p);
    }

    #[test]
    fn shift_eval_consistency() {
        // p(5) == shift(p, 3).eval(2)
        let p = Poly::from_coeffs(vec![r(1), r(-2), r(3), r(1)]);
        let val_direct = p.eval(&r(5));
        let shifted = poly_shift(&p, 3);
        let val_shifted = shifted.eval(&r(2));
        assert_eq!(val_direct, val_shifted);
    }

    // ── root_modulus_bound ──────────────────────────────────────────

    #[test]
    fn root_bound_linear() {
        // h − 3: root 3 ≤ 1 + 3
        let p = Poly::from_coeffs(vec![r(-3), r(1)]);
        assert_eq!(root_modulus_bound(&p), Some(BigInt::from(4)));
    }

    #[test]
    fn root_bound_quadratic_with_scaling() {
        // 2h² − 6h + 4 = 2(h−1)(h−2): 1 + max(3, 2) = 4 ≥ 2
        let p = Poly::from_coeffs(vec![r(4), r(-6), r(2)]);
        assert_eq!(root_modulus_bound(&p), Some(BigInt::from(4)));
    }

    #[test]
    fn root_bound_constant_and_zero() {
        assert_eq!(root_modulus_bound(&Poly::from_int(7)), Some(BigInt::zero()));
        assert_eq!(root_modulus_bound(&Poly::zero()), None);
        // h: only root is 0 ≤ 1
        assert_eq!(root_modulus_bound(&Poly::x()), Some(BigInt::from(1)));
    }

    #[test]
    fn normal_form_high_degree_is_fast() {
        // ratio of k⁸·2ᵏ: p = 2(k+1)⁸, q = k⁸ → a = 2, b = 1, c = k⁸.
        let kp1 = Poly::from_coeffs(vec![r(1), r(1)]);
        let mut p = Poly::from_int(2);
        let mut q = Poly::from_int(1);
        for _ in 0..8 {
            p = &p * &kp1;
            q = &q * &Poly::x();
        }
        let start = std::time::Instant::now();
        let (a, b, c) = gosper_normal(&p, &q).unwrap();
        assert!(start.elapsed().as_secs_f64() < 1.0, "normal form too slow");
        assert_eq!(a, Poly::from_int(2));
        assert_eq!(b, Poly::from_int(1));
        assert_eq!(c, q);
    }

    #[test]
    fn normal_form_dispersion_beyond_cap_is_declined_safely() {
        // p = k + 1000, q = k: the shift 1000 exceeds MAX_DISPERSION, so the
        // normal form keeps a = k + 1000, b = k (incomplete but still an
        // identity), and the certificate for c = 1 does not exist.
        let p = Poly::from_coeffs(vec![r(1000), r(1)]);
        let q = Poly::x();
        let start = std::time::Instant::now();
        let (a, b, c) = gosper_normal(&p, &q).unwrap();
        assert!(start.elapsed().as_secs_f64() < 2.0);
        assert_eq!(c, Poly::from_int(1));
        for kv in 1..=5 {
            let k_val = r(kv);
            let k1_val = r(kv + 1);
            let lhs = &p.eval(&k_val) / &q.eval(&k_val);
            let rhs = &(&a.eval(&k_val) * &c.eval(&k1_val)) / &(&b.eval(&k_val) * &c.eval(&k_val));
            assert_eq!(lhs, rhs);
        }
        // Within the cap the same structure is fully normalised.
        let p = Poly::from_coeffs(vec![r(40), r(1)]);
        let (a, b, c) = gosper_normal(&p, &q).unwrap();
        assert_eq!(a, Poly::from_int(1));
        assert_eq!(b, Poly::from_int(1));
        assert_eq!(c.degree(), Some(40));
    }

    // ── gosper_normal ───────────────────────────────────────────────

    #[test]
    fn normal_form_basic_coprime() {
        // p = k + 2, q = k  (ratio for k·(k+1))
        // gcd should be coprime already, but resultant gives roots to process.
        let p = Poly::from_coeffs(vec![r(2), r(1)]); // k + 2
        let q = Poly::x(); // k
        let (a, b, c) = gosper_normal(&p, &q).unwrap();

        // Verify: p/q = a·c(k+1) / (b·c(k))
        // Evaluate at several k values to check.
        for kv in 1..=10 {
            let k_val = r(kv);
            let k1_val = r(kv + 1);
            let lhs = &p.eval(&k_val) / &q.eval(&k_val);
            let rhs = &(&a.eval(&k_val) * &c.eval(&k1_val)) / &(&b.eval(&k_val) * &c.eval(&k_val));
            assert_eq!(lhs, rhs, "mismatch at k={kv}");
        }
    }

    #[test]
    fn normal_form_already_coprime_shifted() {
        // p/q = (k+1)²/k from the f(k)=k·k! example.
        // After normal form: A = k+1, B = 1, C = k
        let p = Poly::from_coeffs(vec![r(1), r(2), r(1)]); // (k+1)²
        let q = Poly::x(); // k
        let (a, b, c) = gosper_normal(&p, &q).unwrap();

        // Verify numerically.
        for kv in 1..=10 {
            let k_val = r(kv);
            let k1_val = r(kv + 1);
            let lhs = &p.eval(&k_val) / &q.eval(&k_val);
            let rhs = &(&a.eval(&k_val) * &c.eval(&k1_val)) / &(&b.eval(&k_val) * &c.eval(&k_val));
            assert_eq!(lhs, rhs, "mismatch at k={kv}");
        }
    }

    #[test]
    fn normal_form_constant_ratio() {
        // p = 2, q = 1 (geometric with r=2)
        let p = Poly::from_int(2);
        let q = Poly::from_int(1);
        let (a, b, c) = gosper_normal(&p, &q).unwrap();
        // Should be (2, 1, 1)
        assert_eq!(a, Poly::from_int(2));
        assert_eq!(b, Poly::from_int(1));
        assert_eq!(c, Poly::from_int(1));
    }

    // ── gosper_certificate ──────────────────────────────────────────

    #[test]
    fn certificate_geometric_r2() {
        // a = 2, b = 1, c = 1 → 2·x(k+1) − 1·x(k) = 1 → x = 1
        let a = Poly::from_int(2);
        let b = Poly::from_int(1);
        let c = Poly::from_int(1);
        let x = gosper_certificate(&a, &b, &c).unwrap();
        assert_eq!(x, Poly::from_int(1));
    }

    #[test]
    fn certificate_factorial_sum() {
        // From f(k) = k·k!: (a, b, c) = (k+1, 1, k)
        // Equation: (k+1)·x(k+1) − 1·x(k) = k
        // Solution: x(k) = 1
        let a = Poly::from_coeffs(vec![r(1), r(1)]); // k + 1
        let b = Poly::from_int(1);
        let c = Poly::x(); // k
        let x = gosper_certificate(&a, &b, &c).unwrap();
        assert_eq!(x, Poly::from_int(1));
    }

    #[test]
    fn certificate_k_times_kp1() {
        // From f(k) = k(k+1): (a, b, c) = (1, 1, k(k+1))
        // Equation: x(k+1) − x(k) = k² + k
        // Solution: x(k) = k³/3 − k/3
        let a = Poly::from_int(1);
        let b = Poly::from_int(1);
        let c = Poly::from_coeffs(vec![r(0), r(1), r(1)]); // k + k²
        let x = gosper_certificate(&a, &b, &c).unwrap();

        // Verify: x(k+1) − x(k) = k² + k
        let x_shifted = poly_shift(&x, 1);
        let diff = &x_shifted - &x;
        assert_eq!(diff, c);
    }

    #[test]
    fn certificate_harmonic_no_solution() {
        // From f(k) = 1/k: (a, b, c) = (k, k+1, 1)
        // Equation: k·x(k+1) − k·x(k) = 1 → k·(x(k+1)−x(k)) = 1
        // No polynomial solution.
        let a = Poly::x();
        let b = Poly::from_coeffs(vec![r(1), r(1)]); // k + 1
        let c = Poly::from_int(1);
        assert!(gosper_certificate(&a, &b, &c).is_none());
    }

    // ── gosper_sum (full algorithm with arena) ──────────────────────

    #[test]
    fn sum_geometric_2k() {
        // Σ_{k=0}^{n} 2^k = 2^(n+1) − 1
        let mut arena = Arena::new();
        let k = arena.symbol("k");
        let n = arena.symbol("n");
        let two = arena.int(2);
        let body = arena.pow(two, k); // 2^k
        let lower = arena.zero;

        let result = gosper_sum(&mut arena, body, k, lower, n);
        assert!(result.is_some(), "2^k should be Gosper-summable");
        let result = result.unwrap();

        // Verify numerically: substitute n = 10 → 2^11 − 1 = 2047
        let ten = arena.int(10);
        let evaluated = subs::subs(&mut arena, result, n, ten);
        let evaluated = eval::eval(&mut arena, evaluated);
        assert_eq!(display(&arena, evaluated), "2047");
    }

    #[test]
    fn sum_geometric_3k() {
        // Σ_{k=0}^{n} 3^k = (3^(n+1) − 1)/2
        let mut arena = Arena::new();
        let k = arena.symbol("k");
        let n = arena.symbol("n");
        let three = arena.int(3);
        let body = arena.pow(three, k);
        let lower = arena.zero;

        let result = gosper_sum(&mut arena, body, k, lower, n);
        assert!(result.is_some(), "3^k should be Gosper-summable");
        let result = result.unwrap();

        // n = 5 → (3^6 − 1)/2 = (729 − 1)/2 = 364
        let five = arena.int(5);
        let evaluated = subs::subs(&mut arena, result, n, five);
        let evaluated = eval::eval(&mut arena, evaluated);
        assert_eq!(display(&arena, evaluated), "364");
    }

    #[test]
    fn sum_k_factorial_k() {
        // Σ_{k=0}^{n} k·k! = (n+1)! − 1
        let mut arena = Arena::new();
        let k = arena.symbol("k");
        let n = arena.symbol("n");
        let k_fact = arena.factorial(k);
        let body = arena.mul(&[k, k_fact]); // k * k!
        let lower = arena.zero;

        let result = gosper_sum(&mut arena, body, k, lower, n);
        assert!(result.is_some(), "k·k! should be Gosper-summable");
        let result = result.unwrap();

        // n = 5 → 6! − 1 = 720 − 1 = 719
        let five = arena.int(5);
        let evaluated = subs::subs(&mut arena, result, n, five);
        let evaluated = eval::eval(&mut arena, evaluated);
        assert_eq!(display(&arena, evaluated), "719");

        // n = 3 → 4! − 1 = 24 − 1 = 23
        let three = arena.int(3);
        let evaluated = subs::subs(&mut arena, result, n, three);
        let evaluated = eval::eval(&mut arena, evaluated);
        assert_eq!(display(&arena, evaluated), "23");
    }

    /// `Σ_{k=lo}^{n} body` via Gosper, checked against direct enumeration.
    fn check_gosper(arena: &mut Arena, body: ExprId, k: ExprId, lo: i64, ns: &[i64]) -> ExprId {
        let n = arena.symbol("n");
        let lo_id = arena.int(lo);
        let result = gosper_sum(arena, body, k, lo_id, n)
            .unwrap_or_else(|| panic!("{} should be Gosper-summable", display(arena, body)));
        for &nv in ns {
            let mut terms = Vec::new();
            for i in lo..=nv {
                let iv = arena.int(i);
                let t = subs::subs(arena, body, k, iv);
                terms.push(eval::eval(arena, t));
            }
            let expected = arena.add(&terms);
            let expected = eval::eval(arena, expected);
            let nv_id = arena.int(nv);
            let got = subs::subs(arena, result, n, nv_id);
            let got = eval::eval(arena, got);
            assert_eq!(
                display(arena, got),
                display(arena, expected),
                "n = {nv}: {}",
                display(arena, result)
            );
        }
        result
    }

    #[test]
    fn sum_k_times_2_pow_k() {
        // Σ_{k=0}^{n} k·2^k = (n−1)·2^(n+1) + 2
        let mut arena = Arena::new();
        let k = arena.symbol("k");
        let two = arena.int(2);
        let two_k = arena.pow(two, k);
        let body = arena.mul(&[k, two_k]);
        check_gosper(&mut arena, body, k, 0, &[0, 1, 2, 5, 10]);
    }

    #[test]
    fn sum_quadratic_times_3_pow_k() {
        // Σ_{k=0}^{n} (k² + k)·3^k
        let mut arena = Arena::new();
        let k = arena.symbol("k");
        let three = arena.int(3);
        let three_k = arena.pow(three, k);
        let two = arena.int(2);
        let k2 = arena.pow(k, two);
        let quad = arena.add(&[k2, k]);
        let body = arena.mul(&[quad, three_k]);
        check_gosper(&mut arena, body, k, 0, &[0, 1, 3, 6]);
    }

    #[test]
    fn sum_reciprocal_k_k_plus_1() {
        // Σ_{k=1}^{n} 1/(k(k+1)) = 1 − 1/(n+1)
        let mut arena = Arena::new();
        let k = arena.symbol("k");
        let one = arena.one;
        let k1 = arena.add(&[k, one]);
        let denom = arena.mul(&[k, k1]);
        let body = arena.div(one, denom);
        let result = check_gosper(&mut arena, body, k, 1, &[1, 2, 3, 9]);
        assert!(!walk::has_unevaluated(&arena, result));
    }

    #[test]
    fn is_hypergeometric_ratios() {
        let mut arena = Arena::new();
        let k = arena.symbol("k");
        let n = arena.symbol("n");
        let kf = arena.factorial(k);
        let r = is_hypergeometric(&mut arena, kf, k).unwrap();
        assert_eq!(display(&arena, r), "k + 1");
        // C(n, k): ratio (n − k)/(k + 1)
        let c = arena.binomial(n, k);
        let r = is_hypergeometric(&mut arena, c, k).unwrap();
        let five = arena.int(5);
        let two = arena.int(2);
        let at = subs::subs(&mut arena, r, n, five);
        let at = subs::subs(&mut arena, at, k, two);
        let at = eval::eval(&mut arena, at);
        assert_eq!(display(&arena, at), "1");
        // constant term → ratio 1
        let r = is_hypergeometric(&mut arena, n, k).unwrap();
        assert_eq!(display(&arena, r), "1");
        // not hypergeometric
        let s = arena.sin(k);
        assert!(is_hypergeometric(&mut arena, s, k).is_none());
        let kk = arena.pow(k, k);
        assert!(is_hypergeometric(&mut arena, kk, k).is_none());
        let h = arena.harmonic(k);
        assert!(is_hypergeometric(&mut arena, h, k).is_none());
        // the summation index must be a symbol
        assert!(is_hypergeometric(&mut arena, kf, five).is_none());
    }

    #[test]
    fn certificate_degree_bound_uses_second_coefficients() {
        // Σ k³: ratio (k+1)³/k³ → normal form a = b = 1, c = k³, so the
        // operator is the forward difference and x has degree 4 = deg c + 1.
        let a = Poly::from_int(1);
        let b = Poly::from_int(1);
        let c = Poly::from_coeffs(vec![r(0), r(0), r(0), r(1)]);
        let x = gosper_certificate(&a, &b, &c).expect("certificate");
        assert_eq!(x.degree(), Some(4));
        // a = k² − 3k, b = k² + k − 3 so b(k−1) = k² − k − 3, c = 1.  Leading
        // terms agree; the k-coefficients are A = −3 and B = −1, so
        // d₀ = (B − A)/lc = 2.  The unique solution is
        // x = −2/9·k² + 4/9·k + 1/3 (degree 2): for x = u·k + v one gets
        // L(x) = −u·k² − 2v·k + 3v, which is never 1, so the naive bound
        // deg c + 1 = 1 would wrongly report "not summable".
        let a = Poly::from_coeffs(vec![r(0), r(-3), r(1)]);
        let b = Poly::from_coeffs(vec![r(-3), r(1), r(1)]);
        let c = Poly::from_int(1);
        assert_eq!(
            poly_shift(&b, -1),
            Poly::from_coeffs(vec![r(-3), r(-1), r(1)])
        );
        let x = gosper_certificate(&a, &b, &c).expect("degree-2 certificate");
        assert_eq!(x.degree(), Some(2));
        assert_eq!(
            x,
            Poly::from_coeffs(vec![
                Ratio::new(BigInt::from(1), BigInt::from(3)),
                Ratio::new(BigInt::from(4), BigInt::from(9)),
                Ratio::new(BigInt::from(-2), BigInt::from(9))
            ])
        );
        let lhs = &(&a * &poly_shift(&x, 1)) - &(&poly_shift(&b, -1) * &x);
        assert_eq!(lhs, c);
    }

    #[test]
    fn certificate_degree_cap_declines_huge_systems() {
        // a = b = 1 and c = k^70: the certificate would need degree 71 > cap.
        let a = Poly::from_int(1);
        let b = Poly::from_int(1);
        let c = Poly::monomial(r(1), MAX_CERTIFICATE_DEGREE + 6);
        let start = std::time::Instant::now();
        assert!(gosper_certificate(&a, &b, &c).is_none());
        assert!(start.elapsed().as_secs_f64() < 1.0);
        // Just under the cap is still attempted (and solved).
        let c = Poly::monomial(r(1), 8);
        assert!(gosper_certificate(&a, &b, &c).is_some());
    }

    #[test]
    fn sum_harmonic_not_summable() {
        // Σ 1/k is NOT Gosper-summable.
        let mut arena = Arena::new();
        let k = arena.symbol("k");
        let n = arena.symbol("n");
        let body = arena.div(arena.one, k); // 1/k
        let lower = arena.one;

        let result = gosper_sum(&mut arena, body, k, lower, n);
        assert!(
            result.is_none(),
            "harmonic series should NOT be Gosper-summable"
        );
    }

    #[test]
    fn sum_binomial_not_gosper_summable() {
        // Σ_{k=0}^{n} C(n,k) = 2^n, but this is NOT Gosper-summable.
        // (It requires Zeilberger's creative telescoping, not Gosper.)
        let mut arena = Arena::new();
        let k = arena.symbol("k");
        let n = arena.symbol("n");
        let body = arena.binomial(n, k);
        let lower = arena.zero;

        let result = gosper_sum(&mut arena, body, k, lower, n);
        // The Gosper algorithm should fail for this because the certificate
        // equation has no polynomial solution.
        assert!(result.is_none(), "C(n,k) should NOT be Gosper-summable");
    }
}

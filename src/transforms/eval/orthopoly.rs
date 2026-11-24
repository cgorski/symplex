//! Explicit coefficients of the classical orthogonal polynomials (part of [`eval`](super)).

use super::*;

// ── Explicit coefficients of the classical orthogonal polynomials ─────────
//
// Up to 0.29 `p_n(x)` was built by its three-term recurrence, each step
// expanded and re-canonicalised: `O(n)` canonicalisations of polynomials
// with `O(n)` big coefficients.  `legendre(300, x)` took 6 s in `eval`, and
// `legendre(1000, x)`, `hermite(1000, x)`, `gegenbauer(1000, 1/3, x)`,
// `assoc_legendre(1000, 3, x)`, `jacobi(1000, 1/3, 1/5, x)` did not finish
// within a minute.  The coefficients have closed forms (hypergeometric sums,
// DLMF §18.5): each follows from the previous one by a rational factor, so
// the whole polynomial costs `O(n)` rational operations (Jacobi, written in
// powers of `(x − 1)/2`, needs an `O(n²)` change of basis for a symbolic
// `x`).  Every coefficient, and a numeric value, is within the digit guard.

/// A family of [`ortho_coeffs`], with its numeric parameters.
#[derive(Clone, Debug)]
pub(super) enum OrthoFamily {
    Legendre,
    ChebyshevT,
    ChebyshevU,
    Hermite,
    Laguerre,
    /// `C_n^{(α)}`
    Gegenbauer(Q),
    /// `L_n^{(α)}`
    AssocLaguerre(Q),
}

pub(super) fn q_int(n: i64) -> Q {
    Ratio::from_integer(BigInt::from(n))
}

/// Is the rational `r` within the digit guard (numerator and denominator
/// bits, a cheap count)?
pub(super) fn within_guard_bits(arena: &Arena, r: &Q) -> bool {
    let bits = r.numer().bits() + r.denom().bits();
    (bits as f64) * std::f64::consts::LOG10_2 <= arena.config.max_result_digits as f64 + 2.0
}

/// The coefficients `c[i]` of `xⁱ` (`i = 0..=n`) of the family's `p_n`,
/// from the explicit sums (DLMF 18.5.10–18.5.13, 18.5.12 for Laguerre):
///
/// * `P_n = 2⁻ⁿ Σ_k (−1)^k C(n, k) C(2n − 2k, n) x^{n−2k}`
/// * `T_n = (n/2) Σ_k (−1)^k (n−k−1)!/(k!(n−2k)!) (2x)^{n−2k}` (`n ≥ 1`)
/// * `U_n = Σ_k (−1)^k C(n−k, k) (2x)^{n−2k}`
/// * `H_n = n! Σ_k (−1)^k (2x)^{n−2k}/(k!(n−2k)!)`
/// * `C_n^{(α)} = Σ_k (−1)^k (α)_{n−k}/(k!(n−2k)!) (2x)^{n−2k}`
/// * `L_n^{(α)} = Σ_k (−1)^k C(n+α, n−k) x^k/k!`
///
/// `None` beyond the digit guard, or when a ratio would divide by zero
/// (`α` a non-positive integer, where the recurrences degenerate).
pub(super) fn ortho_coeffs(arena: &Arena, family: &OrthoFamily, n: usize) -> Option<Vec<Q>> {
    let mut c = vec![Q::zero(); n + 1];
    let ni = n as i64;
    let even_family = !matches!(
        family,
        OrthoFamily::Laguerre | OrthoFamily::AssocLaguerre(_)
    );
    if even_family {
        // Leading coefficient, then c[n−2k−2] = c[n−2k] · ratio(k).
        let mut lead = match family {
            OrthoFamily::Legendre => {
                // C(2n, n)/2ⁿ
                Ratio::new(
                    crate::base::combinatorics::binomial(2 * n as u64, n as u64),
                    BigInt::one() << n,
                )
            }
            OrthoFamily::ChebyshevT => {
                if n == 0 {
                    Q::one()
                } else {
                    Ratio::from_integer(BigInt::one() << (n - 1))
                }
            }
            OrthoFamily::ChebyshevU | OrthoFamily::Hermite => {
                Ratio::from_integer(BigInt::one() << n)
            }
            OrthoFamily::Gegenbauer(a) => {
                // (α)_n 2ⁿ/n!
                let mut acc = Q::one();
                for j in 0..ni {
                    acc *= a + q_int(j);
                    acc /= q_int(j + 1);
                    acc *= q_int(2);
                }
                acc
            }
            OrthoFamily::Laguerre | OrthoFamily::AssocLaguerre(_) => return None,
        };
        if matches!(family, OrthoFamily::ChebyshevT) && n == 0 {
            c[0] = Q::one();
            return Some(c);
        }
        for k in 0..=(ni / 2) {
            if !within_guard_bits(arena, &lead) {
                return None;
            }
            c[(ni - 2 * k) as usize] = lead.clone();
            if 2 * k + 2 > ni {
                break;
            }
            let (m, m1) = (q_int(ni - 2 * k), q_int(ni - 2 * k - 1));
            let kp1 = q_int(k + 1);
            let den = match family {
                OrthoFamily::Legendre => q_int(2) * &kp1 * q_int(2 * ni - 2 * k - 1),
                OrthoFamily::ChebyshevT => q_int(4) * &kp1 * q_int(ni - k - 1),
                OrthoFamily::ChebyshevU => q_int(4) * &kp1 * q_int(ni - k),
                OrthoFamily::Hermite => q_int(4) * &kp1,
                OrthoFamily::Gegenbauer(a) => q_int(4) * &kp1 * (q_int(ni - k - 1) + a),
                OrthoFamily::Laguerre | OrthoFamily::AssocLaguerre(_) => return None,
            };
            if den.is_zero() {
                return None;
            }
            lead = -(lead * m * m1) / den;
        }
        return Some(c);
    }
    // Laguerre families: c[0] = C(n+α, n), c[k+1] = −c[k]·(n−k)/((k+1)(α+k+1)).
    let alpha = match family {
        OrthoFamily::AssocLaguerre(a) => a.clone(),
        _ => Q::zero(),
    };
    let mut term = Q::one();
    for j in 0..ni {
        term *= &alpha + q_int(j + 1);
        term /= q_int(j + 1);
    }
    for k in 0..=ni {
        if !within_guard_bits(arena, &term) {
            return None;
        }
        c[k as usize] = term.clone();
        if k == ni {
            break;
        }
        let den = q_int(k + 1) * (&alpha + q_int(k + 1));
        if den.is_zero() {
            return None;
        }
        term = -(term * q_int(ni - k)) / den;
    }
    Some(c)
}

/// The Jacobi coefficients `d[ℓ]` of `((x − 1)/2)^ℓ` in `P_n^{(α,β)}`
/// (DLMF 18.5.8): `d_ℓ = (n+α+β+1)_ℓ (α+ℓ+1)_{n−ℓ}/(ℓ!(n−ℓ)!)`,
/// `d_{ℓ+1} = d_ℓ (n+α+β+1+ℓ)(n−ℓ)/((ℓ+1)(α+ℓ+1))`.
fn jacobi_shifted_coeffs(arena: &Arena, n: usize, a: &Q, b: &Q) -> Option<Vec<Q>> {
    let ni = n as i64;
    let mut d = Vec::with_capacity(n + 1);
    // d_0 = (α+1)_n/n!
    let mut term = Q::one();
    for j in 0..ni {
        term *= a + q_int(j + 1);
        term /= q_int(j + 1);
    }
    for l in 0..=ni {
        if !within_guard_bits(arena, &term) {
            return None;
        }
        d.push(term.clone());
        if l == ni {
            break;
        }
        let den = q_int(l + 1) * (a + q_int(l + 1));
        if den.is_zero() {
            return None;
        }
        term = term * (q_int(ni + 1 + l) + a + b) * q_int(ni - l) / den;
    }
    Some(d)
}

/// `Σ c[i] tⁱ` by Horner's rule, refused (`None`) as soon as the partial
/// value outgrows twice the digit guard (`chebyshev_t(1000, 1/10^20)` built
/// 20 000-digit intermediates for 11 s before the final guard refused).
pub(super) fn horner(arena: &Arena, c: &[Q], t: &Q) -> Option<Q> {
    let limit_bits =
        2.0 * arena.config.max_result_digits as f64 / std::f64::consts::LOG10_2 + 256.0;
    let mut acc = Q::zero();
    for ci in c.iter().rev() {
        acc = acc * t + ci;
        if (acc.numer().bits() + acc.denom().bits()) as f64 > limit_bits {
            return None;
        }
    }
    Some(acc)
}

/// `Σ c[i] xⁱ` as an expression; expanded when `x` is a compound (the
/// recurrences expanded at every step, so `legendre(2, y + 1)` is
/// `3/2·y² + 3y + 1`).  `None` if a coefficient is beyond the digit guard.
pub(super) fn poly_from_coeffs(arena: &mut Arena, c: &[Q], x: ExprId) -> Option<ExprId> {
    let mut terms = Vec::with_capacity(c.len());
    for (i, ci) in c.iter().enumerate() {
        if ci.is_zero() {
            continue;
        }
        let coeff = guarded_num(arena, ci.clone())?;
        let term = match i {
            0 => coeff,
            1 => arena.mul(&[coeff, x]),
            _ => {
                let e = arena.int(i as i64);
                let p = arena.pow(x, e);
                arena.mul(&[coeff, p])
            }
        };
        terms.push(term);
    }
    let sum = match terms.len() {
        0 => arena.zero,
        1 => terms[0],
        _ => arena.add(&terms),
    };
    if matches!(arena.node(x), ExprNode::Symbol(_)) {
        Some(sum)
    } else {
        Some(expand_eval(arena, sum))
    }
}

/// `p_n(x)` of `family` from its coefficients: the exact value for a
/// rational `x`, else the explicit polynomial.
pub(super) fn ortho_value(
    arena: &mut Arena,
    family: &OrthoFamily,
    n: usize,
    x: ExprId,
) -> Option<ExprId> {
    let c = ortho_coeffs(arena, family, n)?;
    if let Some(xr) = arena.as_num(x).cloned() {
        let v = horner(arena, &c, &xr)?;
        return guarded_num(arena, v);
    }
    poly_from_coeffs(arena, &c, x)
}

/// `P_n^{(α,β)}(x)` for numeric `α`, `β`: Horner in `(x − 1)/2` for a
/// rational `x`, else the monomial coefficients
/// `Σ_{ℓ≥j} d_ℓ 2^{−ℓ} C(ℓ, j) (−1)^{ℓ−j}` of `xʲ`.
pub(super) fn jacobi_value(arena: &mut Arena, n: usize, a: &Q, b: &Q, x: ExprId) -> Option<ExprId> {
    let d = jacobi_shifted_coeffs(arena, n, a, b)?;
    if let Some(xr) = arena.as_num(x).cloned() {
        let y = (xr - Q::one()) / q_int(2);
        let v = horner(arena, &d, &y)?;
        return guarded_num(arena, v);
    }
    // With u = x − 1, Σ d_ℓ (u/2)^ℓ = Σ e_ℓ u^ℓ (e_ℓ = d_ℓ/2^ℓ); over a
    // common denominator the Taylor shift u = x − 1 is `O(n²)` integer
    // subtractions (Knuth's scheme: a_j −= a_{j+1} for j = n−1 … i).
    let e: Vec<Q> = d
        .iter()
        .enumerate()
        .map(|(l, dl)| dl / Ratio::from_integer(BigInt::one() << l))
        .collect();
    let mut den = BigInt::one();
    for el in &e {
        den = num_integer::Integer::lcm(&den, el.denom());
    }
    let mut a: Vec<BigInt> = e
        .iter()
        .map(|el| el.numer() * (&den / el.denom()))
        .collect();
    for i in 0..n {
        for j in (i..n).rev() {
            let next = a[j + 1].clone();
            a[j] -= next;
        }
    }
    let c: Vec<Q> = a
        .into_iter()
        .map(|aj| Ratio::new(aj, den.clone()))
        .collect();
    poly_from_coeffs(arena, &c, x)
}

// ═══════════════════════════════════════════════════════════════════════════
// Orthogonal polynomial evaluation via recurrence relations
// ═══════════════════════════════════════════════════════════════════════════

/// Legendre polynomial P_n(x) via Bonnet's recurrence.
/// P_0(x) = 1, P_1(x) = x, (k+1)·P_{k+1} = (2k+1)·x·P_k − k·P_{k-1}
#[cfg(test)]
pub(super) fn eval_legendre(arena: &mut Arena, n: usize, x: ExprId) -> ExprId {
    if n == 0 {
        return arena.one;
    }
    if n == 1 {
        return x;
    }

    let mut p_prev = arena.one;
    let mut p_curr = x;

    for k in 1..n {
        let two_k_plus_1 = arena.int(2 * k as i64 + 1);
        let k_val = arena.int(k as i64);
        let k_plus_1 = arena.int(k as i64 + 1);

        let term1 = arena.mul(&[two_k_plus_1, x, p_curr]);
        let term2 = arena.mul(&[k_val, p_prev]);
        let numer = arena.sub(term1, term2);
        let p_next = arena.div(numer, k_plus_1);

        // Expand and eval at each step to keep expressions manageable
        let p_next = crate::transforms::expand::expand(arena, p_next);
        let p_next = eval(arena, p_next);

        p_prev = p_curr;
        p_curr = p_next;
    }

    p_curr
}

/// Chebyshev polynomial of the first kind T_n(x) via recurrence.
/// T_0(x) = 1, T_1(x) = x, T_{k+1} = 2·x·T_k − T_{k-1}
#[cfg(test)]
pub(super) fn eval_chebyshev_t(arena: &mut Arena, n: usize, x: ExprId) -> ExprId {
    if n == 0 {
        return arena.one;
    }
    if n == 1 {
        return x;
    }

    let two = arena.int(2);
    let mut t_prev = arena.one;
    let mut t_curr = x;

    for _ in 1..n {
        let term1 = arena.mul(&[two, x, t_curr]);
        let t_next = arena.sub(term1, t_prev);

        let t_next = crate::transforms::expand::expand(arena, t_next);
        let t_next = eval(arena, t_next);

        t_prev = t_curr;
        t_curr = t_next;
    }

    t_curr
}

/// Chebyshev polynomial of the second kind U_n(x) via recurrence.
/// U_0(x) = 1, U_1(x) = 2x, U_{k+1} = 2·x·U_k − U_{k-1}
#[cfg(test)]
pub(super) fn eval_chebyshev_u(arena: &mut Arena, n: usize, x: ExprId) -> ExprId {
    if n == 0 {
        return arena.one;
    }
    let two = arena.int(2);
    if n == 1 {
        return arena.mul(&[two, x]);
    }

    let mut u_prev = arena.one;
    let mut u_curr = arena.mul(&[two, x]);

    for _ in 1..n {
        let term1 = arena.mul(&[two, x, u_curr]);
        let u_next = arena.sub(term1, u_prev);

        let u_next = crate::transforms::expand::expand(arena, u_next);
        let u_next = eval(arena, u_next);

        u_prev = u_curr;
        u_curr = u_next;
    }

    u_curr
}

/// Physicist's Hermite polynomial H_n(x) via recurrence.
/// H_0(x) = 1, H_1(x) = 2x, H_{k+1} = 2·x·H_k − 2k·H_{k-1}
#[cfg(test)]
pub(super) fn eval_hermite(arena: &mut Arena, n: usize, x: ExprId) -> ExprId {
    if n == 0 {
        return arena.one;
    }
    let two = arena.int(2);
    if n == 1 {
        return arena.mul(&[two, x]);
    }

    let mut h_prev = arena.one;
    let mut h_curr = arena.mul(&[two, x]);

    for k in 1..n {
        let two_k = arena.int(2 * k as i64);
        let term1 = arena.mul(&[two, x, h_curr]);
        let term2 = arena.mul(&[two_k, h_prev]);
        let h_next = arena.sub(term1, term2);

        let h_next = crate::transforms::expand::expand(arena, h_next);
        let h_next = eval(arena, h_next);

        h_prev = h_curr;
        h_curr = h_next;
    }

    h_curr
}

/// Laguerre polynomial L_n(x) via recurrence.
/// L_0(x) = 1, L_1(x) = 1−x, (k+1)·L_{k+1} = (2k+1−x)·L_k − k·L_{k-1}
#[cfg(test)]
pub(super) fn eval_laguerre(arena: &mut Arena, n: usize, x: ExprId) -> ExprId {
    if n == 0 {
        return arena.one;
    }
    if n == 1 {
        return arena.sub(arena.one, x);
    }

    let mut l_prev = arena.one;
    let mut l_curr = arena.sub(arena.one, x);

    for k in 1..n {
        let two_k_plus_1 = arena.int(2 * k as i64 + 1);
        let k_val = arena.int(k as i64);
        let k_plus_1 = arena.int(k as i64 + 1);

        let coeff = arena.sub(two_k_plus_1, x); // (2k+1 − x)
        let term1 = arena.mul(&[coeff, l_curr]);
        let term2 = arena.mul(&[k_val, l_prev]);
        let numer = arena.sub(term1, term2);
        let l_next = arena.div(numer, k_plus_1);

        let l_next = crate::transforms::expand::expand(arena, l_next);
        let l_next = eval(arena, l_next);

        l_prev = l_curr;
        l_curr = l_next;
    }

    l_curr
}

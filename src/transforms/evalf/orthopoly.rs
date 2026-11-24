//! Orthogonal polynomials (part of [`evalf`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Orthogonal polynomials
// ═══════════════════════════════════════════════════════════════════════════

/// Families of classical orthogonal polynomials evaluated numerically.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum OrthoPoly {
    Legendre,
    ChebyshevT,
    ChebyshevU,
    Hermite,
    Laguerre,
}

/// Evaluate `P_n(x)` (or `T_n`, `U_n`, `H_n`, `L_n`) for any non-negative
/// integer `n` via the three-term recurrence:
///
/// * Legendre: `(k+1) P_{k+1} = (2k+1) x P_k − k P_{k−1}`
/// * Chebyshev T/U: `T_{k+1} = 2x T_k − T_{k−1}` (`T_1 = x`, `U_1 = 2x`)
/// * Hermite (physicists'): `H_{k+1} = 2x H_k − 2k H_{k−1}`
/// * Laguerre: `(k+1) L_{k+1} = (2k+1−x) L_k − k L_{k−1}`
///
/// The last value is a difference of two terms of the size of its
/// neighbours: its error, a part of the recurrence's envelope and not of
/// the value (next to a zero), is charged to `w` (`wp::Recurred`).  The
/// norms of `P_k`, `T_k`, `U_k`, `L_k` do not grow with `k`; that of `H_k`
/// grows by `√(2(k+1))` per step.
pub(super) fn arb_orthopoly(kind: OrthoPoly, n: u64, x: &BigFloat, w: &mut Wp<'_>) -> BigFloat {
    let prec = w.bits();
    let wp = prec + 32 + (n as f64).log2().ceil().max(0.0) as usize;
    let one = BigFloat::from_i32(1, wp);
    let two = BigFloat::from_i32(2, wp);
    let mut xw = x.clone();
    let _ = xw.set_precision(wp, RM);
    if n == 0 {
        return BigFloat::from_i32(1, prec);
    }
    let mut p_prev = one.clone();
    let mut p_curr = match kind {
        OrthoPoly::Legendre | OrthoPoly::ChebyshevT => xw.clone(),
        OrthoPoly::ChebyshevU | OrthoPoly::Hermite => two.mul(&xw, wp, RM),
        OrthoPoly::Laguerre => one.sub(&xw, wp, RM),
    };
    let mut seen = Recurred::new(&[&p_prev, &p_curr]);
    for k in 1..n {
        let k_bf = BigFloat::from_i128(k as i128, wp);
        let kp1 = BigFloat::from_i128(k as i128 + 1, wp);
        let next = match kind {
            OrthoPoly::Legendre => {
                let two_k_plus_1 = BigFloat::from_i128(2 * k as i128 + 1, wp);
                let a = two_k_plus_1.mul(&xw, wp, RM).mul(&p_curr, wp, RM);
                let b = k_bf.mul(&p_prev, wp, RM);
                a.sub(&b, wp, RM).div(&kp1, wp, RM)
            }
            OrthoPoly::ChebyshevT | OrthoPoly::ChebyshevU => two
                .mul(&xw, wp, RM)
                .mul(&p_curr, wp, RM)
                .sub(&p_prev, wp, RM),
            OrthoPoly::Hermite => {
                let a = two.mul(&xw, wp, RM).mul(&p_curr, wp, RM);
                let b = two.mul(&k_bf, wp, RM).mul(&p_prev, wp, RM);
                a.sub(&b, wp, RM)
            }
            OrthoPoly::Laguerre => {
                let two_k_plus_1 = BigFloat::from_i128(2 * k as i128 + 1, wp);
                let a = two_k_plus_1.sub(&xw, wp, RM).mul(&p_curr, wp, RM);
                let b = k_bf.mul(&p_prev, wp, RM);
                a.sub(&b, wp, RM).div(&kp1, wp, RM)
            }
        };
        let growth = match kind {
            OrthoPoly::Hermite => 0.5 * (2.0 * (k as f64 + 1.0)).log2(),
            _ => 0.0,
        };
        seen.step(growth, &next);
        p_prev = p_curr;
        p_curr = next;
    }
    let symmetric = !matches!(kind, OrthoPoly::Laguerre);
    w.charge(seen.error(wp, odd_member_scale(symmetric && n % 2 == 1, n, &xw)));
    round_to(p_curr, prec)
}

/// `log₂` of the factor `min(1, n·|x|)` by which the error of the member of
/// degree `n` of a family symmetric in `x` is below its envelope's when the
/// member is `odd`: the roundings of the even members reach it through an
/// odd polynomial of `x` (at most `n·|x|` of its size, Bernstein's
/// inequality), and its own steps are differences of terms with the factor
/// `x`.  So `P_n(x) ≈ P_n′(0)·x` next to 0 keeps a bound that is a part of
/// the value, and an exact 0 at `x = 0` stays exact.
fn odd_member_scale(odd: bool, n: u64, x: &BigFloat) -> f64 {
    if odd {
        (accuracy::part_lg(x) + (n as f64).log2().max(0.0)).min(0.0)
    } else {
        0.0
    }
}

// ── Orthogonal polynomials with parameters ───────────────────────────────────────

/// The degree of an orthogonal polynomial as a non-negative integer.
fn poly_degree(n: &BigFloat, what: &str) -> Result<u64, SymplexError> {
    // An integer exactly: before 0.29 a degree within 10⁻¹² of one was
    // taken for it (`jacobi(3 + 10⁻¹³, …)` evaluated the cubic).
    match bf_as_int(n)? {
        Some(d) if n.is_int() && (0..=10_000_000).contains(&d) => Ok(d as u64),
        _ => Err(unevaluable(format!(
            "{what}: degree must be a non-negative integer"
        ))),
    }
}

/// Gegenbauer `C_n^{(a)}(x)`: `(k+1) C_{k+1} = 2(k+a) x C_k − (k+2a−1) C_{k−1}`.
pub(super) fn arb_gegenbauer(
    n: &BigFloat,
    a: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    let deg = poly_degree(n, "gegenbauer")?;
    let wp = prec + 32 + (deg as f64).log2().ceil().max(0.0) as usize;
    let one = BigFloat::from_i32(1, wp);
    let two = BigFloat::from_i32(2, wp);
    if deg == 0 {
        return Ok(BigFloat::from_i32(1, prec));
    }
    let mut prev = one.clone();
    let mut curr = two.mul(a, wp, RM).mul(x, wp, RM);
    // The norm of `C_k^(a)` grows by at most `(k + 2a)/(k + 1)` per step
    // (`C_k^(a)(1) = Γ(k + 2a)/(k!·Γ(2a))`); see `arb_orthopoly`.
    let a_f = f64_of(a).unwrap_or(f64::NAN);
    // (`C_0 = 1` enters `C_2` with the factor `a`: the envelope starts at
    // `C_1 = 2ax`, so that a small `a` keeps a bound of the values' size.)
    let mut seen = Recurred::new(&[&curr]);
    for k in 1..deg {
        let k_bf = BigFloat::from_i128(k as i128, wp);
        let t1 = two
            .mul(&k_bf.add(a, wp, RM), wp, RM)
            .mul(x, wp, RM)
            .mul(&curr, wp, RM);
        let coeff = k_bf.add(&two.mul(a, wp, RM), wp, RM).sub(&one, wp, RM);
        let t2 = coeff.mul(&prev, wp, RM);
        let next = t1
            .sub(&t2, wp, RM)
            .div(&BigFloat::from_i128(k as i128 + 1, wp), wp, RM);
        seen.step(
            ((k as f64 + 2.0 * a_f).abs() / (k as f64 + 1.0)).log2(),
            &next,
        );
        prev = curr;
        curr = next;
    }
    w.charge(seen.error(wp, odd_member_scale(deg % 2 == 1, deg, x)));
    Ok(round_to(curr, prec))
}

/// The largest degree of [`arb_jacobi`]: its sum takes `O(n)` operations at
/// about `n·log₂(1 + √(1 − x²))` bits: 0.12 s at 2500, 0.6 s at 5000
/// (release), and `O(n²)` with the precision beyond.
const MAX_JACOBI_DEGREE: usize = 5000;

/// Jacobi `P_n^{(a,b)}(x)` via the explicit sum
/// `Σ_s C(n+a, n−s) C(n+b, s) ((x−1)/2)^s ((x+1)/2)^{n−s}`, its
/// cancellation measured: the largest term over the value, made up with as
/// many more bits (mpmath's `hypsum` does the same for the `₂F₁` of its
/// `jacobi`).  The first pass takes `2n` guard bits up to degree 2000 (the
/// same digits as before 0.35, which took them unmeasured — a value next to
/// a zero of `P_n` could lose more), `n·log₂(1 + √(1 − x²))` beyond (the
/// terms' sum over the value's scale for `abs(x) < 1`).  Before 0.35 a degree
/// beyond 2000 was refused: `jacobi(2500, 1/3, 1/5, 1/7)`.
///
/// The coefficients are updated term by term: `∏_{j>s}(a + j)` divided by
/// `a + s` (from the first `s` at which it is nonzero: a negative integer
/// `a = −m` makes the terms below `m` zero), `∏_{j>n−s}(b + j)` multiplied by
/// `b + n − s + 1`, `1/((n−s)!·s!)` by `(n − s + 1)/s`, the powers by
/// `(x−1)/2` and divided by `(x+1)/2`.  Before 0.34 each term recomputed its
/// products and factorials (`O(n²)` operations and exact factorials):
/// `jacobi(1000, 1/3, 1/5, 1/3)` 1.3 s, `jacobi(2000, …)` and up never
/// finished.  Refused beyond [`MAX_JACOBI_DEGREE`].
pub(super) fn arb_jacobi(
    n: &BigFloat,
    a: &BigFloat,
    b: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    let deg = poly_degree(n, "jacobi")? as usize;
    if deg > MAX_JACOBI_DEGREE {
        return Err(SymplexError::NotImplemented(format!(
            "jacobi of degree {deg} (at most {MAX_JACOBI_DEGREE} in evalf)"
        )));
    }
    if deg == 0 {
        return Ok(BigFloat::from_i32(1, prec));
    }
    // `P_n^(a,b)(−x) = (−1)ⁿ P_n^(b,a)(x)`: 0 at `x = 0` for `a = b` and an
    // odd `n` (whose terms cancel exactly, which no measure of the loss
    // certifies).
    if x.is_zero() && deg % 2 == 1 && a.cmp(b) == Some(0) {
        return Ok(BigFloat::new(prec));
    }
    // The rounding of the sum: each term carries at most `12(n + 2)` roundings
    // relative (running products, powers), the additions `n` of the largest
    // partial sum: below `2^(4 + 2·log₂(n + 2) − wp)` of the largest term.
    let rounding = 4 + 2 * (usize::BITS - (deg + 2).leading_zeros()) as usize;
    // The first pass: `2n` guard bits up to the old maximum (the same digits
    // as before 0.35), beyond the cancellation of `abs(x) < 1` estimated as
    // `n·log₂(1 + √(1 − x²))` (the terms' sum over the value's scale).
    let first_extra = if deg <= 2000 {
        32 + 2 * deg
    } else {
        let xf = f64_of(x)?.abs();
        let est = if xf < 1.0 {
            deg as f64 * (1.0 + (1.0 - xf * xf).sqrt()).log2()
        } else {
            0.0
        };
        40 + rounding + est.ceil() as usize
    };
    // At most the `2n` guard bits of before 0.35 beyond `cancellation_cap`.
    let cap = cancellation_cap(prec) + 2 * deg + 64;
    let mut extra = first_extra;
    loop {
        let wp = prec + extra;
        let (sum, largest) = jacobi_sum(deg, a, b, x, &mut w.at(wp))?;
        // The loss: the largest term over the value.
        let lost = match (sum.exponent(), largest) {
            (Some(e), Some(l)) if !sum.is_zero() => (i64::from(l) - i64::from(e)).max(0) as usize,
            (_, None) => 0,
            _ => wp,
        };
        if lost + rounding + 8 <= extra {
            return Ok(round_to(sum, prec));
        }
        if extra >= cap {
            return Err(special_exhausted(prec));
        }
        extra = (lost + rounding + 40).max(extra + 64).min(cap);
    }
}

/// The sum of [`arb_jacobi`] at `wp` bits, and the exponent of its largest
/// term (`None` when all are 0).
fn jacobi_sum(
    deg: usize,
    a: &BigFloat,
    b: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<(BigFloat, Option<i32>), SymplexError> {
    let wp = w.bits();
    let one = BigFloat::from_i32(1, wp);
    let two = BigFloat::from_i32(2, wp);
    let lo = x.sub(&one, wp, RM).div(&two, wp, RM);
    let hi = x.add(&one, wp, RM).div(&two, wp, RM);
    let int = |j: usize| BigFloat::from_u64(j as u64, 64);
    // ∏_{j=from}^{deg} (p + j).
    let shifted = |p: &BigFloat, from: usize| -> BigFloat {
        let mut acc = one.clone();
        for j in from..=deg {
            acc = acc.mul(&p.add(&int(j), wp, RM), wp, RM);
        }
        acc
    };
    // `a + j` vanishes at `j = −a` for a negative integer `a`: the terms
    // `s < −a` (whose product holds that factor) are 0.
    let first = match bf_as_int(a)? {
        Some(m) if a.is_int() && m < 0 && m.unsigned_abs() as usize <= deg => {
            m.unsigned_abs() as usize
        }
        _ => 0,
    };
    // The sum over `s ≥ first`: (x+1)/2 = 0 leaves the term `s = n` alone.
    let hi_zero = hi.is_zero();
    let ratio = if hi_zero {
        BigFloat::new(wp)
    } else {
        lo.div(&hi, wp, RM)
    };
    let mut pa = shifted(a, first + 1);
    let mut pb = one.clone();
    for s in 1..=first {
        pb = pb.mul(&b.add(&int(deg - s + 1), wp, RM), wp, RM);
    }
    // 1/((n − first)!·first!).
    let f0 = factorial_bigint(deg - first) * factorial_bigint(first);
    let mut inv_fact = one.div(&ratio_to_bigfloat(&f0, wp), wp, RM);
    // lo^first·hi^(n − first).
    let direct_pow = |s: usize| lo.powi(s, wp, RM).mul(&hi.powi(deg - s, wp, RM), wp, RM);
    let mut pow = if hi_zero {
        BigFloat::new(wp)
    } else {
        direct_pow(first)
    };
    // A first power beyond the exponent range (`x` within `2^(−wp/n)`-ish of
    // −1): every power directly, as before 0.34.
    let direct = !hi_zero && !lo.is_zero() && (pow.is_zero() || pow.is_inf());
    let mut sum = BigFloat::new(wp);
    let mut largest: Option<i32> = None;
    for s in first..=deg {
        if s > first {
            pa = pa.div(&a.add(&int(s), wp, RM), wp, RM);
            pb = pb.mul(&b.add(&int(deg - s + 1), wp, RM), wp, RM);
            inv_fact = inv_fact.mul(&int(deg - s + 1), wp, RM).div(&int(s), wp, RM);
            pow = if direct {
                direct_pow(s)
            } else {
                pow.mul(&ratio, wp, RM)
            };
        }
        if hi_zero && s == deg {
            pow = lo.powi(deg, wp, RM);
        }
        let term = pa.mul(&pb, wp, RM).mul(&inv_fact, wp, RM).mul(&pow, wp, RM);
        if let Some(e) = term.exponent().filter(|_| !term.is_zero()) {
            largest = Some(largest.map_or(e, |l| l.max(e)));
        }
        sum = sum.add(&term, wp, RM);
    }
    Ok((sum, largest))
}

/// `n!` as a rational.
pub(super) fn factorial_bigint(n: usize) -> Ratio<BigInt> {
    let mut f = BigInt::from(1);
    for i in 2..=n {
        f *= BigInt::from(i as u64);
    }
    Ratio::from_integer(f)
}

/// Associated Legendre `P_n^m(x)` (Condon–Shortley phase) for integer
/// `n ≥ 0`, integer `m`, real `x` (`|x| ≤ 1` when `m` is odd).
pub(super) fn arb_assoc_legendre(
    n: &BigFloat,
    m: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    let deg = poly_degree(n, "assoc_legendre")? as i64;
    let order = bf_as_int(m)?
        .filter(|_| m.is_int())
        .ok_or_else(|| unevaluable("assoc_legendre: order must be an integer"))?;
    // 0 for `m > n` (a derivative of order `m` of a polynomial of degree
    // `n`); for `m < −n` the Ferrers function is not 0 (`P₀⁻¹(x) = √((1 −
    // x)/(1 + x))`; up to 0.56 it was 0 here).
    if order > deg {
        return Ok(BigFloat::new(prec));
    }
    if order < -deg {
        return Err(unevaluable(
            "assoc_legendre: an order below minus the degree is not a polynomial",
        ));
    }
    let wp = prec + 32 + 2 * (deg as f64).log2().ceil().max(0.0) as usize;
    let one = BigFloat::from_i32(1, wp);
    let mu = order.unsigned_abs() as usize;
    let x2 = x.mul(x, wp, RM);
    let one_minus_x2 = one.sub(&x2, wp, RM);
    // (1 − x²)^{μ/2}
    let root_pow = if mu.is_multiple_of(2) {
        one_minus_x2.powi(mu / 2, wp, RM)
    } else {
        if one_minus_x2.is_negative() {
            return Err(unevaluable(
                "assoc_legendre with odd order is complex for |x| > 1",
            ));
        }
        one_minus_x2
            .sqrt(wp, RM)
            .mul(&one_minus_x2.powi(mu / 2, wp, RM), wp, RM)
    };
    // P_μ^μ = (−1)^μ (2μ−1)!! (1−x²)^{μ/2}
    let mut dfact = BigInt::from(1);
    for j in 1..=mu {
        dfact *= BigInt::from((2 * j - 1) as u64);
    }
    if mu % 2 == 1 {
        dfact = -dfact;
    }
    let mut prev = ratio_to_bigfloat(&Ratio::from_integer(dfact), wp).mul(&root_pow, wp, RM);
    let deg_u = deg as usize;
    let mut curr = if deg_u > mu {
        BigFloat::from_i128(2 * mu as i128 + 1, wp)
            .mul(x, wp, RM)
            .mul(&prev, wp, RM)
    } else {
        prev.clone()
    };
    // The norm of `P_k^m` grows by at most `√((k + m + 1)/(k − m + 1))` per
    // step in the degree (`‖P_k^m‖² = 2·(k + m)!/((2k + 1)·(k − m)!)`); see
    // `arb_orthopoly`.
    let mut seen = Recurred::new(&[&prev, &curr]);
    for k in (mu + 1)..deg_u {
        let t1 = BigFloat::from_i128(2 * k as i128 + 1, wp)
            .mul(x, wp, RM)
            .mul(&curr, wp, RM);
        let t2 = BigFloat::from_i128((k + mu) as i128, wp).mul(&prev, wp, RM);
        let next = t1
            .sub(&t2, wp, RM)
            .div(&BigFloat::from_i128((k - mu + 1) as i128, wp), wp, RM);
        seen.step(
            0.5 * ((k + mu + 1) as f64 / (k - mu + 1) as f64).log2(),
            &next,
        );
        prev = curr;
        curr = next;
    }
    let mut own = seen.error(wp, odd_member_scale((deg_u - mu) % 2 == 1, deg_u as u64, x));
    if order < 0 {
        // P_n^{−μ} = (−1)^μ (n−μ)!/(n+μ)! P_n^μ
        let mut ratio = factorial_bigint(deg_u - mu) / factorial_bigint(deg_u + mu);
        if mu % 2 == 1 {
            ratio = -ratio;
        }
        let ratio = ratio_to_bigfloat(&ratio, wp);
        own = accuracy::shift(own, accuracy::part_lg(&ratio));
        curr = curr.mul(&ratio, wp, RM);
    }
    w.charge(own);
    Ok(round_to(curr, prec))
}

/// Generalised Laguerre `L_n^{(a)}(x)`:
/// `(k+1) L_{k+1} = (2k+1+a−x) L_k − (k+a) L_{k−1}`.
pub(super) fn arb_assoc_laguerre(
    n: &BigFloat,
    a: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    let deg = poly_degree(n, "assoc_laguerre")?;
    let wp = prec + 32 + (deg as f64).log2().ceil().max(0.0) as usize;
    let one = BigFloat::from_i32(1, wp);
    if deg == 0 {
        return Ok(BigFloat::from_i32(1, prec));
    }
    let mut prev = one.clone();
    let mut curr = one.add(a, wp, RM).sub(x, wp, RM);
    // The norm of `L_k^(a)` grows by `√((k + a + 1)/(k + 1))` per step
    // (`‖L_k^(a)‖² = Γ(k + a + 1)/k!`); see `arb_orthopoly`.
    let a_f = f64_of(a).unwrap_or(f64::NAN);
    let mut seen = Recurred::new(&[&prev, &curr]);
    for k in 1..deg {
        let k_bf = BigFloat::from_i128(k as i128, wp);
        let coeff = BigFloat::from_i128(2 * k as i128 + 1, wp)
            .add(a, wp, RM)
            .sub(x, wp, RM);
        let t1 = coeff.mul(&curr, wp, RM);
        let t2 = k_bf.add(a, wp, RM).mul(&prev, wp, RM);
        let next = t1
            .sub(&t2, wp, RM)
            .div(&BigFloat::from_i128(k as i128 + 1, wp), wp, RM);
        seen.step(
            0.5 * ((k as f64 + a_f + 1.0).abs() / (k as f64 + 1.0)).log2(),
            &next,
        );
        prev = curr;
        curr = next;
    }
    w.charge(seen.error(wp, 0.0));
    Ok(round_to(curr, prec))
}

/// The reason of the error [`get_cached`] raises for a sub-expression
/// with no value ([`root_cause`] replaces it by the sub-expression's own
/// error).
pub(super) fn not_evaluated_reason(id: ExprId) -> String {
    format!("sub-expression {id:?} has no numerical value")
}

pub(super) fn get_cached(cache: Cache<'_>, id: ExprId) -> Result<&Complex, SymplexError> {
    cache.get(&id).ok_or_else(|| SymplexError::Unevaluable {
        reason: not_evaluated_reason(id),
    })
}

/// Convert a `Ratio<BigInt>` to a `BigFloat` with the given precision in
/// bits (see [`crate::base::numeric::ratio_to_bigfloat`]).
pub(super) fn ratio_to_bigfloat(r: &Ratio<BigInt>, prec: usize) -> BigFloat {
    crate::base::numeric::ratio_to_bigfloat(r, prec, RM)
}

/// Try to extract a small integer exponent (fits in i32) from an ExprId.
pub(super) fn try_as_small_int(arena: &Arena, id: ExprId) -> Option<i32> {
    let r = arena.as_num(id)?;
    if !r.is_integer() {
        return None;
    }
    let n = r.to_integer();
    let val: i64 = n.try_into().ok()?;
    if val >= i32::MIN as i64 && val <= i32::MAX as i64 {
        Some(val as i32)
    } else {
        None
    }
}

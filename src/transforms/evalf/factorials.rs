//! Products and quotients of Γ at real arguments, for the functions `evalf`
//! had no kernel for (or refused at the poles of Γ): the rising and falling
//! factorials, the binomial coefficient at the poles, harmonic numbers of a
//! non-integer argument, and polygamma functions of large order.
//!
//! * `rf(x, n) = Γ(x + n)/Γ(x)`, `ff(x, n) = Γ(x + 1)/Γ(x − n + 1)` and
//!   `C(n, k) = Γ(n + 1)/(Γ(k + 1)·Γ(n − k + 1))` are quotients of Γ
//!   ([`gamma_ratio`], mpmath's `gammaprod`, BSD, see THIRD-PARTY-NOTICES.md):
//!   a pole of Γ more in the denominator makes the value 0, one more in the
//!   numerator infinite, and poles that pair up cancel through
//!   `lim Γ(−i + ε)/Γ(−j + ε) = (−1)^{i+j}·Γ(1 + j)/Γ(1 + i)` — so
//!   `C(−7, 2500) = (−1)^{2500}·C(2506, 6)`.  The quotient is taken in log
//!   space, `Σ ± ln|Γ|` with the signs apart, at enough bits for the
//!   logarithms' absolute error (a huge `n` makes `ln Γ(x + n)` large and
//!   the quotient moderate: `rf(1/3, 10⁹)`).
//! * `H(z) = ψ(z + 1) + γ` ([`harmonic`]), with the cancellation near
//!   `z = 0` measured and made up.
//! * `ψ⁽ⁿ⁾(x) = (−1)^{n+1}·n!·ζ(n + 1, x)` for `n > 10⁴` ([`polygamma_large`]):
//!   `ln n!` plus the logarithm of the Hurwitz zeta function, scaled by its
//!   largest term `d^{−s}` (`d` the distance from `x` to the nearest pole
//!   among the terms, `s = n + 1`), summed directly while the terms matter
//!   (`(d/(x + k))^s` falls off at once for a large `s`) with a rigorous
//!   bound on the rest, `u^{−s}·(1 + u/(s − 1))`, or with the
//!   Euler–Maclaurin tail once `x + k ≥ max(s/π, wp)`.  Left of 0 an odd
//!   `s` gives terms of both signs; they are summed in pairs `(f + j,
//!   1 − f + j)` (`f` the fractional part), which cancel exactly at a
//!   half-integer `x`.

use astro_float::{BigFloat, Consts, RoundingMode};

use crate::base::errors::SymplexError;

fn unevaluable(reason: impl Into<String>) -> SymplexError {
    SymplexError::Unevaluable {
        reason: reason.into(),
    }
}

/// `x > 0` strictly (astro-float's `is_positive` is true for `+0`).
fn pos(x: &BigFloat) -> bool {
    x.is_positive() && !x.is_zero()
}

/// Is `x` a pole of Γ, a non-positive integer?
fn is_npint(x: &BigFloat) -> bool {
    x.is_zero() || (x.is_int() && x.is_negative())
}

/// Is the integer `x` even?
fn is_even(x: &BigFloat) -> bool {
    if x.is_zero() {
        return true;
    }
    let mut half = x.clone();
    match x.exponent() {
        Some(e) => {
            half.set_exponent(e - 1);
            half.is_int()
        }
        None => true,
    }
}

/// The sign of `Γ(x)` at a real `x` off the poles: `+` right of 0,
/// `(−1)^{⌈−x⌉}` left of it (negative on `(−1, 0)`, positive on `(−2, −1)`).
fn gamma_sign(x: &BigFloat) -> i32 {
    if pos(x) || is_even(&x.floor()) { 1 } else { -1 }
}

/// `a + b` exactly (the precision spans both operands' bits), or to
/// `2¹⁸ + prec` bits when they are further apart than that.
pub(super) fn exact_sum(a: &BigFloat, b: &BigFloat, prec: usize, rm: RoundingMode) -> BigFloat {
    let span = |x: &BigFloat| -> Option<(i64, i64)> {
        let e = i64::from(x.exponent()?);
        let m = x.mantissa_max_bit_len().unwrap_or(64) as i64;
        (!x.is_zero()).then_some((e, e - m))
    };
    let p = match (span(a), span(b)) {
        (Some((ea, la)), Some((eb, lb))) => {
            let bits = ea.max(eb) + 2 - la.min(lb);
            usize::try_from(bits).unwrap_or(prec).min(prec + (1 << 18))
        }
        (Some((ea, la)), None) | (None, Some((ea, la))) => {
            usize::try_from(ea - la + 2).unwrap_or(prec)
        }
        (None, None) => prec,
    }
    .max(64);
    a.add(b, p, rm)
}

fn int(n: i32) -> BigFloat {
    BigFloat::from_i32(n, 64)
}

/// `Π Γ(aᵢ) / Π Γ(bⱼ)` at real arguments (mpmath's `gammaprod`): 0 when
/// the denominator has more poles than the numerator, infinite (an error)
/// when the numerator has more, and otherwise the limit, the poles paired
/// by `lim Γ(i + ε)/Γ(j + ε) = (−1)^{i+j}·Γ(1 − j)/Γ(1 − i)`.  Computed as
/// `±exp(Σ ln|Γ(aᵢ)| − Σ ln|Γ(bⱼ)|)` with the logarithms to an absolute
/// error below `2^{−prec−32}` (see the module documentation).
///
/// # Errors
///
/// [`SymplexError::Unevaluable`] for an infinite value, or one beyond the
/// floating-point exponent range.
pub(super) fn gamma_ratio(
    num: &[BigFloat],
    den: &[BigFloat],
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<BigFloat, SymplexError> {
    if num.iter().chain(den).any(|x| x.is_nan() || x.is_inf()) {
        return Err(unevaluable("Gamma quotient of a non-finite argument"));
    }
    let (poles_num, mut regular_num): (Vec<&BigFloat>, Vec<BigFloat>) = split_poles(num);
    let (poles_den, mut regular_den): (Vec<&BigFloat>, Vec<BigFloat>) = split_poles(den);
    if poles_num.len() < poles_den.len() {
        return Ok(BigFloat::new(prec));
    }
    if poles_num.len() > poles_den.len() {
        return Err(unevaluable(
            "the Gamma quotient has a pole here (more poles of Gamma above than below): the value is infinite",
        ));
    }
    let mut sign = 1;
    for (i, j) in poles_num.into_iter().zip(poles_den) {
        if !is_even(&exact_sum(i, j, prec, rm)) {
            sign = -sign;
        }
        regular_num.push(exact_sum(&int(1), &j.neg(), prec, rm));
        regular_den.push(exact_sum(&int(1), &i.neg(), prec, rm));
    }
    for x in regular_num.iter().chain(&regular_den) {
        sign *= gamma_sign(x);
    }
    // A first pass for the logarithms' size, a second with their bits.
    let mut wp = prec + 64;
    loop {
        let mut total = BigFloat::new(wp);
        let mut size = 0.0f64;
        for (x, plus) in regular_num
            .iter()
            .map(|x| (x, true))
            .chain(regular_den.iter().map(|x| (x, false)))
        {
            let l = super::arb_log_gamma(x, wp, rm, cc)?.0;
            size += super::bigfloat_to_f64_rounded(&l, rm)
                .unwrap_or(f64::INFINITY)
                .abs();
            total = if plus {
                total.add(&l, wp, rm)
            } else {
                total.sub(&l, wp, rm)
            };
        }
        let needed = prec + 32 + (size + 1.0).log2().ceil().max(0.0) as usize;
        if needed <= wp || !size.is_finite() {
            let v = total.exp(wp, rm, cc);
            if v.is_inf() || v.is_nan() || (v.is_zero() && !total.is_zero()) {
                return Err(unevaluable(
                    "the Gamma quotient is beyond the floating-point exponent range",
                ));
            }
            let v = super::round_to(v, prec, rm);
            return Ok(if sign < 0 { v.neg() } else { v });
        }
        wp = needed;
    }
}

fn split_poles(xs: &[BigFloat]) -> (Vec<&BigFloat>, Vec<BigFloat>) {
    let mut poles = Vec::new();
    let mut regular = Vec::new();
    for x in xs {
        if is_npint(x) {
            poles.push(x);
        } else {
            regular.push(x.clone());
        }
    }
    (poles, regular)
}

/// `rf(x, n) = Γ(x + n)/Γ(x)` (the Pochhammer symbol) for real `x`, `n`.
pub(super) fn rising_factorial(
    x: &BigFloat,
    n: &BigFloat,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<BigFloat, SymplexError> {
    if n.is_zero() {
        return Ok(BigFloat::from_i32(1, prec));
    }
    let xn = exact_sum(x, n, prec, rm);
    gamma_ratio(&[xn], std::slice::from_ref(x), prec, rm, cc)
}

/// `ff(x, n) = Γ(x + 1)/Γ(x − n + 1)` for real `x`, `n`.
pub(super) fn falling_factorial(
    x: &BigFloat,
    n: &BigFloat,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<BigFloat, SymplexError> {
    if n.is_zero() {
        return Ok(BigFloat::from_i32(1, prec));
    }
    let x1 = exact_sum(x, &int(1), prec, rm);
    let xn1 = exact_sum(&x1, &n.neg(), prec, rm);
    gamma_ratio(&[x1], &[xn1], prec, rm, cc)
}

/// `C(n, k) = Γ(n + 1)/(Γ(k + 1)·Γ(n − k + 1))` through [`gamma_ratio`]:
/// the limit where poles of Γ meet (`C(−7, 2500) = C(2506, 6)`, `C(n, k) = 0`
/// for a negative integer `k` and a non-integer `n`).
pub(super) fn binomial(
    n: &BigFloat,
    k: &BigFloat,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<BigFloat, SymplexError> {
    let n1 = exact_sum(n, &int(1), prec, rm);
    let k1 = exact_sum(k, &int(1), prec, rm);
    let nk1 = exact_sum(&n1, &k.neg(), prec, rm);
    gamma_ratio(&[n1], &[k1, nk1], prec, rm, cc)
}

/// Does `C(n, k)`'s Γ formula meet a pole: is `n + 1`, `k + 1` or
/// `n − k + 1` a non-positive integer?
pub(super) fn binomial_meets_pole(
    n: &BigFloat,
    k: &BigFloat,
    prec: usize,
    rm: RoundingMode,
) -> bool {
    let n1 = exact_sum(n, &int(1), prec, rm);
    let k1 = exact_sum(k, &int(1), prec, rm);
    let nk1 = exact_sum(&n1, &k.neg(), prec, rm);
    is_npint(&n1) || is_npint(&k1) || is_npint(&nk1)
}

/// `H(z) = ψ(z + 1) + γ` for real `z` (poles at the negative integers).
/// The terms cancel near `z = 0` (`H(z) ≈ ζ(2)·z`): the loss is measured
/// and the sum repeated with that many more bits, up to the cancellation
/// cap of the other special functions.
pub(super) fn harmonic(
    z: &BigFloat,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<BigFloat, SymplexError> {
    if z.is_nan() || z.is_inf() {
        return Err(unevaluable("harmonic of a non-finite argument"));
    }
    if z.is_zero() {
        return Ok(BigFloat::new(prec));
    }
    if z.is_int() && z.is_negative() {
        return Err(unevaluable(
            "harmonic has a pole at a negative integer (the value is infinite)",
        ));
    }
    let z1 = exact_sum(z, &int(1), prec, rm);
    let cap = super::cancellation_cap(prec);
    let mut extra = 32;
    loop {
        let wp = prec + extra;
        let psi = super::arb_digamma(&z1, wp, rm, cc)?;
        let gamma = super::arb_euler_gamma(wp, rm, cc)?;
        let h = psi.add(&gamma, wp, rm);
        let top = psi
            .exponent()
            .unwrap_or(0)
            .max(gamma.exponent().unwrap_or(0));
        let lost = match h.exponent() {
            Some(e) if !h.is_zero() => usize::try_from(i64::from(top) - i64::from(e)).unwrap_or(0),
            _ => wp,
        };
        if lost + 16 <= extra {
            return Ok(super::round_to(h, prec, rm));
        }
        if extra >= cap {
            return Err(super::special_exhausted(prec));
        }
        extra = (lost + 32).max(2 * extra).min(cap);
    }
}

/// The largest order `polygamma_large` accepts (`s = n + 1` stays exact in
/// an `f64`).
const MAX_ORDER: f64 = 9.0e15;

/// `ψ⁽ⁿ⁾(x)` for an integer order `n` (beyond the recurrence and
/// asymptotic series of `arb_polygamma`, which handles `n ≤ 10⁴`) and a real
/// `x` off the poles: `(−1)^{n+1}·exp(ln Γ(n + 1) − s·ln d + ln Z)·sign Z`,
/// `ζ(s, x) = d^{−s}·Z` (see the module documentation).
///
/// # Errors
///
/// [`SymplexError::Unevaluable`] at a pole or beyond the floating-point
/// exponent range; [`SymplexError::PrecisionExhausted`] when the pairs of a
/// negative `x` cancel beyond the cap.
pub(super) fn polygamma_large(
    n: &BigFloat,
    x: &BigFloat,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<BigFloat, SymplexError> {
    if x.is_nan() || x.is_inf() {
        return Err(unevaluable("polygamma of a non-finite argument"));
    }
    if is_npint(x) {
        return Err(unevaluable("polygamma at non-positive integer pole"));
    }
    let s = exact_sum(n, &int(1), prec, rm);
    let sf = super::bigfloat_to_f64_rounded(&s, rm)?;
    if !(2.0..=MAX_ORDER).contains(&sf) {
        return Err(SymplexError::NotImplemented(format!(
            "polygamma of order {} (at most {MAX_ORDER:e})",
            sf - 1.0
        )));
    }
    let odd = !is_even(&s);
    let cap = super::cancellation_cap(prec);
    let mut extra = 32;
    loop {
        let wp = prec + extra;
        let (scale, z) = scaled_hurwitz(&s, sf, odd, x, wp, rm, cc)?;
        // `Z` is a sum of terms of magnitude at most 1: its loss is `−log₂|Z|`.
        let lost = match z.exponent() {
            Some(e) if !z.is_zero() => usize::try_from(-i64::from(e)).unwrap_or(0),
            _ => wp,
        };
        if lost + 16 > extra {
            if extra >= cap {
                return Err(super::special_exhausted(prec));
            }
            extra = (lost + 32).max(2 * extra).min(cap);
            continue;
        }
        // ln|ψ| = ln Γ(s) − s·ln d + ln|Z|, to an absolute error below 2^{−prec−32}.
        let lg = super::arb_log_gamma(&s, wp, rm, cc)?.0;
        let ln_d = scale.ln(wp, rm, cc);
        let size = super::bigfloat_to_f64_rounded(&lg, rm)?.abs()
            + sf * super::bigfloat_to_f64_rounded(&ln_d, rm)?.abs();
        let lp = prec + 32 + (size + 1.0).log2().ceil().max(0.0) as usize;
        let lg = super::arb_log_gamma(&s, lp, rm, cc)?.0;
        let s_ln_d = s.mul(&scale.ln(lp, rm, cc), lp, rm);
        let ln_z = z.abs().ln(lp, rm, cc);
        let total = lg.sub(&s_ln_d, lp, rm).add(&ln_z, lp, rm);
        let v = total.exp(lp, rm, cc);
        if v.is_inf() || v.is_nan() || v.is_zero() {
            return Err(unevaluable(
                "polygamma of this order is beyond the floating-point exponent range",
            ));
        }
        // (−1)^{n+1} = (−1)^s.
        let negative = odd != z.is_negative();
        let v = super::round_to(v, prec, rm);
        return Ok(if negative { v.neg() } else { v });
    }
}

/// `ζ(s, x) = d^{−s}·Z`: the scale `d` (the smallest `|x + k|` contributing)
/// and `Z` at `wp` bits, whose largest term is 1.
fn scaled_hurwitz(
    s: &BigFloat,
    sf: f64,
    odd: bool,
    x: &BigFloat,
    wp: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<(BigFloat, BigFloat), SymplexError> {
    if pos(x) {
        return Ok((x.clone(), positive_part(s, sf, x, x, wp, rm, cc)?));
    }
    // x = f − m, f ∈ (0, 1), m ≥ 1: the terms left of 0 are (x + j)^{−s} =
    // (−1)^s·(i − f)^{−s}, i = m − j = 1 … m.
    let m = x.floor().neg();
    let f = exact_sum(x, &m, wp, rm);
    let one_minus_f = exact_sum(&int(1), &f.neg(), wp, rm);
    let mf = super::bigfloat_to_f64_rounded(&m, rm)?;
    if odd && f == one_minus_f {
        // A half-integer x: the pairs (f + j, 1 − f + j), j < m, cancel exactly.
        let start = exact_sum(&f, &m, wp, rm);
        return Ok((
            start.clone(),
            positive_part(s, sf, &start, &start, wp, rm, cc)?,
        ));
    }
    let g = if f < one_minus_f {
        f.clone()
    } else {
        one_minus_f.clone()
    };
    let negligible = |t: &BigFloat, sum: &BigFloat| -> bool {
        match (t.exponent(), sum.exponent()) {
            (_, _) if t.is_zero() => true,
            (Some(te), Some(se)) if !sum.is_zero() => {
                i64::from(se) - i64::from(te) > wp as i64 + 32
            }
            _ => false,
        }
    };
    let mut z = BigFloat::new(wp);
    let mut j = 0.0f64;
    let mut u = f.clone(); // f + j
    let mut v = one_minus_f.clone(); // 1 − f + j
    while j < mf {
        let a = scaled_term(s, &u, &g, wp, rm, cc);
        let b = scaled_term(s, &v, &g, wp, rm, cc);
        let pair = if odd {
            a.sub(&b, wp, rm)
        } else {
            a.add(&b, wp, rm)
        };
        z = z.add(&pair, wp, rm);
        if negligible(&a, &z) && negligible(&b, &z) {
            // The rest of both sums: below `(1 + u/(s − 1))` times these
            // terms each (the bound of `positive_part`), under `2^{−wp−8}`
            // of the sum for `|x| ≤ 10⁶`, `s > 10⁴`.
            return Ok((g, z));
        }
        j += 1.0;
        u = exact_sum(&u, &int(1), wp, rm);
        v = exact_sum(&v, &int(1), wp, rm);
        if j > 1.0e6 {
            return Err(SymplexError::NotImplemented(
                "polygamma of large order far left of 0 (more than a million terms)".into(),
            ));
        }
    }
    // The terms right of the last pair: f + m, f + m + 1, …
    let rest = positive_part(s, sf, &u, &g, wp, rm, cc)?;
    Ok((g, z.add(&rest, wp, rm)))
}

/// `(g/u)^s = exp(−s·ln(u/g))` at `wp` bits (`u ≥ g > 0`).
fn scaled_term(
    s: &BigFloat,
    u: &BigFloat,
    g: &BigFloat,
    wp: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> BigFloat {
    // `s·ln(u/g)` to an absolute `2^{−wp−8}` where the term matters
    // (`s·ln(u/g) ≲ wp`): `log₂ s` bits for the product, `2·log₂ wp` more.
    let s_bits = s.exponent().map_or(0, |e| usize::try_from(e).unwrap_or(0));
    let p = wp + s_bits + 2 * (usize::BITS - wp.leading_zeros()) as usize + 16;
    let l = u.div(g, p, rm).ln(p, rm, cc);
    s.mul(&l, p, rm).neg().exp(wp, rm, cc)
}

/// `Σ_{j≥0} (g/(f + j))^s` for `f ≥ g > 0` and a large `s`: the terms while
/// they matter, with the rigorous bound `(g/u)^s·(1 + u/(s − 1))` on the sum
/// from `u` on, or the Euler–Maclaurin tail once `f + j ≥ max(s/π, wp)`.
fn positive_part(
    s: &BigFloat,
    sf: f64,
    f: &BigFloat,
    g: &BigFloat,
    wp: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<BigFloat, SymplexError> {
    let switch = (sf / std::f64::consts::PI).max(wp as f64) + 1.0;
    let mut sum = BigFloat::new(wp);
    let mut u = f.clone();
    for _ in 0..1_000_000 {
        let uf = super::bigfloat_to_f64_rounded(&u, rm)?;
        if uf >= switch {
            let tail = euler_maclaurin_tail(s, &u, g, wp, rm, cc);
            return Ok(sum.add(&tail, wp, rm));
        }
        let t = scaled_term(s, &u, g, wp, rm, cc);
        sum = sum.add(&t, wp, rm);
        let u1 = exact_sum(&u, &int(1), wp, rm);
        let next = scaled_term(s, &u1, g, wp, rm, cc);
        // Σ_{i≥0} (u1 + i)^{−s} ≤ u1^{−s}·(1 + u1/(s − 1)).
        let factor = 1.0 + (uf + 1.0) / (sf - 1.0);
        let bound_exp =
            next.exponent().map_or(i64::MIN / 2, i64::from) + factor.log2().ceil() as i64 + 1;
        let sum_exp = sum.exponent().map_or(i64::MIN / 2, i64::from);
        if next.is_zero() || sum_exp - bound_exp > wp as i64 + 8 {
            return Ok(sum);
        }
        u = u1;
    }
    Err(SymplexError::NotImplemented(
        "polygamma of large order: the Hurwitz zeta sum did not settle in a million terms".into(),
    ))
}

/// `Σ_{i≥0} (g/(a + i))^s` by Euler–Maclaurin at `a ≥ max(s/π, wp)`:
/// `(g/a)^s·[a/(s − 1) + 1/2 + Σ_{j≥1} B₂ⱼ/(2j)!·(s)₂ⱼ₋₁/a^{2j−1}]`, the
/// series stopped once a term is below `2^{−wp−8}` of the bracket (for the
/// completely monotone `t^{−s}` the remainder is below the first omitted
/// term).
fn euler_maclaurin_tail(
    s: &BigFloat,
    a: &BigFloat,
    g: &BigFloat,
    wp: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> BigFloat {
    let one = BigFloat::from_i32(1, wp);
    let s_minus_1 = s.sub(&one, wp, rm);
    let mut bracket = a
        .div(&s_minus_1, wp, rm)
        .add(&BigFloat::from_f64(0.5, wp), wp, rm);
    let a2 = a.mul(a, wp, rm);
    // (s)_{2j−1}/((2j)!·a^{2j−1}), j = 1: s/(2a).
    let mut coef = s.div(&a.mul(&BigFloat::from_i32(2, wp), wp, rm), wp, rm);
    let mut prev: Option<i64> = None;
    for j in 1usize..=(wp / 2 + 64) {
        if j > 1 {
            // × (s + 2j − 3)(s + 2j − 2)/((2j − 1)(2j)·a²)
            let jj = j as i64;
            let p1 = s.add(&BigFloat::from_i64(2 * jj - 3, 64), wp, rm);
            let p2 = s.add(&BigFloat::from_i64(2 * jj - 2, 64), wp, rm);
            let q = BigFloat::from_i64((2 * jj - 1) * (2 * jj), 64);
            coef = coef
                .mul(&p1, wp, rm)
                .mul(&p2, wp, rm)
                .div(&q, wp, rm)
                .div(&a2, wp, rm);
        }
        let b = super::ratio_to_bigfloat(&super::bernoulli::even(j), wp, rm);
        let term = b.mul(&coef, wp, rm);
        let te = term.exponent().map_or(i64::MIN / 2, i64::from);
        if prev.is_some_and(|p| te > p) {
            break; // the asymptotic series turned (not reached for a ≥ s/π)
        }
        prev = Some(te);
        bracket = bracket.add(&term, wp, rm);
        let be = bracket.exponent().map_or(0, i64::from);
        if term.is_zero() || be - te > wp as i64 + 8 {
            break;
        }
    }
    scaled_term(s, a, g, wp, rm, cc).mul(&bracket, wp, rm)
}

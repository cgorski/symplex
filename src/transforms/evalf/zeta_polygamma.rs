//! Riemann zeta function (part of [`evalf`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Riemann zeta function
// ═══════════════════════════════════════════════════════════════════════════

/// `ζ(s)` for real `s ≠ 1`.
///
/// * `s > 0`: Borwein's algorithm (P. Borwein, *An efficient algorithm for
///   the Riemann zeta function*, 1991, Algorithm 2) — an accelerated
///   alternating series with error `≤ 3/(3+√8)ⁿ`, so `n ≈ 0.39·wp` terms.
/// * `s < 0`: functional equation
///   `ζ(s) = 2ˢ π^{s−1} sin(πs/2) Γ(1−s) ζ(1−s)`; trivial zeros at the
///   negative even integers are returned exactly.
/// * `s = 0`: `−1/2`.
pub(super) fn arb_zeta(s: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    if s.is_nan() || s.is_inf_neg() {
        return Err(SymplexError::Unevaluable {
            reason: "zeta of special float value".into(),
        });
    }
    if s.is_inf_pos() {
        return Ok(BigFloat::from_i32(1, prec));
    }
    if s.is_zero() {
        return Ok(BigFloat::from_f64(-0.5, prec));
    }
    let s_f64 = f64_of(s)?;
    // The pole exactly (before 0.30 in `f64`, where `1 + 10⁻⁴⁵` is 1).
    if *s == BigFloat::from_i32(1, 64) {
        return Err(infinite("zeta(1) is a pole"));
    }
    let mut wp = prec + 32;

    if s.is_negative() {
        // `2^s`, `π^(s−1)` and `Γ(1 − s)` come from exponentials of
        // arguments of about `|s|·ln|s|`, whose absolute error is their
        // relative one: as many guard bits.
        wp += magnitude_bits(s_f64.abs() * (s_f64.abs().ln().max(0.0) + 2.0));
        // Trivial zeros (the `f64` test only filters; the exact difference
        // below decides).
        let s_round = s_f64.round();
        if (s_f64 - s_round).abs() < 1e-14 && (s_round as i64) % 2 == 0 {
            let mut sw = s.clone();
            let _ = sw.set_precision(wp, RM);
            let diff = sw.sub(&bigfloat::from_f64(s_round, wp), wp, RM);
            if diff.is_zero() {
                return Ok(BigFloat::new(prec));
            }
        }
        // An `s` whose unit in the last place at the working precision is
        // 2 or more has no known parity: `sin(πs/2)` is undetermined (and
        // `ζ(s)` is a trivial zero or far beyond the exponent range).  The
        // argument reduction of that `sin` for `s ≈ −2^(1.4·10⁶)` ran for
        // minutes (`zeta(erfi(−1000))`, found by fuzz_evalf in 0.31).
        if s.exponent().is_some_and(|e| i64::from(e) > wp as i64) {
            return Err(special_exhausted(prec));
        }
        // ζ(s) = 2^s π^{s−1} sin(π s/2) Γ(1−s) ζ(1−s), `1 − s` exact: next
        // to 0 the factors `sin(πs/2) ≈ πs/2` and `ζ(1 − s) ≈ −1/s` are both
        // relative to `s` (before 0.30 `1 − s` was rounded, and `ζ(−10⁻⁶⁰)`
        // printed `−0.499999999999999999141761477346` at 30 digits).
        let mut sw = s.clone();
        let _ = sw.set_precision(wp, RM);
        let one = BigFloat::from_i32(1, wp);
        let two = BigFloat::from_i32(2, wp);
        let pi = w.cc.pi(wp, RM).clone();
        let one_minus_s = BigFloat::from_i32(1, 64).sub(s, exact_bits(s, wp), RM);
        let two_pow_s = bf_pow(&two, &sw, &mut w.at(wp));
        let s_minus_1 = sw.sub(&one, wp, RM);
        let pi_pow = bf_pow(&pi, &s_minus_1, &mut w.at(wp));
        // sin(πs/2) = (−1)ⁿ·sin(πd) for s/2 = n + d, `d` exact: next to a
        // trivial zero `sin` of a rounded `πs/2` lost `log₂(1/|d|)` bits
        // (up to 0.43 `zeta(−2 − 2⁻¹⁵⁰)` at 40 digits was wrong from its
        // 33rd digit, `zeta(−70 − 2⁻¹³⁷)` from its 35th).
        let sin_term = sin_half_pi(s, &mut w.at(wp));
        let gamma_term = arb_gamma_real(&one_minus_s, &mut w.at(wp))?;
        let zeta_term = arb_zeta_borwein(&one_minus_s, &mut w.at(wp))?;
        let r = two_pow_s
            .mul(&pi_pow, wp, RM)
            .mul(&sin_term, wp, RM)
            .mul(&gamma_term, wp, RM)
            .mul(&zeta_term, wp, RM);
        return Ok(round_to(r, prec));
    }

    let r = arb_zeta_borwein(s, &mut w.at(wp))?;
    Ok(round_to(r, prec))
}

/// A rational literal `q < 0` within `2⁻⁹` of a trivial zero `−2n` of `ζ`
/// (not on it): `n`'s parity and `d = q/2 + n`, exactly.
pub(super) struct TrivialZeroOffset {
    odd: bool,
    d: crate::base::numeric::Q,
}

/// [`TrivialZeroOffset`] of the node `c`, when it is such a literal.
pub(super) fn trivial_zero_offset(arena: &Arena, c: ExprId) -> Option<TrivialZeroOffset> {
    use num_traits::Signed;
    let q = arena.as_num(c)?;
    if !q.is_negative() {
        return None;
    }
    let half = q / crate::base::numeric::Q::from_integer(BigInt::from(2));
    let n = half.round();
    let d = &half - &n;
    // `abs(d) < 2⁻¹⁰`, from the bit lengths (`abs(d) < 2^(bits(p) − bits(q) + 1)`).
    if d.is_zero() || n.is_zero() || d.numer().bits() + 11 > d.denom().bits() {
        return None;
    }
    let odd = !(n.to_integer() % BigInt::from(2)).is_zero();
    Some(TrivialZeroOffset { odd, d })
}

/// `ζ(q)` for a rational literal `q` next to a trivial zero ([`TrivialZeroOffset`];
/// `s`: `q` as a float), to `prec` bits relative: the functional equation of
/// [`arb_zeta`] with `sin(πq/2) = (−1)ⁿ·sin(πd)` from the exact `d`.  The other
/// factors move by `2^−wp` relative when `q` is rounded; the value itself,
/// `≈ ζ′(−2n)·2d`, is known to that only from `d`.  Up to 0.58 the float
/// argument carried its rounding as a radius, `abs(ζ′)·2^−prec` — far above a
/// value of `10⁻³⁰²` —, and `zeta(−2 + 10⁻³⁰⁰)` took four evaluations, up to
/// 1,152 bits, at 16 digits (0.09 s; mpmath: `-3.044845705839327e-302`).
pub(super) fn zeta_next_to_trivial_zero(
    s: &BigFloat,
    offset: &TrivialZeroOffset,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    let s_f64 = f64_of(s)?;
    let wp = prec + 32 + magnitude_bits(s_f64.abs() * (s_f64.abs().ln().max(0.0) + 2.0));
    let mut sw = s.clone();
    let _ = sw.set_precision(wp, RM);
    let one = BigFloat::from_i32(1, wp);
    let two = BigFloat::from_i32(2, wp);
    let pi = w.cc.pi(wp + 8, RM).clone();
    let one_minus_s = BigFloat::from_i32(1, 64).sub(s, exact_bits(s, wp), RM);
    let two_pow_s = bf_pow(&two, &sw, &mut w.at(wp));
    let pi_pow = bf_pow(&pi, &sw.sub(&one, wp, RM), &mut w.at(wp));
    let d = ratio_to_bigfloat(&offset.d, wp + 8);
    let sin = pi.mul(&d, wp + 8, RM).sin(wp + 8, RM, w.cc);
    let sin_term = if offset.odd { sin.neg() } else { sin };
    let gamma_term = arb_gamma_real(&one_minus_s, &mut w.at(wp))?;
    let zeta_term = arb_zeta_borwein(&one_minus_s, &mut w.at(wp))?;
    let r = two_pow_s
        .mul(&pi_pow, wp, RM)
        .mul(&sin_term, wp, RM)
        .mul(&gamma_term, wp, RM)
        .mul(&zeta_term, wp, RM);
    Ok(round_to(r, prec))
}

/// `sin(πs/2)` to `wp` bits relative for a finite real `s`: with `s/2 = n + d`,
/// `n` the nearest integer and `d` formed exactly, `(−1)ⁿ·sin(πd)` — exactly 0
/// at the even integers, and relative to `d` next to them.
pub(super) fn sin_half_pi(s: &BigFloat, w: &mut Wp<'_>) -> BigFloat {
    let wp = w.bits();
    let exact = exact_bits(s, wp);
    let half = s.div(&BigFloat::from_i32(2, 64), exact, RM);
    let n = half.add(&BigFloat::from_f64(0.5, 64), exact, RM).floor();
    let d = half.sub(&n, exact, RM);
    if d.is_zero() {
        return BigFloat::new(wp);
    }
    let pi = w.cc.pi(wp + 8, RM).clone();
    let v = pi.mul(&d, wp + 8, RM).sin(wp + 8, RM, w.cc);
    let odd = !n
        .div(&BigFloat::from_i32(2, 64), exact, RM)
        .fract()
        .is_zero();
    round_to(if odd { v.neg() } else { v }, wp)
}

/// Borwein's Algorithm 2 for `ζ(s)`, real `s > 0`, `s ≠ 1`:
/// `ζ(s) = η(s) / (1 − 2^{1−s})` with `η` from [`arb_eta_borwein`].
fn arb_zeta_borwein(s: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let wp = w.bits();
    let eta = arb_eta_borwein(s, w)?;
    // 1 − 2^{1−s} = −expm1((1 − s)·ln 2), with `1 − s` exact: next to the
    // pole the difference `1 − 2^{1−s}` cancels `log₂(1/|1 − s|)` bits, and
    // before 0.30 both were rounded (`ζ(1 + 10⁻⁴⁵)` was "a pole").
    let one_minus_s = BigFloat::from_i32(1, 64).sub(s, exact_bits(s, wp), RM);
    let denom_factor = one_minus_pow2(&one_minus_s, w);
    if denom_factor.is_zero() {
        return Err(infinite("zeta(1) is a pole"));
    }
    Ok(eta.div(&denom_factor, wp, RM))
}

/// `1 − 2^t` to `wp` bits relative, also for a tiny `t` (where the
/// difference cancels): `−Σ_{k≥1} u^k/k!` with `u = t·ln 2` for `|u| < 1/4`.
pub(super) fn one_minus_pow2(t: &BigFloat, w: &mut Wp<'_>) -> BigFloat {
    let wp = w.bits();
    if t.is_zero() {
        return BigFloat::new(wp);
    }
    let u = w.cc.ln_2(wp + 8, RM).mul(t, wp + 8, RM);
    if u.exponent().is_some_and(|e| e > -2) {
        return BigFloat::from_i32(1, wp).sub(&u.exp(wp + 8, RM, w.cc), wp, RM);
    }
    let mut term = u.clone();
    let mut sum = u.clone();
    for k in 2..(wp + 16) {
        term = term
            .mul(&u, wp + 8, RM)
            .div(&BigFloat::from_i128(k as i128, wp + 8), wp + 8, RM);
        sum = sum.add(&term, wp + 8, RM);
        if negligible(&term, &sum, wp + 8) {
            break;
        }
    }
    round_to(sum.neg(), wp)
}

/// A table of [`borwein_d`], shared with its memo.
type BorweinTable = std::rc::Rc<Vec<Ratio<BigInt>>>;

/// A table of [`borwein_d`] kept for its `n`.
struct BorweinMemo {
    n: usize,
    d: BorweinTable,
}

thread_local! {
    /// The last tables of [`borwein_d`] (a pure function of `n`; a few
    /// precisions are in use at a time).
    static BORWEIN_D: std::cell::RefCell<Vec<BorweinMemo>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Borwein's coefficients `d_0, …, d_n` ([`arb_eta_borwein`]): `d_k = Σ_{i≤k}
/// t_i`, `t_i = n·(n+i−1)!·4ⁱ/((n−i)!·(2i)!)` — integers (`t_i = n/(n+i)·C(n+i,
/// 2i)·4ⁱ`, the coefficients of the Chebyshev polynomial `T_n` up to sign),
/// formed by `t_i = t_{i−1}·4(n+i−1)(n−i+1)/((2i−1)·2i)` in integer
/// arithmetic, and kept for the next call.  Up to 0.58 every `ζ` and `η`
/// rebuilt them as rationals, reduced by a gcd of `wp`-bit integers at each
/// of the `3n` steps: 45 % of the time of `polylog(−3/2, −1 + 10⁻⁷⁰)` at 60
/// digits.
pub(super) fn borwein_d(n: usize) -> BorweinTable {
    if let Some(d) = BORWEIN_D.with(|memo| {
        memo.borrow()
            .iter()
            .find(|m| m.n == n)
            .map(|m| std::rc::Rc::clone(&m.d))
    }) {
        return d;
    }
    let mut d: Vec<Ratio<BigInt>> = Vec::with_capacity(n + 1);
    let mut acc = BigInt::zero();
    // t_0 = n·(n−1)!/n! = 1.
    let mut term = BigInt::from(1);
    for i in 0..=n {
        if i > 0 {
            let numer = BigInt::from(4u64 * (n + i - 1) as u64) * BigInt::from((n - i + 1) as u64);
            let denom = BigInt::from((2 * i - 1) as u64 * (2 * i) as u64);
            // The quotient is exact (`t_i` is an integer).
            term = term * numer / denom;
        }
        acc += &term;
        d.push(Ratio::from_integer(acc.clone()));
    }
    let d = std::rc::Rc::new(d);
    BORWEIN_D.with(|memo| {
        let mut memo = memo.borrow_mut();
        if memo.len() >= 8 {
            memo.remove(0);
        }
        memo.push(BorweinMemo {
            n,
            d: std::rc::Rc::clone(&d),
        });
    });
    d
}

/// Borwein's accelerated alternating sum for the Dirichlet eta function
/// `η(s) = Σ_{k≥0} (−1)ᵏ/(k+1)ˢ`, real `s > 0` (P. Borwein, *An efficient
/// algorithm for the Riemann zeta function*, 1991, Algorithm 2; error
/// `≤ 3/(3+√8)ⁿ`, so `n ≈ 0.39·wp` terms):
///
/// ```text
/// d_k = n Σ_{i=0}^{k} (n+i−1)! 4ⁱ / ((n−i)! (2i)!)
/// η(s) = − 1/d_n · Σ_{k=0}^{n−1} (−1)ᵏ (d_k − d_n)/(k+1)ˢ
/// ```
pub(super) fn arb_eta_borwein(s: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let wp = w.bits();
    // n such that (3+√8)^{-n} < 2^{-wp}:  n > wp·ln2/ln(3+√8) ≈ 0.393·wp
    let n = ((wp as f64) * std::f64::consts::LN_2 / (3.0 + 8f64.sqrt()).ln()).ceil() as usize + 4;

    // Exact integer coefficients d_k.
    let d = borwein_d(n);
    let d_n = ratio_to_bigfloat(&d[n], wp);

    // S = Σ_{k=0}^{n−1} (−1)^k (d_k − d_n) / (k+1)^s
    let mut sum = BigFloat::new(wp);
    let s_is_int = s.is_int();
    let s_int = if s_is_int {
        f64_of(s)?.round() as usize
    } else {
        0
    };
    // A non-integer power `m^s` costs a logarithm and an exponential, and
    // there are `0.39·wp` of them; `m^s = p^s·(m/p)^s` for the smallest
    // prime factor `p` of `m` leaves them to the primes (a seventh of
    // them at 4,000 bits, where `ζ` took seconds), each composite a product
    // of at most `log₂ m` roundings.
    let powers: Vec<BigFloat> = if s_is_int {
        Vec::new()
    } else {
        let mut spf = vec![0usize; n + 1];
        let mut table: Vec<BigFloat> = Vec::with_capacity(n + 1);
        table.push(BigFloat::new(wp));
        for m in 1..=n {
            if m >= 2 && spf[m] == 0 {
                for multiple in (m..=n).step_by(m) {
                    if spf[multiple] == 0 {
                        spf[multiple] = m;
                    }
                }
            }
            let v = if m == 1 {
                BigFloat::from_i32(1, wp)
            } else if spf[m] == m {
                bf_pow(&BigFloat::from_i128(m as i128, wp), s, &mut w.at(wp + 8))
            } else {
                table[spf[m]].mul(&table[m / spf[m]], wp + 8, RM)
            };
            table.push(v);
        }
        table
    };
    for k in 0..n {
        let coeff = ratio_to_bigfloat(&(&d[k] - &d[n]), wp);
        let kp1 = BigFloat::from_i128(k as i128 + 1, wp);
        let kp1_pow_s = if s_is_int {
            kp1.powi(s_int, wp, RM)
        } else {
            powers[k + 1].clone()
        };
        let term = coeff.div(&kp1_pow_s, wp, RM);
        if k % 2 == 0 {
            sum = sum.add(&term, wp, RM);
        } else {
            sum = sum.sub(&term, wp, RM);
        }
    }

    Ok(sum.div(&d_n, wp, RM).neg())
}

// ═══════════════════════════════════════════════════════════════════════════
// Polygamma
// ═══════════════════════════════════════════════════════════════════════════

/// `ψ⁽ⁿ⁾(x)` for integer `n ≥ 1` and real `x` (not a non-positive integer).
///
/// 1. Recurrence `ψ⁽ⁿ⁾(x) = ψ⁽ⁿ⁾(x+N) − (−1)ⁿ n! Σ_{k<N} 1/(x+k)^{n+1}`
///    shifts the argument to `x + N ≥ max(wp/3, n) + 10`.
/// 2. Asymptotic series (DLMF 5.15.8):
///    `ψ⁽ⁿ⁾(x) ~ (−1)^{n+1} [ (n−1)!/xⁿ + n!/(2x^{n+1}) + Σ_{k≥1} B₂ₖ (2k+n−1)!/((2k)! x^{2k+n}) ]`
///    with exact Bernoulli numbers, truncated when the terms fall below
///    the working precision (or start to diverge).
///
/// The shifted arguments `x + k` are exact, so a pole is only an integer
/// `x ≤ 0`; left of 0 the terms change sign and cancel near the zeros of
/// `ψ⁽ⁿ⁾` (for even `n`): the loss, the largest term over the value, is
/// measured and the evaluation repeated with that many more bits, up to
/// [`cancellation_cap`].  (Before 0.29 `x + k` was rounded at every step
/// and an `x + k` below `2^{−wp/2}` was taken for a pole:
/// `polygamma(1, −3 + √2·10⁻³⁰)` was refused at 16 digits.)
pub(super) fn arb_polygamma(
    n: u32,
    x: &BigFloat,
    prec: usize,
    _cc: &mut Consts,
) -> Result<BigFloat, SymplexError> {
    if x.is_nan() || x.is_inf_pos() || x.is_inf_neg() {
        return Err(SymplexError::Unevaluable {
            reason: "polygamma of special float value".into(),
        });
    }
    let cap = cancellation_cap(prec);
    let mut extra = 32;
    loop {
        let (value, largest) = polygamma_wp(n, x, prec + extra)?;
        let lost = match (largest.exponent(), value.exponent()) {
            _ if value.is_zero() => prec + extra,
            (Some(l), Some(v)) => usize::try_from(i64::from(l) - i64::from(v)).unwrap_or(0),
            _ => 0,
        };
        if lost + 16 <= extra {
            return Ok(round_to(value, prec));
        }
        if extra >= cap {
            return Err(special_exhausted(prec));
        }
        extra = (lost + 32).max(2 * extra).min(cap);
    }
}

/// `q·r` for a reduced fraction `q` and an integer `r`: the reduced fraction
/// `Ratio`'s product gives, with gcds against the denominator of `q` only
/// (of `r` modulo it) — not a binary gcd of `r` with 1, which takes a step
/// per bit of `r` (see [`factorial_exact`]).
pub(super) fn times_integer(q: &Ratio<BigInt>, r: &BigInt) -> Ratio<BigInt> {
    use num_integer::Integer;
    use num_traits::One;
    let mut num = r.clone();
    let mut den = q.denom().clone();
    loop {
        let g = den.gcd(&(&num % &den));
        if g.is_one() || g.is_zero() {
            break;
        }
        num /= &g;
        den /= &g;
    }
    Ratio::new_raw(q.numer() * num, den)
}

/// `ψ⁽ⁿ⁾(x)` at working precision `wp` (see [`arb_polygamma`]) and the
/// magnitude of the largest term summed.
fn polygamma_wp(n: u32, x: &BigFloat, wp: usize) -> Result<(BigFloat, BigFloat), SymplexError> {
    let exact = exact_bits(x, wp);
    let n_us = n as usize;
    let one = BigFloat::from_i32(1, wp);

    // n! and (n−1)! as BigFloats (exact integers first: see `factorial_exact`).
    let nm1_fact = factorial_exact(n_us.saturating_sub(1));
    let n_fact = Ratio::from_integer(nm1_fact.numer() * BigInt::from(n_us as u64));
    let n_fact_bf = ratio_to_bigfloat(&n_fact, wp);
    let nm1_fact_bf = ratio_to_bigfloat(&nm1_fact, wp);
    // (−1)^{n+1}
    let outer_sign_neg = n_us.is_multiple_of(2);

    // ── Step 1: shift x upward ──
    let threshold_val = ((wp / 3).max(n_us) + 10) as i128;
    let threshold = BigFloat::from_i128(threshold_val, wp);
    // Far left of 0 the shift would take `|x|` steps: refused up front (a
    // refusal, not a failure — before 0.30 a million steps were taken and
    // then `ComputationFailed` raised, for `polygamma(100, erfi(−5))`).
    let max_steps = 10 * wp + 1_000_000;
    if bf_lt(x, &BigFloat::from_i128(-(max_steps as i128), wp)) {
        return Err(SymplexError::NotImplemented(
            "polygamma of an argument far left of 0 (the argument shift would take more \
             than a million steps)"
                .into(),
        ));
    }
    let mut xw = x.clone();
    let mut shift_sum = BigFloat::new(wp); // Σ 1/(x+k)^{n+1}
    let mut largest_inv = BigFloat::new(wp);
    let mut steps: usize = 0;
    while xw.sub(&threshold, wp, RM).is_negative() {
        if xw.is_zero() {
            return Err(infinite("polygamma at non-positive integer pole"));
        }
        let p = xw.powi(n_us + 1, wp, RM);
        let inv = one.div(&p, wp, RM);
        if inv.abs() > largest_inv {
            largest_inv = inv.abs();
        }
        shift_sum = shift_sum.add(&inv, wp, RM);
        xw = xw.add(&one, exact, RM);
        steps += 1;
        if steps > max_steps {
            return Err(SymplexError::NotImplemented(
                "polygamma: the argument shift did not terminate".into(),
            ));
        }
    }
    // ψ⁽ⁿ⁾(x) = ψ⁽ⁿ⁾(x+N) + (−1)^{n+1} n! Σ 1/(x+k)^{n+1}
    let shift_term = n_fact_bf.mul(&shift_sum, wp, RM);

    // ── Step 2: asymptotic series at xw ──
    // bracket = (n−1)!/x^n + n!/(2 x^{n+1}) + Σ B_{2k} (2k+n−1)!/((2k)! x^{2k+n})
    let x_pow_n = xw.powi(n_us, wp, RM);
    let x_pow_np1 = x_pow_n.mul(&xw, wp, RM);
    let two = BigFloat::from_i32(2, wp);
    let mut bracket = nm1_fact_bf.div(&x_pow_n, wp, RM);
    bracket = bracket.add(&n_fact_bf.div(&two.mul(&x_pow_np1, wp, RM), wp, RM), wp, RM);

    let x2 = xw.mul(&xw, wp, RM);
    let mut x_pow = x_pow_n.mul(&x2, wp, RM); // x^{n+2} for k = 1
    // (2k+n−1)!/(2k)! = (2k+1)⋯(2k+n−1), an integer: start k=1 → (n+1)!/2!
    // (exact integer steps, the same values as the `Ratio` steps before
    // 0.34 without their gcds).
    let mut ratio_fact = n_fact.numer() * BigInt::from(n_us as u64 + 1) / BigInt::from(2);
    let mut prev_term_exp: Option<i32> = None;
    let max_terms = wp / 2 + 20;
    for k in 1..=max_terms {
        if k > 1 {
            // multiply by (2k+n−2)(2k+n−1) / ((2k−1)(2k))
            let a =
                BigInt::from((2 * k + n_us - 2) as u64) * BigInt::from((2 * k + n_us - 1) as u64);
            let b = BigInt::from((2 * k - 1) as u64) * BigInt::from((2 * k) as u64);
            ratio_fact = ratio_fact * a / b;
            x_pow = x_pow.mul(&x2, wp, RM);
        }
        let b2k = bernoulli::even(k);
        let coeff = ratio_to_bigfloat(&times_integer(&b2k, &ratio_fact), wp);
        let term = coeff.div(&x_pow, wp, RM);
        if let Some(t_exp) = term.exponent() {
            if let Some(prev) = prev_term_exp
                && t_exp > prev + 10
            {
                break; // diverging
            }
            prev_term_exp = Some(t_exp);
            if let Some(b_exp) = bracket.exponent()
                && (b_exp as i64 - t_exp as i64) > wp as i64
            {
                bracket = bracket.add(&term, wp, RM);
                break;
            }
        }
        bracket = bracket.add(&term, wp, RM);
    }

    let total = bracket.add(&shift_term, wp, RM);
    let result = if outer_sign_neg { total.neg() } else { total };
    let largest_shift = n_fact_bf.mul(&largest_inv, wp, RM);
    let largest = if largest_shift > bracket.abs() {
        largest_shift
    } else {
        bracket.abs()
    };
    Ok((result, largest))
}

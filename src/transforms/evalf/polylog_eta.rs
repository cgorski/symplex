//! Polylogarithm / Dirichlet eta (part of [`evalf`](super)).

use super::*;

// ── Polylogarithm / Dirichlet eta ───────────────────────────────────────────────

/// `ζ(m)` for an integer `m ≤ 0`, exactly: `ζ(0) = −1/2`,
/// `ζ(−k) = −B_{k+1}/(k+1)`.
pub(super) fn zeta_nonpositive_int(m: i64, wp: usize) -> BigFloat {
    if m == 0 {
        return BigFloat::from_f64(-0.5, wp);
    }
    let k = (-m) as usize;
    let b = crate::base::bernoulli::bernoulli(k + 1);
    if b.is_zero() {
        return BigFloat::new(wp);
    }
    let v = -b / Ratio::from_integer(BigInt::from(k as i64 + 1));
    ratio_to_bigfloat(&v, wp)
}

/// `k^s` for integer `k ≥ 1`.
fn int_pow_s(k: usize, s: &BigFloat, s_int: Option<i64>, w: &mut Wp<'_>) -> BigFloat {
    let wp = w.bits();
    let kb = BigFloat::from_i128(k as i128, wp);
    match s_int {
        Some(n) if n >= 0 => kb.powi(n as usize, wp, RM),
        Some(n) => BigFloat::from_i32(1, wp).div(&kb.powi((-n) as usize, wp, RM), wp, RM),
        None => bf_pow(&kb, s, w),
    }
}

/// Number of terms after which `|z|^k k^{−s}` has certainly fallen below
/// `2^{−wp}·|z|` (the first term): about `wp·ln 2 / (−ln|z|)` for `s ≥ 0`;
/// for `s < 0` the factor `k^{|s|}` first grows the terms (peak near
/// `k = |s|/(−ln|z|)`), so the bound is found by doubling from there.
pub(super) fn polylog_series_terms(s_f: f64, z_f: f64, wp: usize) -> usize {
    if !(z_f > 0.0 && z_f < 1.0) {
        return wp * 2 + 60;
    }
    let a = -z_f.ln(); // > 0
    let target = (wp as f64) * std::f64::consts::LN_2 + a;
    let mut k = target / a;
    let m = (-s_f).max(0.0);
    if m > 0.0 {
        // k·a − m·ln k increases for k > m/a.
        k = k.max(m / a);
        for _ in 0..200 {
            if k * a - m * k.ln() >= target {
                break;
            }
            k *= 2.0;
        }
    }
    (k.ceil() as usize).saturating_add(20)
}

/// Direct series `Li_s(z) = Σ_{k≥1} z^k/k^s` for `|z| < 1` (used up to
/// [`polylog_series_limit`]); see [`polylog_series_terms`] for the term
/// bound.  For `z < 0` and `s < 0` the alternating terms can be far
/// larger than the sum; the bits lost to that cancellation are measured
/// and the sum recomputed once with that many guard bits.
fn polylog_series(
    s: &BigFloat,
    s_int: Option<i64>,
    z: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let wp = w.bits();
    let (sum, lost) = polylog_series_raw(s, s_int, z, w)?;
    // A few bits (an alternating series with s ≥ 0 loses about one) are
    // covered by the caller's guard bits.
    if lost <= 4 {
        return Ok(sum);
    }
    let wp2 = wp + lost + 8;
    let mut sw = s.clone();
    let _ = sw.set_precision(wp2, RM);
    let mut zw = z.clone();
    let _ = zw.set_precision(wp2, RM);
    let (sum, _) = polylog_series_raw(&sw, s_int, &zw, &mut w.at(wp2))?;
    Ok(round_to(sum, wp))
}

/// One pass of [`polylog_series`] at working precision `wp`; also returns
/// the number of bits by which the largest term exceeded the final sum.
fn polylog_series_raw(
    s: &BigFloat,
    s_int: Option<i64>,
    z: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<(BigFloat, usize), SymplexError> {
    let wp = w.bits();
    let z_f = f64_of(z).unwrap_or(0.5).abs();
    let s_f = f64_of(s).unwrap_or(0.0);
    let max_terms = polylog_series_terms(s_f, z_f, wp);
    let mut zk = z.clone();
    let mut sum = BigFloat::new(wp);
    let mut max_exp: Option<i64> = None;
    let mut converged = false;
    for k in 1..=max_terms {
        if k > 1 {
            zk = zk.mul(z, wp, RM);
        }
        let term = zk.div(&int_pow_s(k, s, s_int, w), wp, RM);
        if let Some(e) = term.exponent() {
            max_exp = Some(max_exp.map_or(e as i64, |m| m.max(e as i64)));
        }
        sum = sum.add(&term, wp, RM);
        if negligible(&term, &sum, wp) {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(series_did_not_converge("polylog", max_terms));
    }
    let lost = match (max_exp, sum.exponent()) {
        (Some(m), Some(se)) if m > se as i64 => (m - se as i64) as usize,
        _ => 0,
    };
    Ok((sum, lost))
}

/// The largest `n` of `Li₋ₙ(z)` by its finite sum ([`polylog_nonpositive`],
/// `polylog::nonpositive`): the row of `n + 1` Stirling numbers takes
/// `O(n²)` operations on integers of up to `n·log₂ n` bits (0.1 s at 1000).
pub(super) const MAX_NONPOSITIVE_ORDER: u64 = 1000;

thread_local! {
    /// The last row of [`stirling2_row`]: the precision loops of `Li₋ₙ` ask
    /// for the same row at every pass.
    static STIRLING_ROW: std::cell::RefCell<Option<(u64, std::rc::Rc<Vec<BigInt>>)>> =
        const { std::cell::RefCell::new(None) };
}

/// The Stirling numbers of the second kind `S(m, j)`, `j = 0 … m`: one row
/// of the triangle `S(i, j) = j·S(i−1, j) + S(i−1, j−1)`, the integers
/// `crate::domains::combinatorics::stirling2` gives one at a time by the
/// same triangle (`O(m·j)` each).  Before 0.34 the row of `Li₋ₙ` took them
/// one at a time, `O(n³)`: `polylog(−601, 4/13)` 18 s, `polylog(−1401,
/// 4/13)` minutes.  Refused beyond [`MAX_NONPOSITIVE_ORDER`]` + 1`.
pub(super) fn stirling2_row(m: u64) -> Result<std::rc::Rc<Vec<BigInt>>, SymplexError> {
    if m > MAX_NONPOSITIVE_ORDER + 1 {
        return Err(SymplexError::NotImplemented(format!(
            "polylog of an order below -{MAX_NONPOSITIVE_ORDER} (a finite sum of {m} terms with \
             Stirling numbers)"
        )));
    }
    if let Some(row) = STIRLING_ROW.with(|c| {
        c.borrow()
            .as_ref()
            .filter(|(k, _)| *k == m)
            .map(|(_, r)| std::rc::Rc::clone(r))
    }) {
        return Ok(row);
    }
    let mu = m as usize;
    let mut row = vec![BigInt::zero(); mu + 1];
    row[0] = BigInt::from(1);
    for i in 1..=mu {
        for j in (1..=i).rev() {
            let (lower, upper) = row.split_at_mut(j);
            upper[0] *= j as u64;
            upper[0] += &lower[j - 1];
        }
        row[0] = BigInt::zero();
    }
    let row = std::rc::Rc::new(row);
    STIRLING_ROW.with(|c| *c.borrow_mut() = Some((m, std::rc::Rc::clone(&row))));
    Ok(row)
}

/// `Li_{−n}(z)` for integer `n ≥ 0` and any real `z ≠ 1`:
/// `Σ_{k=0}^{n} k! S(n+1, k+1) (z/(1−z))^{k+1}`, at `wp` bits, and `log₂` of
/// its largest term.  For `z < 0` the terms alternate and cancel (see
/// [`polylog_nonpositive_measured`]).
fn polylog_nonpositive(n: usize, z: &BigFloat, wp: usize) -> Result<(BigFloat, f64), SymplexError> {
    let one = BigFloat::from_i32(1, wp);
    let one_minus_z = one.sub(z, wp, RM);
    if one_minus_z.is_zero() {
        return Err(infinite("polylog(s, 1) diverges for s ≤ 1"));
    }
    let w = z.div(&one_minus_z, wp, RM);
    let row = stirling2_row(n as u64 + 1)?;
    let mut k_fact = BigInt::from(1);
    let mut w_pow = w.clone();
    let mut sum = BigFloat::new(wp);
    let mut largest = f64::NEG_INFINITY;
    for k in 0..=n {
        if k > 0 {
            k_fact *= BigInt::from(k as u64);
            w_pow = w_pow.mul(&w, wp, RM);
        }
        let c = ratio_to_bigfloat(&Ratio::from_integer(&k_fact * &row[k + 1]), wp);
        let term = c.mul(&w_pow, wp, RM);
        if let Some(e) = term.exponent().filter(|_| !term.is_zero()) {
            largest = largest.max(f64::from(e));
        }
        sum = sum.add(&term, wp, RM);
    }
    Ok((sum, largest))
}

/// [`polylog_nonpositive`] to `prec` bits: the loss of the alternating sum
/// of a negative `z` (the largest term over the value) measured and made up
/// with more bits, up to [`cancellation_cap`]; the first pass, at `prec +
/// 32` bits, as before 0.34.  Before 0.34 it went unmeasured:
/// `polylog(−200, −1/2)` was `2.03·10²⁹⁶`, truly `−5.15·10²⁷²`.
fn polylog_nonpositive_measured(
    n: usize,
    z: &BigFloat,
    prec: usize,
) -> Result<BigFloat, SymplexError> {
    let cap = cancellation_cap(prec);
    let mut extra = 32;
    loop {
        let (value, largest) = polylog_nonpositive(n, z, prec + extra)?;
        let lost = match value.exponent() {
            Some(e) if !value.is_zero() && largest.is_finite() => {
                (largest - f64::from(e)).max(0.0).ceil() as usize
            }
            _ if largest.is_finite() => prec + extra,
            _ => 0,
        };
        if lost + 16 <= extra {
            return Ok(round_to(value, prec));
        }
        if extra >= cap || lost + 16 > cap {
            return Err(special_exhausted(prec));
        }
        extra = (lost + 32).max(2 * extra).min(cap);
    }
}

/// `Li_s(z)` for `1/2 < z < 1` via the expansion in `μ = ln z` (`|μ| < 2π`):
///
/// * integer `s = n ≥ 1`:
///   `Li_n(e^μ) = μ^{n−1}/(n−1)! (H_{n−1} − ln(−μ)) + Σ_{k≥0, k≠n−1} ζ(n−k) μ^k/k!`
/// * non-integer `s`: `Li_s(e^μ) = Γ(1−s)(−μ)^{s−1} + Σ_{k≥0} ζ(s−k) μ^k/k!`
///
/// For `s` next to an integer `n` the first term and the term `k = n − 1`
/// have opposite poles (`Γ(1 − s)` and `ζ(s − n + 1)`, about `1/|s − n|`)
/// that cancel: the loss, the largest term over the sum, is measured and
/// the expansion repeated with that many more bits (up to
/// [`cancellation_cap`]).  Before 0.30 it went unmeasured, and
/// `polylog(2 + 10⁻²⁰, 999/1000)` was wrong from its 20th digit at 60
/// digits (`polylog(1 + 10⁻³⁰, 95/100)` from its 29th).
fn polylog_near_one(
    s: &BigFloat,
    s_int: Option<i64>,
    z: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let wp = w.bits();
    // The caller's `wp` carries 32 guard bits; a loss within 24 of them
    // costs nothing.
    let cap = cancellation_cap(wp);
    let mut extra = 0usize;
    loop {
        let p = wp + extra;
        let (sum, largest) = polylog_near_one_at(s, s_int, z, &mut w.at(p))?;
        let lost = match (largest, sum.exponent()) {
            (Some(l), Some(e)) if !sum.is_zero() => usize::try_from(l - i64::from(e)).unwrap_or(0),
            (Some(_), _) => p,
            _ => 0,
        };
        if lost <= extra + 24 {
            return Ok(round_to(sum, wp));
        }
        if extra >= cap {
            return Err(special_exhausted(wp));
        }
        extra = (lost + 16).max(2 * extra).min(cap);
    }
}

/// `ζ(σ₁), ζ(σ₁ − 1), ζ(σ₁ − 2), …` for a non-integer `σ₁ < −1`: the
/// terms of the `μ`-expansion of [`polylog_near_one`] left of 0, where each
/// used to cost a whole `ζ` evaluation (a `Γ` by Stirling's series and a
/// Borwein sum with its exact coefficients rebuilt: `polylog(1/3, −1 +
/// 10⁻¹²)` spent 0.35 s on some 25 of them at 60 digits).  By the
/// functional equation `ζ(σ) = 2^σ π^{σ−1} sin(πσ/2) Γ(1−σ) ζ(1−σ)`
/// every factor follows from the previous one: `2^σ π^{σ−1}` divided by
/// `2π`, the sine a quarter period on, `Γ(t)` times `t` (`t = 1 − σ`), and
/// `ζ(t) = η(t)/(1 − 2^{1−t})` for `t ≥ 2` by one Borwein sum (Algorithm 2
/// of the paper cited at [`arb_eta_borwein`]) whose coefficients are built
/// once and whose powers `m^{−t}` are divided by `m` from one `t` to the
/// next.  The working precision carries 16 guard bits for the roundings
/// that accumulate over the steps.
pub(super) struct ZetaLeft {
    wp: usize,
    /// `2^σ π^{σ−1}`.
    scale: BigFloat,
    inv_two_pi: BigFloat,
    /// `sin(πσ/2)`, `cos(πσ/2)`.
    sin: BigFloat,
    cos: BigFloat,
    /// `Γ(t)` and `t = 1 − σ`.
    gamma: BigFloat,
    t: BigFloat,
    /// Borwein's `(−1)^k (d_k − d_n)/d_n`, `k < n`.
    coeffs: Vec<BigFloat>,
    /// `m^{−t}`, `m = 1 … n` (index `m − 1`).
    powers: Vec<BigFloat>,
    /// `1/m`.
    inverses: Vec<BigFloat>,
    /// `2^{1−t}`.
    two_pow: BigFloat,
}

impl ZetaLeft {
    pub(super) fn new(sigma: &BigFloat, w: &mut Wp<'_>) -> Result<Self, SymplexError> {
        let wp = w.bits();
        let wp = wp + 16;
        let one = BigFloat::from_i32(1, 64);
        let t = one.sub(sigma, exact_bits(sigma, wp), RM);
        let pi = w.cc.pi(wp, RM).clone();
        let two = BigFloat::from_i32(2, 64);
        let scale = bf_pow(&two, sigma, &mut w.at(wp)).mul(
            &bf_pow(&pi, &sigma.sub(&one, wp, RM), &mut w.at(wp)),
            wp,
            RM,
        );
        let half_pi_sigma = pi.mul(sigma, wp, RM).div(&two, wp, RM);
        let gamma = arb_gamma_real(&t, &mut w.at(wp))?;
        let n =
            ((wp as f64) * std::f64::consts::LN_2 / (3.0 + 8f64.sqrt()).ln()).ceil() as usize + 4;
        let d = borwein_d(n);
        let coeffs = (0..n)
            .map(|k| {
                // (The quotient of the integers as it stands: reducing it
                // costs a gcd of `wp`-bit integers for each `k`.)
                let q = Ratio::new_raw(d[k].numer() - d[n].numer(), d[n].numer().clone());
                let c = ratio_to_bigfloat(&q, wp);
                if k % 2 == 0 { c } else { c.neg() }
            })
            .collect();
        // m^{−t}: a power for each prime, products for the composites.
        let neg_t = t.neg();
        let mut spf = vec![0usize; n + 1];
        let mut powers: Vec<BigFloat> = Vec::with_capacity(n);
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
                bf_pow(&BigFloat::from_u64(m as u64, 64), &neg_t, &mut w.at(wp))
            } else {
                powers[spf[m] - 1].mul(&powers[m / spf[m] - 1], wp, RM)
            };
            powers.push(v);
        }
        let inverses = (1..=n)
            .map(|m| one.div(&BigFloat::from_u64(m as u64, 64), wp, RM))
            .collect();
        let two_pow = bf_pow(&two, &one.sub(&t, wp, RM), &mut w.at(wp));
        Ok(ZetaLeft {
            wp,
            scale,
            inv_two_pi: one.div(&pi.mul(&two, wp, RM), wp, RM),
            sin: half_pi_sigma.sin(wp, RM, w.cc),
            cos: half_pi_sigma.cos(wp, RM, w.cc),
            gamma,
            t,
            coeffs,
            powers,
            inverses,
            two_pow,
        })
    }

    /// `ζ(σ)` at the current `σ`, then a step to `σ − 1`.
    pub(super) fn next(&mut self) -> BigFloat {
        let wp = self.wp;
        let mut eta = BigFloat::new(wp);
        for (c, p) in self.coeffs.iter().zip(self.powers.iter()) {
            eta = eta.add(&c.mul(p, wp, RM), wp, RM);
        }
        let one = BigFloat::from_i32(1, 64);
        let zeta_t = eta.neg().div(&one.sub(&self.two_pow, wp, RM), wp, RM);
        let value = self
            .scale
            .mul(&self.sin, wp, RM)
            .mul(&self.gamma, wp, RM)
            .mul(&zeta_t, wp, RM);
        // σ → σ − 1, t → t + 1.
        self.scale = self.scale.mul(&self.inv_two_pi, wp, RM);
        let (s, c) = (self.cos.neg(), self.sin.clone());
        self.sin = s;
        self.cos = c;
        self.gamma = self.gamma.mul(&self.t, wp, RM);
        self.t = self.t.add(&one, wp, RM);
        for (p, inv) in self.powers.iter_mut().zip(self.inverses.iter()) {
            *p = p.mul(inv, wp, RM);
        }
        self.two_pow = self.two_pow.div(&BigFloat::from_i32(2, 64), wp, RM);
        value
    }
}

/// One evaluation of [`polylog_near_one`] at working precision `wp`, and
/// the binary exponent of the largest term summed.
fn polylog_near_one_at(
    s: &BigFloat,
    s_int: Option<i64>,
    z: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<(BigFloat, Option<i64>), SymplexError> {
    let wp = w.bits();
    let mut largest: Option<i64> = None;
    let mut note = |v: &BigFloat| {
        if !v.is_zero()
            && let Some(e) = v.exponent()
        {
            largest = Some(largest.map_or(i64::from(e), |m| m.max(i64::from(e))));
        }
    };
    let one = BigFloat::from_i32(1, wp);
    let mu = z.ln(wp, RM, w.cc); // negative
    let neg_mu = mu.neg();
    // The terms decay like (|μ|/2π)^k (|ζ(−m)| ~ m!/(2π)^m), |μ| < ln 2.
    let max_terms = wp * 2 + 100;
    let not_converged = || series_did_not_converge("polylog", max_terms);
    let mut sum = BigFloat::new(wp);
    let mut mu_pow_over_fact = one.clone(); // μ^k/k!
    // ζ vanishes at the negative even integers, so a single negligible
    // term proves nothing: stop only after two in a row.
    let mut small_run = 0u8;
    let mut converged = false;
    match s_int {
        Some(n) if n >= 1 => {
            let n = n as usize;
            for k in 0..max_terms {
                if k > 0 {
                    mu_pow_over_fact = mu_pow_over_fact.mul(&mu, wp, RM).div(
                        &BigFloat::from_i128(k as i128, wp),
                        wp,
                        RM,
                    );
                }
                if k + 1 == n {
                    // μ^{n−1}/(n−1)! (H_{n−1} − ln(−μ))
                    let mut h = BigFloat::new(wp);
                    for j in 1..n {
                        h = h.add(
                            &one.div(&BigFloat::from_i128(j as i128, wp), wp, RM),
                            wp,
                            RM,
                        );
                    }
                    let ln_neg_mu = neg_mu.ln(wp, RM, w.cc);
                    let t = mu_pow_over_fact.mul(&h.sub(&ln_neg_mu, wp, RM), wp, RM);
                    note(&t);
                    sum = sum.add(&t, wp, RM);
                    continue;
                }
                let arg = n as i64 - k as i64;
                let zeta = if arg <= 0 {
                    zeta_nonpositive_int(arg, wp)
                } else {
                    arb_zeta(&BigFloat::from_i128(arg as i128, wp), w)?
                };
                let term = zeta.mul(&mu_pow_over_fact, wp, RM);
                note(&term);
                sum = sum.add(&term, wp, RM);
                if k >= n {
                    if negligible(&term, &sum, wp) {
                        small_run += 1;
                        if small_run >= 2 {
                            converged = true;
                            break;
                        }
                    } else {
                        small_run = 0;
                    }
                }
            }
            if !converged {
                return Err(not_converged());
            }
            Ok((sum, largest))
        }
        _ => {
            // `1 − s` exact, as `s − k` below: next to an integer the poles of
            // `Γ(1 − s)` and `ζ(s − n + 1)` cancel only for the same `s` (a
            // rounded `1 − s` moves `1/|s − n|` by its rounding over
            // `(s − n)²`).
            let one_minus_s = BigFloat::from_i32(1, 64).sub(s, exact_bits(s, wp), RM);
            let g = arb_gamma_real(&one_minus_s, w)?;
            let s_minus_1 = s.sub(&one, wp, RM);
            let lead = g.mul(&bf_pow(&neg_mu, &s_minus_1, w), wp, RM);
            note(&lead);
            let minus_one = BigFloat::from_i32(-1, 64);
            let mut left: Option<ZetaLeft> = None;
            for k in 0..max_terms {
                if k > 0 {
                    mu_pow_over_fact = mu_pow_over_fact.mul(&mu, wp, RM).div(
                        &BigFloat::from_i128(k as i128, wp),
                        wp,
                        RM,
                    );
                }
                let arg = s.sub(&BigFloat::from_i128(k as i128, wp), exact_bits(s, wp), RM);
                // Left of −1 from the sequence (see [`ZetaLeft`]).
                let zeta = if left.is_some() || bf_lt(&arg, &minus_one) {
                    if left.is_none() {
                        left = Some(ZetaLeft::new(&arg, w)?);
                    }
                    match left.as_mut() {
                        Some(l) => l.next(),
                        None => arb_zeta(&arg, w)?,
                    }
                } else {
                    arb_zeta(&arg, w)?
                };
                let term = zeta.mul(&mu_pow_over_fact, wp, RM);
                note(&term);
                sum = sum.add(&term, wp, RM);
                if k >= 2 {
                    if negligible(&term, &sum, wp) {
                        small_run += 1;
                        if small_run >= 2 {
                            converged = true;
                            break;
                        }
                    } else {
                        small_run = 0;
                    }
                }
            }
            if !converged {
                return Err(not_converged());
            }
            Ok((lead.add(&sum, wp, RM), largest))
        }
    }
}

/// `Li_s(−x)` for `½ < x < 1` by the expansion at `z = −1` in `μ = ln x`:
///
/// ```text
/// Li_s(−e^μ) = −Σ_{k≥0} η(s − k)·μ^k/k!,   abs(μ) < π,
/// ```
///
/// with `−η(σ) = (2^{1−σ} − 1)·ζ(σ)` (`−ln 2` at `σ = 1`).  It is the
/// duplication formula `Li_s(−x) = 2^{1−s}·Li_s(x²) − Li_s(x)` with both terms
/// expanded at `z = 1` ([`polylog_near_one`]): their terms `Γ(1 − s)(−μ)^{s−1}`
/// (the logarithmic ones for an integer `s`) cancel identically.  Up to 0.58
/// the two were evaluated and subtracted: next to `−1` that cancellation is
/// `(1 − s)·log₂(1/abs(μ))` bits — 580 for `polylog(−3/2, −1 + 10⁻⁷⁰)` —, found
/// by repeating both expansions with more bits (0.72 s at 60 digits).
///
/// The loss where `Li_s(−x)` is small next to its terms (next to a zero of
/// `η`) is measured and made up as in [`polylog_near_one`].
fn polylog_near_minus_one(
    s: &BigFloat,
    s_int: Option<i64>,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let wp = w.bits();
    let cap = cancellation_cap(wp);
    let mut extra = 0usize;
    loop {
        let p = wp + extra;
        let (sum, largest) = polylog_near_minus_one_at(s, s_int, x, &mut w.at(p))?;
        let lost = match (largest, sum.exponent()) {
            (Some(l), Some(e)) if !sum.is_zero() => usize::try_from(l - i64::from(e)).unwrap_or(0),
            (Some(_), _) => p,
            _ => 0,
        };
        if lost <= extra + 24 {
            return Ok(round_to(sum, wp));
        }
        if extra >= cap {
            return Err(special_exhausted(wp));
        }
        extra = (lost + 16).max(2 * extra).min(cap);
    }
}

/// One evaluation of [`polylog_near_minus_one`] at working precision `wp`,
/// and the binary exponent of the largest term summed.
///
/// The sum stops on a bound of its tail from the next index `j` on: the
/// terms with `σ = s − i > −1` by `abs(η(σ)) ≤ 1` (`η` lies in `(0, 1)` right of
/// 0 and in `[1/4, 1/2]` on `[−1, 0]`) and the tail of `e^abs(μ)`; those with
/// `σ ≤ −1` by the functional equation, `abs(η(σ)) ≤ (1 + 2^t)·2(2π)^(−t)Γ(t)ζ(t) ≤
/// 4.125·π^(−t)·Γ(t)` for `t = 1 − σ ≥ 2`, whose terms fall by `abs(μ)/π·max(1,
/// (i + 1 − s)/(i + 1))` or more from one to the next.
fn polylog_near_minus_one_at(
    s: &BigFloat,
    s_int: Option<i64>,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<(BigFloat, Option<i64>), SymplexError> {
    const LOG2E: f64 = std::f64::consts::LOG2_E;
    const LOG2_PI: f64 = 1.651_496_129_472_319;
    let wp = w.bits();
    let s_f = f64_of(s)?;
    let mut largest: Option<i64> = None;
    let mut note = |v: &BigFloat| {
        if !v.is_zero()
            && let Some(e) = v.exponent()
        {
            largest = Some(largest.map_or(i64::from(e), |m| m.max(i64::from(e))));
        }
    };
    let mu = x.ln(wp, RM, w.cc); // negative
    let lmu = accuracy::part_lg(&mu);
    let mu_abs = lmu.exp2();
    let max_terms = wp * 2 + 100;
    let minus_one = BigFloat::from_i32(-1, 64);
    let mut sum = BigFloat::new(wp);
    let mut mu_pow_over_fact = BigFloat::from_i32(1, wp); // μ^k/k!
    let mut left: Option<ZetaLeft> = None;
    for k in 0..max_terms {
        if k > 0 {
            mu_pow_over_fact =
                mu_pow_over_fact
                    .mul(&mu, wp, RM)
                    .div(&BigFloat::from_i128(k as i128, wp), wp, RM);
        }
        // `σ = s − k` and `1 − σ`, exactly.
        let sigma = s.sub(&BigFloat::from_i128(k as i128, wp), exact_bits(s, wp), RM);
        let one_minus_sigma = BigFloat::from_i32(1, 64).sub(&sigma, exact_bits(&sigma, wp), RM);
        // `−η(σ) = (2^{1−σ} − 1)·ζ(σ)`, each factor relative to itself next to
        // `σ = 1`.
        let coefficient = if one_minus_sigma.is_zero() {
            w.cc.ln_2(wp, RM).neg()
        } else {
            let zeta = match s_int {
                Some(n) if n - k as i64 <= 0 => zeta_nonpositive_int(n - k as i64, wp),
                Some(_) => arb_zeta(&sigma, w)?,
                // Left of −1 from the sequence (see [`ZetaLeft`]).
                None if left.is_some() || bf_lt(&sigma, &minus_one) => {
                    if left.is_none() {
                        left = Some(ZetaLeft::new(&sigma, w)?);
                    }
                    match left.as_mut() {
                        Some(l) => l.next(),
                        None => arb_zeta(&sigma, w)?,
                    }
                }
                None => arb_zeta(&sigma, w)?,
            };
            one_minus_pow2(&one_minus_sigma, w).neg().mul(&zeta, wp, RM)
        };
        let term = coefficient.mul(&mu_pow_over_fact, wp, RM);
        note(&term);
        sum = sum.add(&term, wp, RM);
        if sum.is_zero() {
            continue;
        }
        // The tail from `j = k + 1` on.
        let j = (k + 1) as f64;
        let tail_of = |i: f64| i * lmu - polylog::ln_gamma(i + 1.0) * LOG2E;
        let right = tail_of(j) - (1.0 - mu_abs / (j + 1.0)).log2();
        let j0 = j.max((s_f + 1.0).ceil());
        let q = mu_abs / std::f64::consts::PI * ((j0 + 1.0 - s_f) / (j0 + 1.0)).max(1.0);
        let left_tail = if q < 1.0 {
            2.05 - (j0 + 1.0 - s_f) * LOG2_PI
                + polylog::ln_gamma(j0 + 1.0 - s_f) * LOG2E
                + tail_of(j0)
                - (1.0 - q).log2()
        } else {
            f64::INFINITY
        };
        let tail = accuracy::lsum(right, left_tail);
        if tail <= accuracy::part_lg_low(&sum) - wp as f64 - 2.0 {
            return Ok((sum, largest));
        }
    }
    Err(series_did_not_converge("polylog", max_terms))
}

/// Largest `|z|` for which the direct series is used.  For integer `s` the
/// `μ`-expansion is cheap (Bernoulli numbers), so it takes over at `1/2`;
/// for non-integer `s` its terms `ζ(s − k)` come from one incremental
/// sequence left of −1 ([`ZetaLeft`]), a few multiplications per Borwein
/// term, and it takes over at `3/4`, where the direct series needs
/// `wp·ln 2/ln(4/3)` terms of a non-integer power each.  (Before 0.30 every
/// `ζ(s − k)` was a full evaluation and the direct series ran up to `0.95`:
/// 1,800 powers for `polylog(1/3, −95/100)` at 60 digits, 0.57 s.)
fn polylog_series_limit(s_int: Option<i64>, wp: usize) -> BigFloat {
    if s_int.is_some() {
        BigFloat::from_f64(0.5, wp)
    } else {
        BigFloat::from_f64(0.75, wp)
    }
}

/// Is `Li_s(−x)` (`0 < x < 1`) taken by [`polylog_near_minus_one`]: `1 − x <
/// 2⁻⁴`, where `abs(ln x) ≤ (1 − x)/x < 0.067` and each term of the expansion
/// at `−1` gains `log₂(π/0.067) > 5.5` bits?  Farther from `−1` the
/// duplication formula stands, as before: there its two expansions cancel a
/// few bits only.
fn polylog_minus_one_expansion_applies(x: &BigFloat, wp: usize) -> bool {
    let one = BigFloat::from_i32(1, wp);
    let gap = one.sub(x, wp, RM);
    !gap.is_negative() && !gap.is_zero() && gap.exponent().is_some_and(|e| e <= -4)
}

/// `Li_s(z)` for `0 < |z| < 1`.
fn polylog_unit_interval(
    s: &BigFloat,
    s_int: Option<i64>,
    z: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let wp = w.bits();
    let limit = polylog_series_limit(s_int, wp);
    let az = z.abs();
    // A large order: `k^(−s) < 2^(−wp)` from `k = 16` on, the direct series
    // ends after a few terms wherever `abs(z) < 1`.  Before 0.37 the expansion
    // in `ln z` took over beyond `abs(z) = ½` and summed `ζ(s − k)` for every
    // `k < s`: `polylog(1038, −9/10)` was refused ("did not converge").
    let large = f64_of(s).is_ok_and(|v| v >= wp as f64 / 4.0 + 8.0);
    if large || !bf_gt(&az, &limit) {
        return polylog_series(s, s_int, z, w);
    }
    if z.is_negative() {
        // The expansion at `z = −1` where it takes few terms (each gains
        // `log₂(π/abs(ln x))` bits): no cancellation of the two expansions
        // at `z = 1` below.
        if polylog_minus_one_expansion_applies(&az, wp) {
            return polylog_near_minus_one(s, s_int, &az, w);
        }
        // Li_s(−x) = 2^{1−s} Li_s(x²) − Li_s(x).  The difference cancels
        // where Li_s(−x) is small next to its terms (next to a zero of
        // η(s) = −Li_s(−1), s ≈ −2, −4, …): the loss is measured and the
        // difference formed again with that many more bits.
        let cap = cancellation_cap(wp);
        let mut extra = 0usize;
        loop {
            let p = wp + extra;
            let one = BigFloat::from_i32(1, p);
            let x2 = az.mul(&az, 2 * p, RM);
            let a = polylog_unit_interval(s, s_int, &x2, &mut w.at(p))?;
            let b = polylog_near_one(s, s_int, &az, &mut w.at(p))?;
            let two = BigFloat::from_i32(2, p);
            let one_minus_s = one.sub(s, exact_bits(s, p), RM);
            let scale = bf_pow(&two, &one_minus_s, &mut w.at(p));
            let first = scale.mul(&a, p, RM);
            let r = first.sub(&b, p, RM);
            let largest = first.exponent().max(b.exponent()).map_or(0, i64::from);
            let lost = match r.exponent() {
                Some(e) if !r.is_zero() => usize::try_from(largest - i64::from(e)).unwrap_or(0),
                _ => p,
            };
            if lost <= extra + 24 {
                return Ok(round_to(r, wp));
            }
            if extra >= cap {
                return Err(special_exhausted(wp));
            }
            extra = (lost + 16).max(2 * extra).min(cap);
        }
    }
    polylog_near_one(s, s_int, z, w)
}

/// Polylogarithm `Li_s(z)` for real `s` and real `−1 ≤ z ≤ 1` (any real
/// `z ≠ 1` when `s` is a non-positive integer, where `Li_s` is rational).
/// A non-real `z` or `|z| > 1` is `polylog::polylog_general`.
pub(super) fn arb_polylog(
    s: &BigFloat,
    z: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    if s.is_nan() || z.is_nan() || s.is_inf() || z.is_inf() {
        return Err(unevaluable("polylog of special float value"));
    }
    if z.is_zero() {
        return Ok(BigFloat::new(prec));
    }
    let wp = prec + 32;
    let one = BigFloat::from_i32(1, wp);
    let s_int = bf_as_int(s)?;
    if let Some(n) = s_int
        && n <= 0
    {
        // Beyond the Stirling sum: the poles of `polylog::nonpositive_poles`,
        // their cancellation measured by `polylog_general` (up to its
        // largest order; beyond, the Stirling sum refuses).
        if n.unsigned_abs() > MAX_NONPOSITIVE_ORDER && n.unsigned_abs() <= 1_000_000 {
            let v = polylog::polylog_general(s, &(z.clone(), BigFloat::new(prec)), w)?;
            return Ok(round_to(v.0, prec));
        }
        return polylog_nonpositive_measured(n.unsigned_abs() as usize, z, prec);
    }
    let az = z.abs();
    if bf_gt(&az, &one) {
        return Err(unevaluable(
            "polylog outside the unit disc not yet supported in evalf",
        ));
    }
    if az.sub(&one, wp, RM).is_zero() {
        // z = ±1
        if z.is_negative() {
            // Li_s(−1) = −η(s), entire in s.
            let eta = arb_eta(s, &mut w.at(wp))?;
            return Ok(round_to(eta.neg(), prec));
        }
        if !bf_gt(s, &one) {
            return Err(infinite("polylog(s, 1) diverges for s ≤ 1"));
        }
        let zeta = arb_zeta(s, &mut w.at(wp))?;
        return Ok(round_to(zeta, prec));
    }
    let r = polylog_unit_interval(s, s_int, z, &mut w.at(wp))?;
    Ok(round_to(r, prec))
}

/// Dirichlet eta `η(s)` at working precision `wp` for real `s`: Borwein's
/// alternating sum ([`arb_eta_borwein`]) for `s > 0` — `η` is entire, so
/// nothing special happens near `s = 1` (`η(1) = ln 2`) — and
/// `(1 − 2^{1−s}) ζ(s)` for `s ≤ 0`.
fn arb_eta(s: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let wp = w.bits();
    // Strictly: `+0` has the sign bit of a positive number, and Borwein's sum at
    // `s = 0` is not `η(0) = 1/2` (before 0.30 `dirichlet_eta` of an argument
    // that cancelled to `+0` printed `0.4913903202446295`).
    if bf_strictly_positive(s) {
        return arb_eta_borwein(s, w);
    }
    let one = BigFloat::from_i32(1, wp);
    let two = BigFloat::from_i32(2, wp);
    let one_minus_s = one.sub(s, wp, RM);
    let coeff = one.sub(&bf_pow(&two, &one_minus_s, w), wp, RM);
    let zeta = arb_zeta(s, w)?;
    Ok(coeff.mul(&zeta, wp, RM))
}

/// Dirichlet eta `η(s)` (see [`arb_eta`]), with `η(+∞) = 1`.
pub(super) fn arb_dirichlet_eta(s: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    if s.is_nan() || s.is_inf_neg() {
        return Err(unevaluable("dirichlet_eta of special float value"));
    }
    if s.is_inf_pos() {
        return Ok(BigFloat::from_i32(1, prec));
    }
    let wp = prec + 32;
    let mut sw = s.clone();
    let _ = sw.set_precision(wp, RM);
    Ok(round_to(arb_eta(&sw, &mut w.at(wp))?, prec))
}

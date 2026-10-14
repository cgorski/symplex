//! `ζ(s)` for large `abs(Im s)` by the Riemann–Siegel formula with Arias de
//! Reyna's explicit error bounds ("High precision computation of Riemann's
//! zeta function by the Riemann–Siegel formula", I: Math. Comp. 80 (2011),
//! 995–1009; II: preprint), as mpmath's `rs_zeta` (`zeta_offline`,
//! `Rzeta_simul`, `_coef` of `mpmath/functions/rszeta.py`, BSD; written by
//! J. Arias de Reyna) — a port of its derivative-0 path.  Before 0.45 the
//! Euler–Maclaurin sum of `zeta.rs` needed `abs(t)/(2π)` terms: `ζ(1/2 +
//! 4·10⁵i)` took a second and `ζ(1/2 + 10⁶i)` was refused ("an order or
//! argument this large is not supported in evalf").
//!
//! For `s = σ + it`, `t > 0` (`ζ(s̄) = conj ζ(s)` otherwise),
//!
//! ```text
//! ζ(s) = R(s) + e^(−2iθ)·conj(R(1 − s̄)),   θ = ϑ(t − i(σ − 1/2)),
//! R(s) = Σ_{n≤N} n^(−s) + (−1)^(N−1)·U·a^(−σ)·Σ_{k<L} C_k(p)/a^k,
//! ```
//!
//! `a = √(t/2π)`, `N = ⌊a⌋`, `p = 1 − 2(a − N)`, `U = e^(−i(t/2·ln(t/2π) −
//! t/2 − π/8))`, `ϑ` the Riemann–Siegel theta function (from `ln Γ`,
//! `loggamma.rs`); the corrections `C_k` come from the derivatives of
//! `F(p) = Σ c_{2n} p^(2n)` (the Taylor coefficients from the Euler
//! numbers) and the recurrence coefficients `d(n, k)`.  Every truncation
//! (`L` terms, `J` coefficients of `F`, `M` derivatives) and every working
//! precision is chosen by the paper's bounds so that the absolute error is
//! below `2^−wp`; where its conditions fail (`3L ≥ 2a²/25`, `abs(σ) > a/2`)
//! the formula is not used and `zeta.rs` sums by Euler–Maclaurin as before.
//! The bounds' parameters are computed in `f64`, in `log₂` where mpmath's
//! 15-bit floats would underflow at high precision.

use super::wp::{RM, Wp};
use crate::base::bigfloat::to_f64;
use astro_float::{BigFloat, Consts};
use num_traits::Signed;

use crate::base::bigcomplex::{Complex, c_add, c_mul, c_zero};
use crate::base::errors::SymplexError;

const LN2: f64 = std::f64::consts::LN_2;
const LOG2E: f64 = std::f64::consts::LOG2_E;
const PI: f64 = std::f64::consts::PI;
/// `log₂ 2π`.
const LOG2_TWO_PI: f64 = 2.651_496_129_472_319;

/// Most corrections `L`, coefficients `J` and terms `N` this port takes
/// (`N = 100,000` is `t ≈ 6·10¹⁰`, about a second and a half; beyond,
/// refused).
const MAX_L: usize = 2_000;
const MAX_J: usize = 4_000;
const MAX_N: u64 = 100_000;

// ── f64 helpers ─────────────────────────────────────────────────────────────

/// `ln Γ(x)` for `x > 0` (Stirling's series after a shift to `x ≥ 10`).
fn lgamma(x: f64) -> f64 {
    let mut x = x;
    let mut acc = 0.0;
    while x < 10.0 {
        acc -= x.ln();
        x += 1.0;
    }
    let x2 = x * x;
    acc + (x - 0.5) * x.ln() - x + 0.5 * (2.0 * PI).ln() + 1.0 / (12.0 * x) - 1.0 / (360.0 * x * x2)
        + 1.0 / (1260.0 * x2 * x2 * x)
}

/// `log₂ abs(Γ(x))` for `x > 0`, and at `−1/2` (`Γ(−1/2) = −2√π`).
fn log2_gamma(x: f64) -> f64 {
    if x == -0.5 {
        return (2.0 * PI.sqrt()).log2();
    }
    lgamma(x) * LOG2E
}

/// mpmath's `mag` of a positive number given by its `log₂`: the `m` with
/// `2^(m−1) ≤ x < 2^m` (saturated far beyond any precision).
fn mag_l(l: f64) -> i64 {
    if l.is_nan() {
        return 1 << 40;
    }
    l.clamp(-1.0e12, 1.0e12).floor() as i64 + 1
}

/// `log₂(1 + 2^x)`, also where `2^x` overflows.
fn log2_1p_exp2(x: f64) -> f64 {
    if x > 60.0 {
        x + (-x).exp2() * LOG2E
    } else {
        x.exp2().ln_1p() * LOG2E
    }
}

/// mpmath's `mag` of a positive `f64`.
fn mag(x: f64) -> i64 {
    mag_l(x.abs().log2())
}

/// A working precision from mpmath's (possibly fractional) one.
fn prec_of(p: f64) -> usize {
    if p.is_finite() {
        p.max(16.0) as usize
    } else {
        16
    }
}

// ── BigFloat helpers ────────────────────────────────────────────────────────────────────

fn scale(z: &Complex, x: &BigFloat, p: usize) -> Complex {
    (z.0.mul(x, p, RM), z.1.mul(x, p, RM))
}

fn conj(z: &Complex) -> Complex {
    (z.0.clone(), z.1.neg())
}

fn abs_f64(z: &Complex) -> f64 {
    let a = to_f64(&z.0).unwrap_or(f64::INFINITY);
    let b = to_f64(&z.1).unwrap_or(f64::INFINITY);
    a.hypot(b)
}

/// `e^(iθ)` for a real `θ`.
fn expj(theta: &BigFloat, w: &mut Wp<'_>) -> Complex {
    let p = w.bits();
    (theta.cos(p, RM, w.cc), theta.sin(p, RM, w.cc))
}

/// `√(t/2π)` at `p` bits.
fn a_of(t: &BigFloat, w: &mut Wp<'_>) -> BigFloat {
    let p = w.bits();
    let two_pi =
        w.cc.pi(p + 8, RM)
            .clone()
            .mul(&BigFloat::from_i32(2, 64), p + 8, RM);
    t.div(&two_pi, p + 8, RM).sqrt(p, RM)
}

// ── The coefficients c_n of F (II, Sections 3.14–3.16) ──────────────────────

/// The Taylor coefficients `c_n` (`n < 2J + 4`) of
/// `F(z) = (e^(πi(z²/2 + 3/8)) − i√2·cos(πz/2))/(2cos πz)` with error below
/// `2^leps6`, and the powers `π^n`, `n ≤ 2J + 4` (mpmath's `_coef`).
fn coefficients(j: usize, leps6: f64, cc: &mut Consts) -> (Vec<Complex>, Vec<BigFloat>) {
    let newj = j + 2;
    let lneweps6 = leps6 - 1.0;
    let wpvw = mag(10.0 * (newj + 3) as f64).max(4 * newj as i64 + 5 - mag_l(lneweps6));
    let wppi = prec_of(mag(40.0 * newj as f64).max(mag(newj as f64) + 3 + wpvw) as f64);
    let pi = cc.pi(wppi, RM).clone();
    let mut pipow = Vec::with_capacity(2 * newj + 1);
    pipow.push(BigFloat::from_i32(1, wppi));
    for n in 1..=2 * newj {
        let next = pipow[n - 1].mul(&pi, wppi, RM);
        pipow.push(next);
    }
    // v(n) = (−1)^n E_{2n}/(2n)!·π^(2n), w(n) = π^n/(n!·2^n).
    let pv = prec_of((wpvw + 2) as f64);
    let mut v = Vec::with_capacity(newj + 1);
    let mut fact = BigFloat::from_i32(1, pv); // (2n)!
    for n in 0..=newj {
        if n > 0 {
            fact = fact
                .mul(&BigFloat::from_u64((2 * n - 1) as u64, 64), pv, RM)
                .mul(&BigFloat::from_u64((2 * n) as u64, 64), pv, RM);
        }
        let e = crate::base::combinatorics::euler_even(n);
        let e = crate::base::numeric::bigint_to_bigfloat(&e.abs(), pv);
        v.push(e.div(&fact, pv, RM).mul(&pipow[2 * n], pv, RM));
    }
    let mut w = Vec::with_capacity(2 * newj + 1);
    let mut f = BigFloat::from_i32(1, pv); // n!·2^n
    for (n, pn) in pipow.iter().enumerate() {
        if n > 0 {
            f = f.mul(&BigFloat::from_u64((2 * n) as u64, 64), pv, RM);
        }
        w.push(pn.div(&f, pv, RM));
    }
    // P1(n) = (−1)^(n+1)·i·Σ (−1)^k v(k) w(2n−2k), P2(n) = Σ i^(n−k) v(k) w(n−k).
    let wpp1a = 9 - mag_l(lneweps6);
    let mut p1 = Vec::with_capacity(newj + 1);
    let mut p2 = Vec::with_capacity(newj + 1);
    for n in 0..=newj {
        let wp = prec_of(mag(10.0 * (n + 4) as f64).max(4 * n as i64 + wpp1a) as f64);
        let mut s = BigFloat::new(wp);
        for k in 0..=n {
            let term = v[k].mul(&w[2 * n - 2 * k], wp, RM);
            s = if k % 2 == 0 {
                s.add(&term, wp, RM)
            } else {
                s.sub(&term, wp, RM)
            };
        }
        // (−1)^(n+1)·i·s
        let im = if n % 2 == 1 { s } else { s.neg() };
        p1.push((BigFloat::new(wp), im));
        let mut q = c_zero(wp);
        for k in 0..=n {
            let term = v[k].mul(&w[n - k], wp, RM);
            // i^(n−k)
            q = match (n - k) % 4 {
                0 => (q.0.add(&term, wp, RM), q.1),
                1 => (q.0, q.1.add(&term, wp, RM)),
                2 => (q.0.sub(&term, wp, RM), q.1),
                _ => (q.0, q.1.sub(&term, wp, RM)),
            };
        }
        p2.push(q);
    }
    // c(2n) = μ·P1(n) + ν·P2(n), μ = √2/2, ν = e^(3πi/8)/2.
    let wpc0 = 5 - mag_l(lneweps6);
    let wpc = prec_of(6i64.max(4 * newj as i64 + wpc0) as f64);
    let half = BigFloat::from_f64(0.5, 64);
    let mu = BigFloat::from_i32(2, 64).sqrt(wpc, RM).mul(&half, wpc, RM);
    let ang = cc
        .pi(wpc + 8, RM)
        .clone()
        .mul(&BigFloat::from_i32(3, 64), wpc + 8, RM)
        .div(&BigFloat::from_i32(8, 64), wpc + 8, RM);
    let nu = scale(&expj(&ang, &mut Wp::new(wpc, cc)), &half, wpc);
    let mut c = vec![c_zero(64); 2 * newj];
    for n in 0..newj {
        let wp = prec_of(6i64.max(4 * n as i64 + wpc0) as f64);
        c[2 * n] = c_add(&scale(&p1[n], &mu, wp), &c_mul(&nu, &p2[n], wp, RM), wp, RM);
    }
    (c, pipow)
}

// ── R(s) and conj R(1 − s̄) (I, Theorem 2; II, Sections 3.1–3.19) ────────────

/// The constants `b`, `c`, `A`, `B₁` of one side (I, (26) and (50)).
#[derive(Clone, Copy)]
struct Side {
    sigma: f64,
    b: f64,
    c: f64,
    big_a: f64,
    b1: f64,
}

fn side(sigma: f64) -> Side {
    if sigma > 0.0 {
        Side {
            sigma,
            b: 2.0,
            c: 9f64.powf(sigma) / 4.44288,
            big_a: 9f64.powf(sigma),
            b1: 1.0,
        }
    } else {
        Side {
            sigma,
            b: 2.25158,
            c: 2f64.powf(-sigma) / 4.44288,
            big_a: 2f64.powf(-sigma),
            b1: 1.10789,
        }
    }
}

/// The number `L` of Riemann–Siegel corrections (II, Section 3.2).
fn corrections(sd: &Side, leps2: f64, a: f64) -> usize {
    let mut l = 1usize;
    while l < MAX_L
        && 3f64.log2() + sd.c.log2() + log2_gamma(l as f64 * 0.5) - l as f64 * (sd.b * a).log2()
            >= leps2
    {
        l += 1;
    }
    l.max(2)
}

/// The number `M` of derivatives of `F` (II, (47), (48); `aux_M_Fp`).
fn derivatives(sd: &Side, leps4: f64, a: f64, l: usize) -> usize {
    let aux1 = (126.065_760_6 * sd.big_a).ln() - leps4 * LN2;
    let aux2 = (2.0 * PI.ln() + sd.b1.ln() + a.ln()) / 3.0 - (2.0 * PI).ln() / 2.0;
    let aux3 = |m: f64| (lgamma(m + 1.0) - lgamma(m / 3.0 + 2.0)) / 2.0 - lgamma((m + 1.0) / 2.0);
    let mut m = 3 * l - 3;
    while aux1 < m as f64 * aux2 + aux3(m as f64) && m > 1 {
        m -= 1;
    }
    m
}

/// `log₂` of the bound that decides `J` (II, (49); `aux_J_needed`).
fn j_bound(sd: &Side, leps4: f64, a: f64, m: usize) -> f64 {
    let h1 = leps4 - (632.0 * sd.big_a).log2();
    let h2c = sd.b1 * a * 126.313_374_195_292_6;
    let mf = m as f64;
    let h2 = h1 + (mf - 1.0) / 3.0 * (h2c / (mf * mf)).log2() - mf.log2();
    h1.min(h2)
}

/// The coefficients `d(n, k)`, `n < L`, `k ≤ 3n/2` (II, Section 3.17).
fn d_coefficients(sig: &BigFloat, sd: &Side, leps4: f64, a: f64, l: usize) -> Vec<Vec<BigFloat>> {
    let d1 = mag(40.0 * (l * l) as f64).max(6);
    let d2 = 13 + mag((1.0 + sd.sigma.abs()) * sd.big_a) - mag_l(leps4) - 1;
    let cnst = (8.0 / (PI * PI * a * a * sd.b1 * sd.b1)).ln() / 2.0;
    let wpd: Vec<usize> = (0..l)
        .map(|n| {
            let d3 =
                mag_l(0.5 * log2_gamma(n as f64 - 0.5)) - (n as f64 * cnst).floor() as i64 + d2;
            prec_of(d3.max(d1) as f64)
        })
        .collect();
    let mut d: Vec<Vec<BigFloat>> = Vec::with_capacity(l);
    d.push(vec![BigFloat::from_i32(1, 64)]);
    let get = |row: &Vec<BigFloat>, k: i64| -> Option<BigFloat> {
        usize::try_from(k).ok().and_then(|k| row.get(k).cloned())
    };
    for n in 1..l {
        let wp = wpd[n] + 10;
        let psig =
            BigFloat::from_i32(1, 64).sub(&sig.mul(&BigFloat::from_i32(2, 64), wp, RM), wp, RM);
        let kmax = 3 * n / 2;
        let mut row: Vec<BigFloat> = Vec::with_capacity(kmax + 1);
        for k in 0..=kmax {
            let m = 3 * n as i64 - 2 * k as i64;
            let prev = &d[n - 1];
            let v = if m != 0 {
                let mf = BigFloat::from_i64(m, 64);
                let mut acc = BigFloat::new(wp);
                if let Some(x) = get(prev, k as i64 - 2) {
                    acc = acc.sub(&x.mul(&BigFloat::from_i64(m + 1, 64), wp, RM), wp, RM);
                }
                if let Some(x) = get(prev, k as i64) {
                    let c1 = BigFloat::from_i32(1, 64).div(
                        &mf.mul(&BigFloat::from_i32(4, 64), wp, RM),
                        wp,
                        RM,
                    );
                    acc = acc.add(&x.mul(&c1, wp, RM), wp, RM);
                }
                if let Some(x) = get(prev, k as i64 - 1) {
                    let c2 = psig.div(&mf.mul(&BigFloat::from_i32(2, 64), wp, RM), wp, RM);
                    acc = acc.add(&x.mul(&c2, wp, RM), wp, RM);
                }
                acc
            } else {
                // −Σ_{r<k} (−1)^(k−r)·d(n, r)·(2k−2r)!/(k−r)!
                let mut acc = BigFloat::new(wp);
                for (r, dr) in row.iter().enumerate().take(k) {
                    let q = k - r;
                    let mut f = BigFloat::from_i32(1, wp);
                    for i in (q + 1)..=(2 * q) {
                        f = f.mul(&BigFloat::from_u64(i as u64, 64), wp, RM);
                    }
                    let add = dr.mul(&f, wp, RM);
                    acc = if q % 2 == 0 {
                        acc.sub(&add, wp, RM)
                    } else {
                        acc.add(&add, wp, RM)
                    };
                }
                acc
            };
            row.push(v);
        }
        d.push(row);
    }
    d
}

/// `(R(s), conj(R(1 − s̄)))` for `t > 0` with absolute errors below
/// `2^−wpinit` (`Rzeta_simul` at derivative 0), or `None` where the formula
/// does not reach that precision.
#[allow(clippy::too_many_lines)]
fn rzeta_pair(s: &Complex, wpc: &mut Wp<'_>) -> Result<Option<(Complex, Complex)>, SymplexError> {
    let wpinit = wpc.bits();
    let (Some(sig), Some(t)) = (to_f64(&s.0), to_f64(&s.1)) else {
        return Ok(None);
    };
    let a = (t / (2.0 * PI)).sqrt();
    let la = a.log2();
    let xs = side(sig);
    let ys = side(1.0 - sig);
    let leps = -(wpinit as f64);
    let leps1 = leps - 6f64.log2();
    let lxeps2 = leps + (sig * la).floor() - 3f64.log2();
    let lyeps2 = leps + ((1.0 - sig) * la).floor() - 3f64.log2();
    let xl = corrections(&xs, lxeps2, a);
    let yl = corrections(&ys, lyeps2, a);
    let bad = |l: usize, sg: f64| {
        l >= MAX_L
            || 3.0 * l as f64 >= 2.0 * a * a / 25.0
            || 3.0 * l as f64 + 2.0 + sg < 0.0
            || sg.abs() > a / 2.0
    };
    if bad(xl, sig) || bad(yl, 1.0 - sig) {
        return Ok(None);
    }
    let l = xl.max(yl);
    let lxeps3 = lxeps2 - (4.0 * xl as f64).log2();
    let lyeps3 = lyeps2 - (4.0 * yl as f64).log2();
    let lxeps4 = lxeps3 - (3.0 * xl as f64).log2();
    let lyeps4 = lyeps3 - (3.0 * yl as f64).log2();
    let xm = derivatives(&xs, lxeps4, a, xl);
    let ym = derivatives(&ys, lyeps4, a, yl);
    let m = xm.max(ym);
    let lh3 = j_bound(&xs, lxeps4, a, xm).min(j_bound(&ys, lyeps4, a, ym));
    let mut j = 12usize;
    let mut ljv = 12.0 * LOG2_TWO_PI - log2_gamma(13.0);
    while ljv > lh3 && j < MAX_J {
        j += 1;
        ljv += LOG2_TWO_PI - (j as f64).log2();
    }
    // The powers of π go to 2J + 4 and the corrections need them to 2L − 2
    // (mpmath's "we need J ≥ L").
    if j >= MAX_J || 2 * j <= m + 1 || l > j + 3 {
        return Ok(None);
    }
    // eps5(m) and the precision of the derivatives of F (II, (43), (59)).
    let leps5: Vec<f64> = (0..22)
        .map(|mm| {
            let mmf = mm as f64;
            let aux1 =
                |sd: &Side| mmf / 3.0 * (PI * PI * sd.b1 * a).log2() - (316.0 * sd.big_a).log2();
            let aux2 = 0.5 * (log2_gamma(mmf + 1.0) - log2_gamma(mmf / 3.0 + 0.5));
            aux1(&xs).min(aux1(&ys)) + aux2 + lxeps4.min(lyeps4)
        })
        .collect();
    let twenty = (3 * l - 3).min(21) + 1;
    let mut wpfp = mag(44.0 * j as f64);
    for (mm, le) in leps5.iter().enumerate().take(twenty) {
        wpfp = wpfp.max(mag_l(
            (6812.0 * j as f64).log2() + log2_gamma(mm as f64 + 1.0) - le,
        ));
    }
    let wpfp = prec_of(wpfp as f64);
    // N and p.
    let pa = wpfp + prec_of(mag(t) as f64) + 20;
    let a_bf = a_of(&s.1, &mut wpc.at(pa));
    let n_bf = a_bf.floor();
    let Some(n) = to_f64(&n_bf)
        .map(|v| v as u64)
        .filter(|&n| (2..=MAX_N).contains(&n))
    else {
        return Ok(None);
    };
    let frac = a_bf.sub(&n_bf, pa, RM);
    let p_full =
        BigFloat::from_i32(1, 64).sub(&frac.mul(&BigFloat::from_i32(2, 64), pa, RM), pa, RM);
    // p rounded to a multiple of 2^−wpfp.
    let mut scaled = p_full.clone();
    if let Some(e) = scaled.exponent() {
        scaled.set_exponent(e + i32::try_from(wpfp).unwrap_or(i32::MAX / 4));
    }
    let rounded = scaled.add(&BigFloat::from_f64(0.5, 64), pa + 8, RM).floor();
    let mut p = rounded;
    if let Some(e) = p.exponent() {
        p.set_exponent(e - i32::try_from(wpfp).unwrap_or(i32::MAX / 4));
    }
    // c_n and π^n, then F^(m)(p), m ≤ M (II, Sections 3.13, 3.14).
    let leps6 = j as f64 * LOG2_TWO_PI - log2_gamma(j as f64 + 1.0) - (3.0 * j as f64).log2();
    let (c, pipow) = coefficients(j, leps6, wpc.cc);
    let mut ccw = c;
    let mut fp: Vec<Complex> = Vec::with_capacity(m + 1);
    for mm in 0..=m {
        let mut sum = c_zero(wpfp);
        for k in (0..(2 * j - mm)).rev() {
            sum = c_add(&scale(&sum, &p, wpfp), &ccw[k], wpfp, RM);
        }
        fp.push(sum);
        for k in 0..(2 * j - mm - 1) {
            ccw[k] = scale(&ccw[k + 1], &BigFloat::from_u64(k as u64 + 1, 64), wpfp);
        }
    }
    // The working precisions of the corrections (II, Section 3.9).
    let c1 = mag(40.0 * (l + 2) as f64);
    let wps = |sd: &Side, leps3: f64, leps4: f64, xside: bool| -> (Vec<usize>, Vec<usize>) {
        let c2 = mag(68.0 * (l + 2) as f64 * sd.big_a);
        let c4 = mag(sd.b1 * a * PI.sqrt()) - 1;
        let mut tcoef = Vec::with_capacity(l);
        let mut term = Vec::with_capacity(l);
        for k in 0..l {
            let c3 = (c2 - k as i64 * c4) as f64 + mag_l(log2_gamma(k as f64 + 1.5)) as f64 / 2.0;
            let base = (c1 as f64).max(c3 - mag_l(leps4) as f64 + 1.0);
            let tc = if xside {
                (base + 1.0 + 20.0) * 1.5
            } else {
                (base + 10.0) * 1.5
            };
            let base_t =
                (c1 as f64).max(mag((l + 2) as f64) as f64 + c3 - mag_l(leps3) as f64 + 1.0);
            let tt = if xside {
                base_t + 1.0 + 20.0
            } else {
                base_t + 1.0 + 10.0
            };
            tcoef.push(prec_of(tc));
            term.push(prec_of(tt));
        }
        (tcoef, term)
    };
    let (xwptcoef, xwpterm) = wps(&xs, lxeps3, lxeps4, true);
    let (ywptcoef, ywpterm) = wps(&ys, lyeps3, lyeps4, false);
    // a^(−k).
    let wpa = xwptcoef[0].max(ywptcoef[0]);
    let av1 = BigFloat::from_i32(1, 64).div(&a_bf, wpa + 2, RM);
    let mut av = vec![BigFloat::from_i32(1, 64), av1.clone()];
    for k in 2..l {
        let next = av[k - 1].mul(&av1, wpa, RM);
        av.push(next);
    }
    // The sum of the corrections of one side (II, Sections 3.5–3.8).
    let rssum = |sig_bf: &BigFloat,
                 sd: &Side,
                 leps2: f64,
                 leps4: f64,
                 wptcoef: &[usize],
                 wpterm: &[usize]|
     -> Complex {
        let d = d_coefficients(sig_bf, sd, leps4, a, l);
        let mut terms: Vec<Complex> = Vec::with_capacity(l);
        for k in 0..l {
            let wp = wptcoef[k];
            let mut te = c_zero(wpterm[k]);
            for (ell, dk) in d[k].iter().enumerate() {
                // d(k, ℓ)·F^(3k−2ℓ)(p)/π^(2k−ℓ)/(2i)^ℓ·a^(−k); the derivatives
                // beyond M are below the bound and taken as 0 (mpmath sets
                // them so, II Section 3.11).
                let Some(f) = fp.get(3 * k - 2 * ell) else {
                    continue;
                };
                let v = scale(f, &dk.div(&pipow[2 * k - ell], wp, RM), wp);
                // /(2i)^ℓ = ·(−i)^ℓ·2^(−ℓ).
                let mut v = match ell % 4 {
                    0 => v,
                    1 => (v.1.clone(), v.0.neg()),
                    2 => (v.0.neg(), v.1.neg()),
                    _ => (v.1.neg(), v.0.clone()),
                };
                for x in [&mut v.0, &mut v.1] {
                    if let Some(e) = x.exponent() {
                        x.set_exponent(e - ell as i32);
                    }
                }
                let v = scale(&v, &av[k], wp);
                te = c_add(&te, &v, wpterm[k], RM);
            }
            terms.push(te);
        }
        let rsbound = PI.sqrt() * sd.c / (sd.b * a);
        let wprssum = prec_of(
            mag_l((4.4 * ((l + 3) * (l + 3)) as f64).log2() + rsbound.log2() - leps2)
                .max(mag(10.0 * (l + 1) as f64)) as f64,
        );
        let mut sum = c_zero(wprssum);
        for k in 1..=l {
            sum = c_add(&sum, &terms[l - k], wprssum, RM);
        }
        sum
    };
    let ysig_bf = BigFloat::from_i32(1, 64).sub(&s.0, super::exact_bits(&s.0, wpinit), RM);
    let xr = rssum(&s.0, &xs, lxeps2, lxeps4, &xwptcoef, &xwpterm);
    let yr = rssum(&ysig_bf, &ys, lyeps2, lyeps4, &ywptcoef, &ywpterm);
    // S3 = (−1)^(N−1)·a^(−σ)·U (II, Section 3.19).
    let la2 = mag(abs_f64(&xr)).max(mag(abs_f64(&yr))) as f64;
    let leps8 = leps - 3f64.log2() - la2;
    let big_t = t * (t / (2.0 * PI)).ln();
    // 5 + mag((1 + (2/eps8)·a^(−σ))·T), in log₂ (`eps8` far below the `f64`
    // range at high precision).
    let wps3 = |sg: f64| 5 + mag_l(big_t.log2() + log2_1p_exp2(1.0 - leps8 - sg * la));
    let ps3 = prec_of(wps3(sig).max(wps3(1.0 - sig)) as f64);
    let tb = &s.1;
    let two_pi = wpc
        .cc
        .pi(ps3 + 8, RM)
        .clone()
        .mul(&BigFloat::from_i32(2, 64), ps3 + 8, RM);
    let tpi = tb.div(&two_pi, ps3 + 8, RM);
    let half_t = tb.div(&BigFloat::from_i32(2, 64), ps3 + 8, RM);
    let pi8 = wpc
        .cc
        .pi(ps3 + 8, RM)
        .clone()
        .div(&BigFloat::from_i32(8, 64), ps3 + 8, RM);
    let arg = half_t
        .mul(&tpi.ln(ps3 + 8, RM, wpc.cc), ps3 + 8, RM)
        .sub(&half_t, ps3 + 8, RM)
        .sub(&pi8, ps3 + 8, RM);
    let u = expj(&arg.neg(), &mut wpc.at(ps3));
    let a3 = a_of(tb, &mut wpc.at(ps3));
    let lna = a3.ln(ps3, RM, wpc.cc);
    let sign_odd = n % 2 == 0; // (−1)^(N−1)
    let s3 = |sg: &BigFloat, cc: &mut Consts| {
        let pw = sg.mul(&lna, ps3, RM).neg().exp(ps3, RM, cc);
        let v = scale(&u, &pw, ps3);
        if sign_odd { (v.0.neg(), v.1.neg()) } else { v }
    };
    let xs3 = s3(&s.0, wpc.cc);
    let ys3 = s3(&ysig_bf, wpc.cc);
    // S1 = Σ_{w≤N} w^(−s) and Σ_{w≤N} w^(σ−1)e^(−it ln w) (II, Section 3.18).
    let nf = n as f64;
    let wpsum = |sg: f64| 4 + mag_l((nf + nf.powf(1.0 - sg)).log2() + nf.ln().log2() - leps1);
    let ws = prec_of(wpsum(sig).max(wpsum(1.0 - sig)) as f64) + 10;
    // w^(−s) for a prime w from its logarithm, for a composite one the
    // product of two earlier values (a smallest-prime-factor sieve; at most
    // log₂ N roundings per value, covered by the guard bits); the second
    // sum's terms are w^(−s)·w^(2σ−1), multiplicative likewise.
    let wq = ws + 4 + 2 * super::accuracy::ceil_log2(64 - n.leading_zeros() as usize + 1) as usize;
    let lt = super::magnitude_bits(t * nf.ln()) + 8;
    let nn = n as usize;
    let mut spf = vec![0u32; nn + 1];
    for i in 2..=nn {
        if spf[i] == 0 {
            let mut k = i;
            while k <= nn {
                if spf[k] == 0 {
                    spf[k] = i as u32;
                }
                k += i;
            }
        }
    }
    let two_sig_m1 =
        s.0.mul(&BigFloat::from_i32(2, 64), wq, RM)
            .sub(&BigFloat::from_i32(1, 64), wq, RM);
    let mut xw: Vec<Complex> = Vec::with_capacity(nn + 1);
    let mut rw: Vec<BigFloat> = Vec::with_capacity(nn + 1);
    xw.push(c_zero(64));
    rw.push(BigFloat::new(64));
    xw.push((BigFloat::from_i32(1, 64), BigFloat::new(64)));
    rw.push(BigFloat::from_i32(1, 64));
    let mut xs1 = (BigFloat::from_i32(1, ws), BigFloat::new(ws));
    let mut ys1 = (BigFloat::from_i32(1, ws), BigFloat::new(ws));
    for (w, &pr) in spf.iter().enumerate().skip(2) {
        let pr = pr as usize;
        let (x, r) = if pr == w {
            let lw = BigFloat::from_u64(w as u64, 64).ln(wq + lt, RM, wpc.cc);
            let ph = tb.mul(&lw, wq + lt, RM).neg();
            let e = expj(&ph, &mut wpc.at(wq));
            let mag_x = s.0.mul(&lw, wq, RM).neg().exp(wq, RM, wpc.cc);
            let r = two_sig_m1.mul(&lw, wq, RM).exp(wq, RM, wpc.cc);
            (scale(&e, &mag_x, wq), r)
        } else {
            let q = w / pr;
            (c_mul(&xw[pr], &xw[q], wq, RM), rw[pr].mul(&rw[q], wq, RM))
        };
        xs1 = c_add(&xs1, &x, ws, RM);
        ys1 = c_add(&ys1, &scale(&x, &r, wq), ws, RM);
        xw.push(x);
        rw.push(r);
    }
    // R(s) = S1 + rssum·S3, conj R(1 − s̄) likewise (II, Section 3.1).
    let end = |s1: &Complex, rs: &Complex, s3v: &Complex| {
        let p =
            prec_of(6i64.max(
                wpinit as i64 + mag(6.0 * (3.0 * abs_f64(s1) + 7.0 * abs_f64(rs) * abs_f64(s3v))),
            ) as f64);
        c_add(s1, &c_mul(rs, s3v, p, RM), p, RM)
    };
    let xrz = end(&xs1, &xr, &xs3);
    let yrz = conj(&end(&ys1, &yr, &ys3));
    Ok(Some((xrz, yrz)))
}

/// `ζ(s)` by the Riemann–Siegel formula with absolute error below `2^−wp`
/// (`zeta_offline` at derivative 0), or `None` where it is not used.
pub(super) fn zeta_rs(s: &Complex, w: &mut Wp<'_>) -> Result<Option<Complex>, SymplexError> {
    // ζ(s̄) = conj ζ(s): below the real axis by the conjugate.
    let below = s.1.is_negative();
    let upper = if below { conj(s) } else { s.clone() };
    let v = zeta_upper(&upper, w)?;
    Ok(v.map(|v| if below { conj(&v) } else { v }))
}

/// [`zeta_rs`] for `Im s > 0`.
fn zeta_upper(s: &Complex, w: &mut Wp<'_>) -> Result<Option<Complex>, SymplexError> {
    let wp = w.bits();
    let (Some(sig), Some(t)) = (to_f64(&s.0), to_f64(&s.1)) else {
        return Ok(None);
    };
    if t.is_nan() || t <= 0.0 {
        return Ok(None);
    }
    let abs_s = sig.hypot(t);
    let abs_1s = (1.0 - sig).hypot(t);
    let x = if sig > 0.0 {
        abs_s.sqrt()
    } else {
        (2.0 * PI).powf(sig - 1.0) * abs_1s.powf(0.5 - sig)
    };
    let m1 = if sig > 0.0 {
        2.0 * (t / (2.0 * PI)).sqrt()
    } else {
        4.0 * t * x
    };
    let m2 = if 1.0 - sig > 0.0 {
        2.0 * (t / (2.0 * PI)).sqrt()
    } else {
        4.0 * t * (2.0 * PI).powf(-sig) * abs_s.powf(sig - 0.5)
    };
    let abst = (0.5 - sig).hypot(t);
    let big_t = 2.0 * abst * abst.ln();
    let wpbasic = (3 + mag(t))
        .max(6)
        .max(2 + mag(2.12 * m1 + 21.2 * m2 * x + 1.3 * m2 * x * big_t) + wp as i64 + 1);
    let wptheta = 4i64.max(3 + mag(2.7 * m2 * x) + wp as i64 + 1);
    let wpr = 3 + mag(1.1 + 2.0 * x) + wp as i64 + 1;
    if [wpbasic, wptheta, wpr]
        .iter()
        .any(|&v| v <= 0 || v > 1 << 24)
    {
        return Ok(None);
    }
    let (wpbasic, wptheta, wpr) = (wpbasic as usize, wptheta as usize, wpr as usize);
    let Some((xrz, yrz)) = rzeta_pair(s, &mut w.at(wpr))? else {
        return Ok(None);
    };
    // θ = ϑ(t − i(σ − 1/2)) = −(ln π/2)(t − i(σ − 1/2)) − (i/2)(ln Γ(s/2) − ln Γ((1 − s)/2)).
    let lg_size = super::magnitude_bits(abst * (abst.ln() + 2.0)) + 16;
    let pg = wptheta + lg_size;
    let exact = super::exact_bits(&s.0, pg);
    let two = BigFloat::from_i32(2, 64);
    let s2 = (s.0.div(&two, exact, RM), s.1.div(&two, exact, RM));
    let one_minus = (
        BigFloat::from_i32(1, 64)
            .sub(&s.0, exact, RM)
            .div(&two, exact, RM),
        s.1.neg().div(&two, exact, RM),
    );
    let g1 = super::loggamma::loggamma_complex(&s2, &mut w.at(pg))?;
    let g2 = super::loggamma::loggamma_complex(&one_minus, &mut w.at(pg))?;
    let dg = (g1.0.sub(&g2.0, pg, RM), g1.1.sub(&g2.1, pg, RM));
    // −(i/2)·dg = (dg.im/2, −dg.re/2).
    let half = BigFloat::from_f64(0.5, 64);
    let part = (dg.1.mul(&half, pg, RM), dg.0.neg().mul(&half, pg, RM));
    let lnpi2 = w.cc.pi(pg, RM).clone().ln(pg, RM, w.cc).mul(&half, pg, RM);
    let delta = s.0.sub(&half, exact, RM);
    // −(ln π/2)·(t − iδ) = (−(ln π/2)t, (ln π/2)δ).
    let lin = (s.1.mul(&lnpi2, pg, RM).neg(), delta.mul(&lnpi2, pg, RM));
    let theta = c_add(&lin, &part, pg, RM);
    // e^(−2iθ) = e^(2·Im θ)·e^(−2i·Re θ).
    let pe = wpbasic + super::magnitude_bits(to_f64(&theta.0).unwrap_or(1e18).abs() * 2.0) + 8;
    let modulus = theta.1.mul(&two, pe, RM).exp(pe, RM, w.cc);
    let ph = theta.0.mul(&two, pe, RM).neg();
    let et = scale(&expj(&ph, &mut w.at(pe)), &modulus, pe);
    let zv = c_add(&xrz, &c_mul(&et, &yrz, wpbasic, RM), wpbasic, RM);
    Ok(Some(zv))
}

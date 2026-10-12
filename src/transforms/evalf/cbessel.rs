//! The Bessel functions `J_ν`, `Y_ν`, `I_ν`, `K_ν` and the Airy functions
//! at non-real arguments or orders (and at a negative argument where the
//! value is not real).  (The real routines are `arb_bessel_j`, …, `arb_airy`
//! in `evalf.rs`.)  Before 0.45 they were refused: `besselj(1/3, 2 + 3i)`,
//! "Bessel of complex argument not yet supported in evalf", `airyai(1 + i)`.
//!
//! Principal branches, as mpmath and SymPy: `(z/2)^ν` and `ln(z/2)` cut
//! along `(−∞, 0]`, a negative real argument taken from above.
//!
//! # Power series (small and moderate `abs(z)`)
//!
//! * `J_ν(z), I_ν(z) = (z/2)^ν Σ_k (∓z²/4)^k/(k!·Γ(ν + k + 1))` (DLMF 10.2.2,
//!   10.25.2), `J_{−n} = (−1)^n J_n`, `I_{−n} = I_n`; the rest after a term is
//!   at most the next one over `1 − q`, `q = abs(z)²/(4(k + 2)·d)`, `d =
//!   max(Re ν + k + 2, abs(Im ν))`, once `q < 1`.
//! * `Y_ν = (J_ν cos νπ − J_{−ν})/sin νπ`, `K_ν = (π/2)(I_{−ν} − I_ν)/sin νπ`
//!   for a non-integer `ν` (DLMF 10.2.3, 10.27.4), `νπ` reduced exactly; near
//!   an integer the cancellation is made up with more bits.
//! * an integer order: DLMF 10.8.1 and 10.31.1 (with `ψ(k+1) = H_k − γ`),
//!   the rest of the `ψ`-weighted sum bounded with the ratio `q·(1 + 2/(k +
//!   2))` (`H_{j+1} + H_{n+j+1} ≤ (1 + 2/(j+1))(H_j + H_{n+j})`).
//! * Airy: `Ai = c₁f − c₂g`, `Bi = √3(c₁f + c₂g)` and their derivatives from
//!   the series `f`, `g` of DLMF 9.4.1–9.4.4, `c₁ = 3^(−2/3)/Γ(2/3)`, `c₂ =
//!   3^(−1/3)/Γ(1/3)`.
//!
//! # Hankel's expansion (large `abs(z)`)
//!
//! `K_ν(w) = √(π/(2w)) e^(−w) (Σ_{k<ℓ} a_k(ν) w^(−k) + R_ℓ)`, `a_k(ν) = (4ν² −
//! 1²)(4ν² − 3²)⋯(4ν² − (2k−1)²)/(k!·8^k)`, with Olver's bound (DLMF
//! 10.40.10–10.40.12, complex `ν` and `w`)
//!
//! ```text
//! abs(R_ℓ) ≤ 2·abs(a_ℓ(ν))·V_ℓ·exp(abs(ν² − 1/4)·V_1),
//! V_ℓ = abs(w)^(−ℓ) for abs(ph w) ≤ π/2, χ(ℓ)·abs(w)^(−ℓ) up to π,
//!       2χ(ℓ)·abs(Re w)^(−ℓ) from π to 3π/2 (the continuation),
//! ```
//!
//! `χ(ℓ) = √π·Γ(ℓ/2 + 1)/Γ(ℓ/2 + 1/2) < √(π(ℓ/2 + 1))` (Gautschi's
//! inequality).  It is used where that bound reaches `2^−wp`; then
//!
//! * `I_ν(z) = (K_ν(−z) − e^(iνπ)K_ν(z))/(πi)` above the real axis (and on
//!   the negative axis), `(e^(−iνπ)K_ν(z) − K_ν(−z))/(πi)` below (DLMF
//!   10.34.2 with `m = ∓1`);
//! * `H⁽¹⁾_ν(z) = (2/(πi))e^(−iνπ/2)K_ν(−iz)`, `H⁽²⁾_ν(z) =
//!   −(2/(πi))e^(iνπ/2)K_ν(iz)` in the right half-plane (DLMF 10.27.8), `J =
//!   (H⁽¹⁾ + H⁽²⁾)/2`, `Y = (H⁽¹⁾ − H⁽²⁾)/(2i)`, and in the left half-plane
//!   `J_ν(z) = e^(±iνπ)J_ν(−z)`, `Y_ν(z) = e^(∓iνπ)Y_ν(−z) ± 2i·cos(νπ)·J_ν(−z)`
//!   (DLMF 10.11.1, 10.11.2; upper signs above the axis);
//! * `Ai(w) = w^(−1/4)e^(−ζ)/(2√π)·Σ a_k(1/3)ζ^(−k)` and `Ai′(w) =
//!   −w^(1/4)e^(−ζ)/(2√π)·Σ a_k(2/3)ζ^(−k)`, `ζ = (2/3)w^(3/2)` (DLMF 9.6.1,
//!   9.6.2) for `abs(ph w) ≤ 0.7π` (`abs(ph ζ) ≤ 1.05π`); beyond `0.65π`
//!   through `Ai(z) = −e^(2πi/3)Ai(ze^(2πi/3)) − e^(−2πi/3)Ai(ze^(−2πi/3))`
//!   (DLMF 9.2.12), and `Bi(z) = e^(πi/6)Ai(ze^(2πi/3)) +
//!   e^(−πi/6)Ai(ze^(−2πi/3))` (DLMF 9.2.10), the derivatives by
//!   differentiating them.
//!
//! The roundings and the loss to cancellation are accounted for as in
//! `cexpint.rs`.  The reductions to `K_ν` and the connection formulas follow
//! mpmath (`mpmath/functions/bessel.py`, BSD), which uses the same
//! hypergeometric series and asymptotic `2F0` expansions; Olver's bound
//! replaces its size test.

use astro_float::{BigFloat, Consts, RoundingMode};

use super::AiryKind;
use super::accuracy;
use super::cexpint::{budget, f64_parts, guard};
use super::zeta::{Sum, plus_int};
use crate::base::bigcomplex::{Complex, c_add, c_div, c_mul, c_neg, c_one, c_sub, c_zero};
use crate::base::errors::SymplexError;

const LOG2E: f64 = std::f64::consts::LOG2_E;
const PI: f64 = std::f64::consts::PI;
const HALF_PI: f64 = std::f64::consts::FRAC_PI_2;

/// Most terms of Hankel's expansion.
const MAX_ASYMPTOTIC: usize = 100_000;

/// Which Bessel function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    J,
    Y,
    I,
    K,
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn to_f64(x: &BigFloat) -> Option<f64> {
    super::bigfloat_to_f64_rounded(x, RoundingMode::ToEven)
        .ok()
        .filter(|v| v.is_finite())
}

/// The parts of `z` in `f64` (a zero imaginary part as `+0`).
fn parts(z: &Complex) -> Option<(f64, f64)> {
    f64_parts(z).map(|p| (p.x, p.y))
}

fn round_c(v: Complex, prec: usize, rm: RoundingMode) -> Complex {
    (
        super::round_to(v.0, prec, rm),
        super::round_to(v.1, prec, rm),
    )
}

fn is_zero(z: &Complex) -> bool {
    z.0.is_zero() && z.1.is_zero()
}

fn add_term(sum: &mut Sum, t: &Complex, p: usize, rm: RoundingMode) {
    sum.largest = sum.largest.max(accuracy::lg_abs(t));
    sum.value = c_add(&sum.value, t, p, rm);
}

fn new_sum(p: usize) -> Sum {
    Sum {
        value: c_zero(p),
        largest: f64::NEG_INFINITY,
    }
}

/// `z/n`, `z·n` for an integer `n`.
fn div_u(z: &Complex, n: u64, p: usize, rm: RoundingMode) -> Complex {
    let d = BigFloat::from_u64(n, 64);
    (z.0.div(&d, p, rm), z.1.div(&d, p, rm))
}

/// `z·x` for a real `x`.
fn scale(z: &Complex, x: &BigFloat, p: usize, rm: RoundingMode) -> Complex {
    (z.0.mul(x, p, rm), z.1.mul(x, p, rm))
}

/// `z·2^k` exactly.
fn ldexp(z: &Complex, k: i32) -> Complex {
    let mut v = z.clone();
    for x in [&mut v.0, &mut v.1] {
        if let Some(e) = x.exponent() {
            x.set_exponent(e + k);
        }
    }
    v
}

fn times_i(z: &Complex) -> Complex {
    (z.1.neg(), z.0.clone())
}

fn times_minus_i(z: &Complex) -> Complex {
    (z.1.clone(), z.0.neg())
}

/// `ν` as an integer exactly.
fn exact_int(nu: &Complex) -> Option<i64> {
    if !nu.1.is_zero() || !(nu.0.is_zero() || nu.0.is_int()) {
        return None;
    }
    let v = to_f64(&nu.0)?;
    (v.abs() < 1.0e9).then_some(v as i64)
}

/// The phase of `z` in `f64`, a negative real from above (`π`).
fn phase(z: &Complex) -> Option<f64> {
    let (x, y) = parts(z)?;
    Some(y.atan2(x))
}

/// Above the real axis, or on its negative half (from above)?
fn upper(z: &Complex) -> bool {
    if z.1.is_zero() {
        z.0.is_negative()
    } else {
        z.1.is_positive()
    }
}

/// `e^(iπu)` for a complex `u`, `Re u` reduced exactly: `(−1)^n e^(−π Im u)
/// (cos πd + i sin πd)`, `u = n + d + i·Im u`.
fn exp_i_pi(u: &Complex, p: usize, rm: RoundingMode, cc: &mut Consts) -> Complex {
    let exact = super::exact_bits(&u.0, p);
    let n = u.0.add(&BigFloat::from_f64(0.5, 64), exact, rm).floor();
    let d = u.0.sub(&n, exact, rm);
    let q = p + 8 + super::magnitude_bits(to_f64(&u.1).map_or(1e9, |v| v.abs() * 4.6));
    let pi = cc.pi(q, rm).clone();
    let e = (u.1.mul(&pi, q, rm).neg(), d.mul(&pi, q, rm));
    let v = super::c_exp(&e, q, rm, cc);
    let odd = !n
        .div(&BigFloat::from_i32(2, 64), exact, rm)
        .fract()
        .is_zero();
    round_c(if odd { c_neg(&v) } else { v }, p, rm)
}

/// `(sin νπ, cos νπ)`, `Re ν` reduced exactly.
fn sin_cos_pi(nu: &Complex, p: usize, rm: RoundingMode, cc: &mut Consts) -> (Complex, Complex) {
    // e^(±iνπ): sin = (e^(iνπ) − e^(−iνπ))/(2i), cos = (e^(iνπ) + e^(−iνπ))/2.
    // Formed from the exactly reduced angle instead, to keep sin's relative
    // accuracy next to an integer.
    let exact = super::exact_bits(&nu.0, p);
    let n = nu.0.add(&BigFloat::from_f64(0.5, 64), exact, rm).floor();
    let d = nu.0.sub(&n, exact, rm);
    let q = p + 8 + super::magnitude_bits(to_f64(&nu.1).map_or(1e9, |v| v.abs() * 4.6));
    let pi = cc.pi(q, rm).clone();
    let a = (d.mul(&pi, q, rm), nu.1.mul(&pi, q, rm));
    let s = super::c_sin(&a, q, rm, cc);
    let c = super::c_cos(&a, q, rm, cc);
    let odd = !n
        .div(&BigFloat::from_i32(2, 64), exact, rm)
        .fract()
        .is_zero();
    if odd {
        (round_c(c_neg(&s), p, rm), round_c(c_neg(&c), p, rm))
    } else {
        (round_c(s, p, rm), round_c(c, p, rm))
    }
}

/// `1/Γ(w)` for `w` off the poles, to `p` bits relative.
fn rgamma(
    w: &Complex,
    p: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<Complex, SymplexError> {
    let g = if w.1.is_zero() {
        let wf = to_f64(&w.0).unwrap_or(f64::INFINITY).abs();
        let gb = super::magnitude_bits((wf + 2.0) * ((wf + 2.0).ln() + 1.0));
        (
            super::arb_gamma_real(&w.0, p + gb + 16, rm, cc)?,
            BigFloat::new(p),
        )
    } else {
        super::zeta::gamma_complex(w, p + 8, rm, cc)?
    };
    Ok(round_c(c_div(&c_one(p + 8), &g, p + 8, rm), p, rm))
}

/// `1/n!`.
fn inv_factorial(n: u64, p: usize, rm: RoundingMode) -> BigFloat {
    let q = p + guard(n as usize + 2);
    let mut f = BigFloat::from_i32(1, q);
    for k in 2..=n {
        f = f.mul(&BigFloat::from_u64(k, 64), q, rm);
    }
    BigFloat::from_i32(1, q).div(&f, p, rm)
}

// ── Power series ────────────────────────────────────────────────────────────

/// `J_ν(z)` (`modified` false) or `I_ν(z)` by the power series, `z ≠ 0`.
fn ji_series(
    nu: &Complex,
    z: &Complex,
    modified: bool,
    wp: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<Sum, SymplexError> {
    let (Some((x, y)), Some((nr, ni))) = (parts(z), parts(nu)) else {
        return Err(super::unevaluable("Bessel argument beyond f64"));
    };
    let abs_z = x.hypot(y);
    let max = budget(abs_z, nr.hypot(ni), wp);
    let p = wp + guard(max);
    let h = ldexp(z, -1);
    let int = exact_int(nu);
    let (order, pref, t0, sign) = match int {
        Some(n) => {
            let m = n.unsigned_abs();
            let pw = crate::base::bigcomplex::c_powi(&h, m as usize, p + guard(m as usize + 2), rm);
            let t0 = (inv_factorial(m, p, rm), BigFloat::new(p));
            let negate = n < 0 && !modified && m % 2 == 1;
            (
                (BigFloat::from_u64(m, 64), BigFloat::new(64)),
                round_c(pw, p, rm),
                t0,
                negate,
            )
        }
        None => {
            let nu1 = (plus_int(&nu.0, 1, p, rm), nu.1.clone());
            let t0 = rgamma(&nu1, p, rm, cc)?;
            let pref = super::cexpint::power_exp(&h, nu, false, p, rm, cc);
            (nu.clone(), pref, t0, false)
        }
    };
    let (or, oi) = parts(&order).unwrap_or((nr, ni));
    let z2 = ldexp(&c_mul(z, z, p, rm), -2);
    let w = if modified { z2 } else { c_neg(&z2) };
    let aw = abs_z * abs_z / 4.0;
    let mut sum = new_sum(p);
    let mut t = t0;
    let mut converged = false;
    for k in 0..max as u64 {
        add_term(&mut sum, &t, p, rm);
        let d = (plus_int(&order.0, k as i64 + 1, p, rm), order.1.clone());
        t = div_u(&c_div(&c_mul(&t, &w, p, rm), &d, p, rm), k + 1, p, rm);
        let dmin = (or + k as f64 + 2.0).max(oi.abs());
        let q = aw / ((k as f64 + 2.0) * dmin);
        if dmin > 0.0 && q < 1.0 {
            let rest = accuracy::lg_abs(&t) - (1.0 - q).log2();
            if rest < sum.largest - wp as f64 || is_zero(&t) {
                converged = true;
                break;
            }
        }
    }
    if !converged {
        return Err(super::series_did_not_converge("Bessel", max));
    }
    let mut v = c_mul(&pref, &sum.value, p, rm);
    if sign {
        v = c_neg(&v);
    }
    Ok(Sum {
        largest: accuracy::lg_abs(&pref) + sum.largest,
        value: v,
    })
}

/// `Y_ν` or `K_ν` of a non-integer order from `J_{±ν}`, `I_{±ν}`.
fn yk_noninteger(
    kind: Kind,
    nu: &Complex,
    z: &Complex,
    wp: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<Sum, SymplexError> {
    let modified = kind == Kind::K;
    let p = wp + 8;
    let plus = ji_series(nu, z, modified, p, rm, cc)?;
    let minus = ji_series(&c_neg(nu), z, modified, p, rm, cc)?;
    let (s, c) = sin_cos_pi(nu, p, rm, cc);
    let ls = accuracy::lg_abs_low(&s);
    if !ls.is_finite() {
        return Err(super::unevaluable("Bessel: sin(νπ) vanished"));
    }
    let (num, largest) = if modified {
        // (π/2)(I_{−ν} − I_ν)
        let pi2 = ldexp(&(cc.pi(p, rm).clone(), BigFloat::new(p)), -1);
        let d = c_sub(&minus.value, &plus.value, p, rm);
        (
            c_mul(&pi2, &d, p, rm),
            plus.largest.max(minus.largest) + 0.66,
        )
    } else {
        let a = c_mul(&plus.value, &c, p, rm);
        (
            c_sub(&a, &minus.value, p, rm),
            (plus.largest + accuracy::lg_abs(&c)).max(minus.largest),
        )
    };
    Ok(Sum {
        value: c_div(&num, &s, p, rm),
        largest: largest + 1.0 - ls,
    })
}

/// `Y_n` or `K_n` of an integer order (DLMF 10.8.1, 10.31.1), `z ≠ 0`.
fn yk_integer(
    kind: Kind,
    n: i64,
    z: &Complex,
    wp: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<Sum, SymplexError> {
    let Some((x, y)) = parts(z) else {
        return Err(super::unevaluable("Bessel argument beyond f64"));
    };
    let m = n.unsigned_abs();
    let abs_z = x.hypot(y);
    let max = budget(abs_z, m as f64, wp) + m as usize;
    let p = wp + guard(max) + 8;
    let modified = kind == Kind::K;
    let h = ldexp(z, -1);
    let z2 = ldexp(&c_mul(z, z, p, rm), -2);
    // The series' variable and the finite sum's.
    let w = if modified { z2.clone() } else { c_neg(&z2) };
    let wf = if modified { c_neg(&z2) } else { z2.clone() };
    let aw = abs_z * abs_z / 4.0;
    // Finite sum F = Σ_{k<m} ((m−k−1)!/k!)·wf^k.
    let mut fsum = new_sum(p);
    if m > 0 {
        let mut f = (inv_factorial(0, p, rm), BigFloat::new(p));
        // (m − 1)!
        let mut g = BigFloat::from_i32(1, p);
        for k in 2..m {
            g = g.mul(&BigFloat::from_u64(k, 64), p, rm);
        }
        f = (f.0.mul(&g, p, rm), f.1);
        for k in 0..m {
            add_term(&mut fsum, &f, p, rm);
            if k + 1 < m {
                f = div_u(
                    &div_u(&c_mul(&f, &wf, p, rm), k + 1, p, rm),
                    m - k - 1,
                    p,
                    rm,
                );
            }
        }
    }
    // A = Σ u_k, B = Σ (H_k + H_{m+k}) u_k, u_k = w^k/(k!(m+k)!).
    let mut a = new_sum(p);
    let mut b = new_sum(p);
    let mut u = (inv_factorial(m, p, rm), BigFloat::new(p));
    let mut hk = BigFloat::new(p);
    let mut hmk = BigFloat::new(p);
    for j in 1..=m {
        hmk = hmk.add(
            &BigFloat::from_i32(1, p).div(&BigFloat::from_u64(j, 64), p, rm),
            p,
            rm,
        );
    }
    let mut converged = false;
    for k in 0..max as u64 {
        add_term(&mut a, &u, p, rm);
        let hs = hk.add(&hmk, p, rm);
        add_term(&mut b, &(u.0.mul(&hs, p, rm), u.1.mul(&hs, p, rm)), p, rm);
        u = div_u(
            &div_u(&c_mul(&u, &w, p, rm), k + 1, p, rm),
            m + k + 1,
            p,
            rm,
        );
        let one = BigFloat::from_i32(1, p);
        hk = hk.add(&one.div(&BigFloat::from_u64(k + 1, 64), p, rm), p, rm);
        hmk = hmk.add(&one.div(&BigFloat::from_u64(m + k + 1, 64), p, rm), p, rm);
        let q = aw / ((k as f64 + 2.0) * ((m + k) as f64 + 2.0));
        let qb = q * (1.0 + 2.0 / (k as f64 + 2.0));
        if qb < 1.0 {
            let lu = accuracy::lg_abs(&u);
            let lh = to_f64(&hk.add(&hmk, p, rm)).map_or(f64::INFINITY, |v| v.max(1e-300).log2());
            let rest_a = lu - (1.0 - q).log2();
            let rest_b = lu + lh - (1.0 - qb).log2();
            let base = a.largest.max(b.largest);
            if (rest_a < base - wp as f64 && rest_b < base - wp as f64) || is_zero(&u) {
                converged = true;
                break;
            }
        }
    }
    if !converged {
        return Err(super::series_did_not_converge("Bessel", max));
    }
    let l = super::cexpint::ln_c(&h, p, rm, cc);
    let g = super::arb_euler_gamma(p, rm, cc)?;
    let lg = (l.0.add(&g, p, rm), l.1.clone());
    let pi = cc.pi(p, rm).clone();
    let hm = crate::base::bigcomplex::c_powi(&h, m as usize, p + guard(m as usize + 2), rm);
    let hm_inv = if m == 0 {
        c_one(p)
    } else {
        c_div(&c_one(p), &hm, p, rm)
    };
    let lgl = accuracy::lg_abs(&lg).max(0.0);
    let inner_largest = (a.largest + lgl + 1.0).max(b.largest) + 1.0;
    let (value, largest) = if modified {
        // ½h^(−m)F' + (−1)^m h^m [½B' − (L + γ)A']
        let inner = c_sub(&ldexp(&b.value, -1), &c_mul(&lg, &a.value, p, rm), p, rm);
        let mut second = c_mul(&hm, &inner, p, rm);
        if m % 2 == 1 {
            second = c_neg(&second);
        }
        let first = ldexp(&c_mul(&hm_inv, &fsum.value, p, rm), -1);
        let la = accuracy::lg_abs(&hm) + inner_largest;
        let lf = accuracy::lg_abs(&hm_inv) + fsum.largest;
        (c_add(&first, &second, p, rm), la.max(lf))
    } else {
        // (h^m/π)[2(L + γ)A − B] − h^(−m)F/π
        let inner = c_sub(&ldexp(&c_mul(&lg, &a.value, p, rm), 1), &b.value, p, rm);
        let second = c_mul(&hm, &inner, p, rm);
        let first = c_mul(&hm_inv, &fsum.value, p, rm);
        let d = c_sub(&second, &first, p, rm);
        let la = accuracy::lg_abs(&hm) + inner_largest;
        let lf = accuracy::lg_abs(&hm_inv) + fsum.largest;
        let mut v = (d.0.div(&pi, p, rm), d.1.div(&pi, p, rm));
        if n < 0 && m % 2 == 1 {
            v = c_neg(&v);
        }
        (v, la.max(lf) - 1.65)
    };
    Ok(Sum { value, largest })
}

// ── Hankel's expansion ──────────────────────────────────────────────────────

/// `log₂ χ(ℓ)` bounded above (`χ(ℓ) < √(π(ℓ/2 + 1))`).
fn log2_chi(l: usize) -> f64 {
    0.5 * (PI * (l as f64 / 2.0 + 1.0)).log2()
}

/// `log₂ V_ℓ` of DLMF 10.40.12 for `abs(w)`, `ph w` (`+∞` on the boundary).
fn log2_v(l: usize, abs_w: f64, ph: f64) -> f64 {
    let a = ph.abs();
    if a <= HALF_PI {
        -(l as f64) * abs_w.log2()
    } else if a <= PI {
        log2_chi(l) - (l as f64) * abs_w.log2()
    } else if a < 1.5 * PI {
        let re = (abs_w * ph.cos()).abs();
        if re > 0.0 {
            1.0 + log2_chi(l) - (l as f64) * re.log2()
        } else {
            f64::INFINITY
        }
    } else {
        f64::INFINITY
    }
}

/// The terms of `K_ν(w)`'s expansion needed for a remainder below
/// `2^−(wp+2)` (simulated in `f64`): `(ℓ, exact)`, `exact` when the
/// expansion terminates (a half-integer order).
fn hankel_terms(nr: f64, ni: f64, abs_w: f64, ph: f64, wp: usize) -> Option<(usize, bool)> {
    if !(abs_w.is_finite() && abs_w > 1.0) {
        return None;
    }
    // 4ν² and abs(ν² − 1/4).
    let (n2r, n2i) = (nr * nr - ni * ni, 2.0 * nr * ni);
    let mu = (n2r - 0.25).hypot(n2i);
    let v1 = log2_v(1, abs_w, ph).exp2();
    let e = mu * v1 * LOG2E;
    if !e.is_finite() || e > 1e6 {
        return None;
    }
    let target = -(wp as f64) - 2.0;
    let past = 2.0 * abs_w + (nr.abs() + ni.abs()) * 2.0 + 8.0;
    let mut la = 0.0f64; // log₂ abs(a_k)
    for l in 1..MAX_ASYMPTOTIC {
        let k = l - 1;
        let f = (4.0 * n2r - ((2 * k + 1) as f64).powi(2)).hypot(4.0 * n2i);
        if f == 0.0 {
            return Some((l, true));
        }
        let next = la + f.log2() - (8.0 * l as f64).log2();
        let bound = 1.0 + next + log2_v(l, abs_w, ph) + e;
        if bound < target {
            return Some((l, false));
        }
        if l as f64 > past && next - (l as f64) * abs_w.log2() > la - (k as f64) * abs_w.log2() {
            return None;
        }
        la = next;
    }
    None
}

/// `Σ_{k<ℓ} a_k(ν) w^(−k)` with Olver's remainder below `2^−wp`, `ph` the
/// phase of `w` (on the continued sheet beyond `π`), or `None` where the
/// expansion does not get there.
fn hankel_bracket(nu: &Complex, w: &Complex, ph: f64, wp: usize, rm: RoundingMode) -> Option<Sum> {
    let (nr, ni) = parts(nu)?;
    let (x, y) = parts(w)?;
    let (l, exact) = hankel_terms(nr, ni, x.hypot(y), ph, wp)?;
    let p = wp + guard(l + 2);
    let nu2x4 = ldexp(&c_mul(nu, nu, p + 8, rm), 2);
    let mut sum = new_sum(p);
    let mut t = c_one(p);
    for k in 0..l {
        add_term(&mut sum, &t, p, rm);
        if k + 1 == l {
            break;
        }
        // a_{k+1} = a_k·(4ν² − (2k+1)²)/(8(k+1)), divided by w.
        let sq = BigFloat::from_u64(((2 * k + 1) * (2 * k + 1)) as u64, 64);
        let f = (nu2x4.0.sub(&sq, p, rm), nu2x4.1.clone());
        t = div_u(
            &c_div(&c_mul(&t, &f, p, rm), w, p, rm),
            8 * (k as u64 + 1),
            p,
            rm,
        );
    }
    let _ = exact;
    Some(sum)
}

/// `K_ν(w)` (principal `w`) by Hankel's expansion, or `None`.
fn k_asymptotic(
    nu: &Complex,
    w: &Complex,
    wp: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Option<Sum> {
    let ph = phase(w)?;
    let b = hankel_bracket(nu, w, ph, wp, rm)?;
    let p = wp + 8;
    // √(π/2)·w^(−1/2)·e^(−w).
    let mhalf = (BigFloat::from_f64(-0.5, 64), BigFloat::new(64));
    let e = super::cexpint::power_exp(w, &mhalf, true, p, rm, cc);
    let c = cc
        .pi(p, rm)
        .clone()
        .div(&BigFloat::from_i32(2, 64), p, rm)
        .sqrt(p, rm);
    let pref = scale(&e, &c, p, rm);
    Some(Sum {
        largest: accuracy::lg_abs(&pref) + b.largest,
        value: c_mul(&pref, &b.value, p, rm),
    })
}

/// `1/(πi)·v = −i·v/π`.
fn over_pi_i(v: &Complex, p: usize, rm: RoundingMode, cc: &mut Consts) -> Complex {
    let pi = cc.pi(p, rm).clone();
    let w = times_minus_i(v);
    (w.0.div(&pi, p, rm), w.1.div(&pi, p, rm))
}

/// `I_ν(z)` from `K_ν(±z)`, or `None`.
fn i_asymptotic(
    nu: &Complex,
    z: &Complex,
    wp: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Option<Sum> {
    let p = wp + 8;
    let k1 = k_asymptotic(nu, z, p, rm, cc)?;
    let k2 = k_asymptotic(nu, &c_neg(z), p, rm, cc)?;
    let up = upper(z);
    let u = if up { nu.clone() } else { c_neg(nu) };
    let e = exp_i_pi(&u, p, rm, cc);
    let ek1 = c_mul(&e, &k1.value, p, rm);
    let d = if up {
        c_sub(&k2.value, &ek1, p, rm)
    } else {
        c_sub(&ek1, &k2.value, p, rm)
    };
    Some(Sum {
        largest: (accuracy::lg_abs(&e) + k1.largest).max(k2.largest) - 1.65,
        value: over_pi_i(&d, p, rm, cc),
    })
}

/// `(H⁽¹⁾_ν(z), H⁽²⁾_ν(z))` for `Re z > 0` or `z` on the positive imaginary
/// axis, and `log₂` of the larger of their error scales, or `None`.
fn hankel_pair(
    nu: &Complex,
    z: &Complex,
    wp: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Option<(Complex, Complex, f64)> {
    let p = wp + 8;
    let ka = k_asymptotic(nu, &times_minus_i(z), p, rm, cc)?;
    let kb = k_asymptotic(nu, &times_i(z), p, rm, cc)?;
    let half = ldexp(nu, -1);
    let em = exp_i_pi(&c_neg(&half), p, rm, cc);
    let ep = exp_i_pi(&half, p, rm, cc);
    // H1 = (2/(πi))·e^(−iνπ/2)·ka, H2 = −(2/(πi))·e^(iνπ/2)·kb.
    let h1 = ldexp(&over_pi_i(&c_mul(&em, &ka.value, p, rm), p, rm, cc), 1);
    let h2 = c_neg(&ldexp(
        &over_pi_i(&c_mul(&ep, &kb.value, p, rm), p, rm, cc),
        1,
    ));
    let largest =
        (accuracy::lg_abs(&em) + ka.largest).max(accuracy::lg_abs(&ep) + kb.largest) - 0.65;
    Some((h1, h2, largest))
}

/// `J_ν(z)` or `Y_ν(z)` by Hankel's expansion, or `None`.
fn jy_asymptotic(
    kind: Kind,
    nu: &Complex,
    z: &Complex,
    wp: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Option<Sum> {
    let p = wp + 8;
    let right = z.0.is_positive() && !z.0.is_zero() || (z.0.is_zero() && z.1.is_positive());
    let base = if right { z.clone() } else { c_neg(z) };
    let (h1, h2, largest) = hankel_pair(nu, &base, p, rm, cc)?;
    let j = ldexp(&c_add(&h1, &h2, p, rm), -1);
    let yv = times_minus_i(&ldexp(&c_sub(&h1, &h2, p, rm), -1));
    if right {
        let v = if kind == Kind::J { j } else { yv };
        return Some(Sum { value: v, largest });
    }
    let up = upper(z);
    let u = if up { nu.clone() } else { c_neg(nu) };
    let e = exp_i_pi(&u, p, rm, cc);
    let le = accuracy::lg_abs(&e);
    if kind == Kind::J {
        return Some(Sum {
            value: c_mul(&e, &j, p, rm),
            largest: largest + le,
        });
    }
    // Y_ν(z) = e^(∓iνπ)Y_ν(−z) ± 2i·cos(νπ)·J_ν(−z).
    let einv = exp_i_pi(&c_neg(&u), p, rm, cc);
    let (_, c) = sin_cos_pi(nu, p, rm, cc);
    let t = ldexp(&times_i(&c_mul(&c, &j, p, rm)), 1);
    let first = c_mul(&einv, &yv, p, rm);
    let v = if up {
        c_add(&first, &t, p, rm)
    } else {
        c_sub(&first, &t, p, rm)
    };
    Some(Sum {
        value: v,
        largest: (largest + accuracy::lg_abs(&einv)).max(largest + accuracy::lg_abs(&c) + 1.0),
    })
}

// ── Bessel entry ────────────────────────────────────────────────────────────

/// A Bessel function at working precision `wp`, `z ≠ 0`.
fn bessel_at(
    kind: Kind,
    nu: &Complex,
    z: &Complex,
    wp: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<Sum, SymplexError> {
    let asym = match kind {
        Kind::K => k_asymptotic(nu, z, wp, rm, cc),
        Kind::I => i_asymptotic(nu, z, wp, rm, cc),
        Kind::J | Kind::Y => jy_asymptotic(kind, nu, z, wp, rm, cc),
    };
    if let Some(s) = asym {
        return Ok(s);
    }
    match kind {
        Kind::J => ji_series(nu, z, false, wp, rm, cc),
        Kind::I => ji_series(nu, z, true, wp, rm, cc),
        Kind::Y | Kind::K => match exact_int(nu) {
            Some(n) => yk_integer(kind, n, z, wp, rm, cc),
            None => yk_noninteger(kind, nu, z, wp, rm, cc),
        },
    }
}

/// A Bessel function at a non-real argument or order (or a negative
/// argument where the value is not real), rounded to `prec + 16` bits, its
/// error below `2^−(prec+12)` of its modulus.
pub(super) fn bessel(
    kind: Kind,
    nu: &Complex,
    z: &Complex,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<Complex, SymplexError> {
    if [&nu.0, &nu.1, &z.0, &z.1]
        .iter()
        .any(|x| x.is_nan() || x.is_inf())
    {
        return Err(super::unevaluable("Bessel of special float value"));
    }
    if is_zero(z) {
        // J_0(0) = I_0(0) = 1, J_ν(0) = I_ν(0) = 0 for Re ν > 0.
        return match kind {
            Kind::J | Kind::I if is_zero(nu) => Ok(c_one(prec + 16)),
            Kind::J | Kind::I if nu.0.is_positive() && !nu.0.is_zero() => Ok(c_zero(prec + 16)),
            Kind::J | Kind::I if exact_int(nu).is_some() => Ok(c_zero(prec + 16)),
            _ => Err(super::unevaluable("Bessel function at 0 is not finite")),
        };
    }
    tracing::debug!(
        prec,
        ?kind,
        "evalf: Bessel function at a complex argument or order"
    );
    super::zeta::with_loss(prec, rm, cc, &mut |wp, cc| {
        bessel_at(kind, nu, z, wp, rm, cc)
    })
}

// ── Airy ────────────────────────────────────────────────────────────────────

/// `Ai(w)` (or `Ai′(w)`) by Hankel's expansion of `K_{1/3}` (`K_{2/3}`) for
/// `abs(ph w) ≤ 0.7π`, or `None`.
fn ai_direct(
    w: &Complex,
    deriv: bool,
    wp: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Option<Sum> {
    let ph = phase(w)?;
    if ph.abs() > 0.7 * PI {
        return None;
    }
    let p = wp + 8;
    let (x, y) = parts(w)?;
    let lz = (x.hypot(y).powf(1.5) * 2.0 / 3.0).max(1.0);
    let pe = p + super::magnitude_bits(lz * (x.hypot(y).ln().abs() + 4.0)) + 8;
    let l = super::cexpint::ln_c(w, pe, rm, cc);
    // ζ = (2/3)·exp(1.5·Ln w).
    let three_halves = BigFloat::from_f64(1.5, 64);
    let w32 = super::c_exp(&scale(&l, &three_halves, pe, rm), pe, rm, cc);
    let two_thirds = BigFloat::from_i32(2, 64).div(&BigFloat::from_i32(3, 64), pe, rm);
    let zeta = scale(&w32, &two_thirds, pe, rm);
    let nu = if deriv {
        (two_thirds.clone(), BigFloat::new(64))
    } else {
        (
            BigFloat::from_i32(1, 64).div(&BigFloat::from_i32(3, 64), pe, rm),
            BigFloat::new(64),
        )
    };
    let b = hankel_bracket(&nu, &zeta, 1.5 * ph, wp, rm)?;
    // e^(c·Ln w − ζ)/(2√π), c = ∓1/4.
    let c = BigFloat::from_f64(if deriv { 0.25 } else { -0.25 }, 64);
    let e = c_sub(&scale(&l, &c, pe, rm), &zeta, pe, rm);
    let ex = super::c_exp(&e, pe, rm, cc);
    let two_sqrt_pi = ldexp(&(cc.pi(p, rm).clone().sqrt(p, rm), BigFloat::new(p)), 1);
    let mut pref = (
        ex.0.div(&two_sqrt_pi.0, p, rm),
        ex.1.div(&two_sqrt_pi.0, p, rm),
    );
    if deriv {
        pref = c_neg(&pref);
    }
    Some(Sum {
        largest: accuracy::lg_abs(&pref) + b.largest,
        value: c_mul(&pref, &b.value, p, rm),
    })
}

/// `e^(2πik/3)` for an integer `k`, and `e^(±πi/6)`, `e^(±5πi/6)`: `(cos,
/// sin)` of the angle `πm/6`.
fn root(m: i32, p: usize, rm: RoundingMode) -> Complex {
    let s3 = BigFloat::from_i32(3, p).sqrt(p, rm);
    let half = BigFloat::from_f64(0.5, 64);
    let h3 = ldexp(&(s3, BigFloat::new(p)), -1).0; // √3/2
    // cos and sin of πm/6 for m mod 12.
    let (c, s) = match m.rem_euclid(12) {
        0 => (BigFloat::from_i32(1, p), BigFloat::new(p)),
        1 => (h3.clone(), half.clone()),
        2 => (half.clone(), h3.clone()),
        3 => (BigFloat::new(p), BigFloat::from_i32(1, p)),
        4 => (half.neg(), h3.clone()),
        5 => (h3.neg(), half.clone()),
        6 => (BigFloat::from_i32(-1, p), BigFloat::new(p)),
        7 => (h3.neg(), half.neg()),
        8 => (half.neg(), h3.neg()),
        9 => (BigFloat::new(p), BigFloat::from_i32(-1, p)),
        10 => (half.clone(), h3.neg()),
        _ => (h3.clone(), half.neg()),
    };
    (c, s)
}

/// `Ai(w)` or `Ai′(w)` by Hankel's expansion anywhere (the connection
/// formula beyond `abs(ph w) = 0.65π`), or `None`.
fn ai_any(w: &Complex, deriv: bool, wp: usize, rm: RoundingMode, cc: &mut Consts) -> Option<Sum> {
    let ph = phase(w)?;
    if ph.abs() <= 0.65 * PI {
        return ai_direct(w, deriv, wp, rm, cc);
    }
    let (x, y) = parts(w)?;
    let p = wp + 8 + super::magnitude_bits(x.hypot(y).powf(1.5) * 2.0) + 4;
    let w1 = c_mul(w, &root(4, p, rm), p, rm); // w·e^(2πi/3)
    let w2 = c_mul(w, &root(-4, p, rm), p, rm); // w·e^(−2πi/3)
    let a1 = ai_direct(&w1, deriv, p, rm, cc)?;
    let a2 = ai_direct(&w2, deriv, p, rm, cc)?;
    // Ai: −e^(2πi/3)Ai(w1) − e^(−2πi/3)Ai(w2); Ai′: −e^(4πi/3)Ai′(w1) − e^(−4πi/3)Ai′(w2).
    let (r1, r2) = if deriv {
        (root(8, p, rm), root(-8, p, rm))
    } else {
        (root(4, p, rm), root(-4, p, rm))
    };
    let v = c_neg(&c_add(
        &c_mul(&r1, &a1.value, p, rm),
        &c_mul(&r2, &a2.value, p, rm),
        p,
        rm,
    ));
    Some(Sum {
        value: v,
        largest: a1.largest.max(a2.largest) + 1.0,
    })
}

/// An Airy function by Hankel's expansion, or `None`.
fn airy_asymptotic(
    kind: AiryKind,
    z: &Complex,
    wp: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Option<Sum> {
    match kind {
        AiryKind::Ai => ai_any(z, false, wp, rm, cc),
        AiryKind::AiPrime => ai_any(z, true, wp, rm, cc),
        AiryKind::Bi | AiryKind::BiPrime => {
            let deriv = kind == AiryKind::BiPrime;
            let (x, y) = parts(z)?;
            let p = wp + 8 + super::magnitude_bits(x.hypot(y).powf(1.5) * 2.0) + 4;
            let w1 = c_mul(z, &root(4, p, rm), p, rm);
            let w2 = c_mul(z, &root(-4, p, rm), p, rm);
            let a1 = ai_any(&w1, deriv, p, rm, cc)?;
            let a2 = ai_any(&w2, deriv, p, rm, cc)?;
            // Bi: e^(πi/6), e^(−πi/6); Bi′: e^(5πi/6), e^(−5πi/6).
            let (r1, r2) = if deriv {
                (root(5, p, rm), root(-5, p, rm))
            } else {
                (root(1, p, rm), root(-1, p, rm))
            };
            let v = c_add(
                &c_mul(&r1, &a1.value, p, rm),
                &c_mul(&r2, &a2.value, p, rm),
                p,
                rm,
            );
            Some(Sum {
                value: v,
                largest: a1.largest.max(a2.largest) + 1.0,
            })
        }
    }
}

/// An Airy function by its power series (module documentation).
fn airy_series(
    kind: AiryKind,
    z: &Complex,
    wp: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<Sum, SymplexError> {
    let Some((x, y)) = parts(z) else {
        return Err(super::unevaluable("Airy argument beyond f64"));
    };
    let abs_z = x.hypot(y);
    let a3 = abs_z.powi(3);
    let max = budget(abs_z.powf(1.5), 0.0, wp);
    let p = wp + guard(max) + 8;
    let deriv = matches!(kind, AiryKind::AiPrime | AiryKind::BiPrime);
    let z3 = c_mul(&c_mul(z, z, p, rm), z, p, rm);
    // The two series: (first term, the factors (3k + a)(3k + b) of the
    // ratio t_{k+1}/t_k = z³/((3k+a)(3k+b))).  For the derivatives the
    // first series starts at k = 1.
    let series = |t0: Complex, a: u64, b: u64, k0: u64| -> Result<Sum, SymplexError> {
        let mut s = new_sum(p);
        let mut t = t0;
        for k in k0..k0 + max as u64 {
            add_term(&mut s, &t, p, rm);
            t = div_u(&c_mul(&t, &z3, p, rm), (3 * k + a) * (3 * k + b), p, rm);
            let q = a3 / (((3 * k + 3 + a) * (3 * k + 3 + b)) as f64);
            if q < 1.0 {
                let rest = accuracy::lg_abs(&t) - (1.0 - q).log2();
                if rest < s.largest - wp as f64 || is_zero(&t) {
                    return Ok(s);
                }
            }
        }
        Err(super::series_did_not_converge("Airy", max))
    };
    let (f, g) = if deriv {
        // f′ = Σ_{k≥1} p_k, p_1 = z²/2, p_{k+1} = p_k z³/((3k)(3k+2));
        // g′ = Σ_{k≥0} q_k, q_0 = 1, q_{k+1} = q_k z³/((3k+1)(3k+3)).
        let p1 = ldexp(&c_mul(z, z, p, rm), -1);
        (series(p1, 0, 2, 1)?, series(c_one(p), 1, 3, 0)?)
    } else {
        // f: f_0 = 1, ratio z³/((3k+2)(3k+3)); g: g_0 = z, ratio z³/((3k+3)(3k+4)).
        (series(c_one(p), 2, 3, 0)?, series(z.clone(), 3, 4, 0)?)
    };
    // c₁ = 3^(−2/3)/Γ(2/3), c₂ = 3^(−1/3)/Γ(1/3).
    let three = BigFloat::from_i32(3, 64);
    let third = BigFloat::from_i32(1, 64).div(&three, p + 8, rm);
    let two_thirds = BigFloat::from_i32(2, 64).div(&three, p + 8, rm);
    let ln3 = three.ln(p + 8, rm, cc);
    let c1 = two_thirds
        .mul(&ln3, p + 8, rm)
        .neg()
        .exp(p + 8, rm, cc)
        .div(
            &super::arb_gamma_real(&two_thirds, p + 8, rm, cc)?,
            p + 8,
            rm,
        );
    let c2 = third.mul(&ln3, p + 8, rm).neg().exp(p + 8, rm, cc).div(
        &super::arb_gamma_real(&third, p + 8, rm, cc)?,
        p + 8,
        rm,
    );
    let a = scale(&f.value, &c1, p, rm);
    let b = scale(&g.value, &c2, p, rm);
    let largest = f.largest.max(g.largest);
    let v = match kind {
        AiryKind::Ai | AiryKind::AiPrime => c_sub(&a, &b, p, rm),
        _ => {
            let s3 = three.sqrt(p, rm);
            scale(&c_add(&a, &b, p, rm), &s3, p, rm)
        }
    };
    Ok(Sum {
        value: v,
        largest: largest + 1.0,
    })
}

/// An Airy function at a non-real argument, rounded to `prec + 16` bits,
/// its error below `2^−(prec+12)` of its modulus.
pub(super) fn airy(
    kind: AiryKind,
    z: &Complex,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<Complex, SymplexError> {
    if [&z.0, &z.1].iter().any(|x| x.is_nan() || x.is_inf()) {
        return Err(super::unevaluable("Airy function of special float value"));
    }
    tracing::debug!(prec, ?kind, "evalf: Airy function at a complex argument");
    super::zeta::with_loss(prec, rm, cc, &mut |wp, cc| {
        if let Some(s) = airy_asymptotic(kind, z, wp, rm, cc) {
            return Ok(s);
        }
        airy_series(kind, z, wp, rm, cc)
    })
}

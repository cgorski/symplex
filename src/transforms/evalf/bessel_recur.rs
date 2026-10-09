//! `J_ν(x)` and `Y_ν(x)` of large order near the turning point `x ≈ ν`, and
//! `Y_ν(x)` for `x < ν`, by the three-term recurrence in the order from
//! values the uniform expansions give (`bessel_debye`), with a rigorous
//! bound on the error it accumulates.
//!
//! The expansions of `bessel_debye` (DLMF 10.41) stop short of the turning
//! point: at 16 digits they need `abs(x − ν) ≳ 25·ν^(1/3)` (`J₂₀₀₀₀`, `Y₂₀₀₀₀`
//! from `x = 20600`), and they give no `Y_ν(x)` for `x < ν` at all.  Both
//! functions satisfy (DLMF 10.6.1)
//!
//! ```text
//! f_{k+1} = (2k/x)·f_k − f_{k−1},      k = μ + 1, μ + 2, …
//! ```
//!
//! so the values at `μ` and `μ + 1` with `x − μ − 1` beyond that distance
//! (the Hankel form of the expansions, `x > μ`) give them at `ν = μ + m` by
//! `m` steps.  The computed sequence obeys the recurrence up to a local
//! error `ρ_k` per step (the rounding of `2k/x`, of the product and of the
//! difference: `abs(ρ_k) ≤ 2^(−wp)·(2.01·abs(2k/x·f_k) + 1.01·abs(f_{k+1}))`),
//! and the error is a solution of the recurrence driven by them.  A
//! solution `g` with `(g_k, g_{k+1}) = (a, b)` is `α·J + β·Y` with `α`, `β`
//! from the cross product `J_{k+1}Y_k − J_kY_{k+1} = 2/(πx)` (DLMF 10.5.4):
//!
//! ```text
//! g_n = (πx/2)·[b·(Y_k J_n − J_k Y_n) + a·(J_{k+1} Y_n − Y_{k+1} J_n)],
//! abs(g_n) ≤ πx·L·M_n·(abs(a) + abs(b)),
//! ```
//!
//! for `μ ≤ k < k + 1 ≤ n`, with `L = b_L·μ^(−1/3) ≥ abs(J_j(x))` for every order
//! `j ≥ μ` (Landau's bound, DLMF 10.14.2, `b_L = 0.674885…`) and `M_n =
//! (J_n² + Y_n²)^½ ≥ abs(Y_j)` for `j ≤ n` (`J_j² + Y_j²` increases with the order
//! `j ≥ 0`: Nicholson's integral, DLMF 10.9.30, has `cosh(2jt)` under it).
//! So the error at `n` is at most `K·M_n`, `K = πxL·(e_μ + e_{μ+1} + Σρ_k)`,
//! for `J` and `Y` alike, and `M_n ≤ L + abs(Y_n) ≤ L + abs(Ỹ_n) + K_Y·M_n`
//! bounds `M_n` by the computed `Ỹ_n`: `M_n ≤ (L + abs(Ỹ_n))/(1 − K_Y)`.  The
//! bound is relative where it matters: `Y` beyond the turning point, which
//! grows as the recurrence goes up (it is the dominant solution there), loses
//! `log₂(πxL·abs(Y_n))` bits to the roundings of the steps and only about
//! `log₂(πxL²)` to its starting values; `J` there is recessive and loses
//! `log₂(M_n/abs(J_n))` bits to its starting values, which the expansions
//! give with that many more.
//!
//! The working precisions are raised until the bound covers the digits
//! asked for; `None` where the starting values or the cost (at most
//! [`MAX_STEPS`] steps, [`MAX_WORK`] steps times bits) do not allow it, and
//! the callers refuse as before: `Y_ν(x)` far below the turning point
//! (`bessely(200000, 150000)`), where it is huge and the steps many.

use astro_float::{BigFloat, Consts, RoundingMode};

use super::bessel_debye::{self, Kind};
use crate::base::errors::SymplexError;

/// The most steps of the recurrence.
const MAX_STEPS: usize = 30_000;

/// The most steps times bits of the passes of one evaluation.
const MAX_WORK: f64 = 1.5e8;

/// Landau's constant `b_L` of DLMF 10.14.2 (`0.6748851…`), rounded up.
const LANDAU: f64 = 0.674_886;

/// `log₂(2^a + 2^b)`, rounded up (`−∞` for an empty sum).
fn lsum(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    hi + (lo - hi).exp2().ln_1p() / std::f64::consts::LN_2 + 1e-12
}

/// An upper bound of `log₂ abs(x)` (`−∞` for 0).
fn lg_up(x: &BigFloat) -> f64 {
    match x.exponent() {
        Some(e) if !x.is_zero() => f64::from(e),
        _ => f64::NEG_INFINITY,
    }
}

/// A lower bound of `log₂ abs(x)` (`−∞` for 0).
fn lg_low(x: &BigFloat) -> f64 {
    match x.exponent() {
        Some(e) if !x.is_zero() => f64::from(e) - 1.0,
        _ => f64::NEG_INFINITY,
    }
}

/// The values at two consecutive orders and their error bounds.
struct Pair {
    lo: BigFloat,
    hi: BigFloat,
    /// `log₂` of the bounds on the errors of `lo` and `hi`.
    err: f64,
}

/// `kind` at `μ` and `μ + 1` by the uniform expansions at `wp` bits.
fn start(
    kind: Kind,
    mu: &BigFloat,
    mu1: &BigFloat,
    x: &BigFloat,
    wp: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Option<Pair> {
    let lo = bessel_debye::debye(kind, mu, x, wp, rm, cc)?.ok()?;
    let hi = bessel_debye::debye(kind, mu1, x, wp, rm, cc)?.ok()?;
    // Each is correct to `2^(−wp)` relative (and the rounding to `wp` bits).
    let err = lsum(lg_up(&lo), lg_up(&hi)) - wp as f64 + 1.0;
    Some(Pair { lo, hi, err })
}

/// Run the recurrence from `(f_μ, f_{μ+1})` up to `f_n`, `n = μ + 1 + steps`:
/// the value at `n` and `log₂ Σ abs(ρ_k)`.
fn run(
    p: &Pair,
    mu1: &BigFloat,
    x: &BigFloat,
    steps: usize,
    wp: usize,
    rm: RoundingMode,
) -> (BigFloat, f64) {
    let mut prev = p.lo.clone();
    let mut cur = p.hi.clone();
    let mut order = mu1.clone();
    let mut rho = f64::NEG_INFINITY;
    let one = BigFloat::from_i32(1, 64);
    // `2k/x = k·(2/x)`, `2/x` to `2^(−wp−16)` relative: `c` is within
    // `(1 + 2⁻¹⁶)·2^(−wp)` of `2k/x` (no division per step).
    let inv = BigFloat::from_i32(2, 64).div(x, wp + 16, rm);
    // The orders `μ + 1 + j` exactly, at the bits of `μ + 1` (and room for
    // the steps).
    let po = super::exact_bits(mu1, 64);
    for _ in 0..steps {
        let c = order.mul(&inv, wp, rm);
        let next = c.mul(&cur, wp, rm).sub(&prev, wp, rm);
        let local = lsum(lg_up(&c) + lg_up(&cur) + 1.01, lg_up(&next) + 0.015) - wp as f64;
        rho = lsum(rho, local);
        prev = cur;
        cur = next;
        order = order.add(&one, po, rm);
    }
    (cur, rho)
}

/// `J_ν(x)` (`kind` J) or `Y_ν(x)` for `ν ≥ 16`, `x > 0`, from orders below
/// `x` (see the module documentation), rounded to `prec` bits; `None` where
/// the starting values or the steps do not reach the precision.
pub(super) fn recur(
    kind: Kind,
    nu: &BigFloat,
    x: &BigFloat,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Option<Result<BigFloat, SymplexError>> {
    if !matches!(kind, Kind::J | Kind::Y)
        || !super::bf_strictly_positive(nu)
        || !super::bf_strictly_positive(x)
    {
        return None;
    }
    let nu_f = super::bigfloat_to_f64_rounded(nu, rm).ok()?;
    let x_f = super::bigfloat_to_f64_rounded(x, rm).ok()?;
    if !nu_f.is_finite() || !x_f.is_finite() || nu_f < 32.0 {
        return None;
    }
    // The first distance below `x` tried for the starting orders: a little
    // more than the expansions need at 16 digits.
    let mut dist = 24.0 * nu_f.cbrt();
    let mut wp_start = prec + 24;
    let mut wp_run = prec + 32;
    let mut work = 0.0f64;
    for _ in 0..12 {
        // `μ + 1 = ν − m` at least `dist` below `x`, and above 16.
        let below = (nu_f - (x_f - dist)).max(1.0).ceil();
        let m = below as usize;
        if m > MAX_STEPS || nu_f - below - 1.0 < 16.0 {
            return None;
        }
        let one = BigFloat::from_i32(1, 64);
        let mb = BigFloat::from_u64(m as u64, 64);
        let mu1 = nu.sub(&mb, super::exact_bits(nu, prec), rm);
        let mu = mu1.sub(&one, super::exact_bits(&mu1, prec), rm);
        let mu_f = nu_f - below - 1.0;
        let Some(y) = start(Kind::Y, &mu, &mu1, x, wp_start, rm, cc) else {
            dist *= 2.0;
            continue;
        };
        let j = if kind == Kind::J {
            match start(Kind::J, &mu, &mu1, x, wp_start, rm, cc) {
                Some(j) => Some(j),
                None => {
                    dist *= 2.0;
                    continue;
                }
            }
        } else {
            None
        };
        // The steps of all the passes together: at most about half a second
        // (`bessely(20000, 14800)` takes `2.5·10⁷` bit-steps).
        let runs = if kind == Kind::J { 2.0 } else { 1.0 };
        work += runs * (m as f64) * (wp_run as f64);
        if wp_run > 8 * prec + 65_536 || wp_start > 4 * prec + 4096 || work > MAX_WORK {
            return None;
        }
        let (yn, rho_y) = run(&y, &mu1, x, m, wp_run, rm);
        // `πx·L`, `L = b_L·μ^(−1/3)`.
        let lg_l = (LANDAU * mu_f.powf(-1.0 / 3.0)).log2();
        let lg_pxl = (std::f64::consts::PI * x_f).log2() + lg_l + 1e-9;
        // `K_Y` from the starting values and from the steps.
        let (start_y, steps_y) = (lg_pxl + y.err + 1.0, lg_pxl + rho_y);
        let k_y = lsum(start_y, steps_y);
        if k_y > -2.0 {
            // Each part to below 2^(−prec−8), where the value's bound will
            // want it (relative to `Y_n`, which is about `M_n` here), with 32
            // bits to spare.
            let more = |part: f64| (part + (prec + 8) as f64).max(0.0).ceil() as usize + 32;
            if steps_y > -3.0 {
                wp_run += more(steps_y);
            }
            if start_y > -3.0 {
                wp_start += more(start_y);
            }
            continue;
        }
        // `M_n ≤ (L + abs(Ỹ_n))/(1 − K_Y)`, `1/(1 − K_Y) ≤ 2^(2·K_Y)` for `K_Y ≤ ¼`.
        let lg_m = lsum(lg_l, lg_up(&yn)) + 2.0 * k_y.exp2();
        let (value, start_part, steps_part) = match &j {
            None => (yn, start_y, steps_y),
            Some(j) => {
                let (jn, rho_j) = run(j, &mu1, x, m, wp_run, rm);
                (jn, lg_pxl + j.err + 1.0, lg_pxl + rho_j)
            }
        };
        // The error `K·M_n` relative to the value, against `2^(−prec−3)`,
        // for each part of `K`.
        let target = lg_low(&value) - (prec + 4) as f64 - lg_m;
        if value.is_zero() || lsum(start_part, steps_part) > target + 1.0 {
            let more = |part: f64| {
                if value.is_zero() {
                    prec + 32
                } else {
                    (part - target).max(0.0).ceil() as usize + 32
                }
            };
            if value.is_zero() || steps_part > target {
                wp_run += more(steps_part);
            }
            if value.is_zero() || start_part > target {
                wp_start += more(start_part);
            }
            continue;
        }
        return Some(Ok(super::round_to(value, prec, rm)));
    }
    None
}

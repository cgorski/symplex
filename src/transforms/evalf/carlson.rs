//! Elliptic integrals (Carlson symmetric forms) (part of [`evalf`](super)).

use super::*;

// ── Elliptic integrals (Carlson symmetric forms) ────────────────────────────────

/// The deviations `1 − v/μ` of the Carlson arguments `vals` from their
/// weighted mean `μ` (`None` for `μ = 0`).  The duplication iterations stop
/// once Carlson's series truncated after degree 5 is within `2^−wp` there
/// (`elliptic::series_ready`: deviations below `2^−(wp/6 + 2)`); up to 0.55
/// they ran until the arguments agreed to `2^−(wp/2 + 8)` and took `μ^{−1/2}`
/// (`μ^{−3/2}`), three times as many steps.
fn carlson_devs(vals: &[&BigFloat], mu: &BigFloat, wp: usize) -> Option<Vec<Complex>> {
    if mu.is_zero() {
        return None;
    }
    let one = BigFloat::from_i32(1, wp);
    Some(
        vals.iter()
            .map(|v| (one.sub(&v.div(mu, wp, RM), wp, RM), BigFloat::new(wp)))
            .collect(),
    )
}

/// Carlson `R_C(x, y) = R_F(x, y, y)`, `x ≥ 0`, `y > 0`.
///
/// With `w = (y − x)/y`, `R_C = y^{−1/2} arcsin(√w)/√w = y^{−1/2} Σ_k C(2k,k) w^k / (4^k (2k+1))`
/// (the series is analytic through `w = 0`, covering `x > y` as well).
/// The series is used for `|w| < 1/10`: the closed forms
/// `arccos(√(x/y))/√(y−x)` / `arccosh(√(x/y))/√(x−y)` lose relative
/// precision like `1/w` there, which the `R_J` duplication (where
/// `y − x` shrinks quadratically) would amplify into a first-order error.
fn carlson_rc(x: &BigFloat, y: &BigFloat, wpc: &mut Wp<'_>) -> BigFloat {
    let wp = wpc.bits();
    let one = BigFloat::from_i32(1, wp);
    let diff = y.sub(x, wp, RM);
    let inv_sqrt_y = one.div(&y.sqrt(wp, RM), wp, RM);
    if diff.is_zero() {
        return inv_sqrt_y;
    }
    let w = diff.div(y, wp, RM);
    let w_f = f64_of(&w).unwrap_or(1.0);
    if w_f.abs() < 0.1 {
        // c_k = c_{k−1} (2k−1)² / ((2k)(2k+1)):  1, 1/6, 3/40, 5/112, …
        let mut sum = one.clone();
        let mut c = one.clone();
        let mut wk = one.clone();
        for k in 1..(wp * 2 + 100) {
            let k_i = k as i128;
            let num = BigFloat::from_i128((2 * k_i - 1) * (2 * k_i - 1), wp);
            let den = BigFloat::from_i128(2 * k_i * (2 * k_i + 1), wp);
            c = c.mul(&num, wp, RM).div(&den, wp, RM);
            wk = wk.mul(&w, wp, RM);
            let term = c.mul(&wk, wp, RM);
            sum = sum.add(&term, wp, RM);
            if negligible(&term, &sum, wp) {
                break;
            }
        }
        return inv_sqrt_y.mul(&sum, wp, RM);
    }
    let ratio = x.div(y, wp, RM).sqrt(wp, RM); // √(x/y)
    if is_sign_positive(&diff) {
        // y > x: arccos(√(x/y)) / √(y − x)
        ratio.acos(wp, RM, wpc.cc).div(&diff.sqrt(wp, RM), wp, RM)
    } else {
        // x > y: arccosh(√(x/y)) / √(x − y)
        ratio
            .acosh(wp, RM, wpc.cc)
            .div(&diff.neg().sqrt(wp, RM), wp, RM)
    }
}

/// Carlson `R_F(x, y, z)` (`x, y, z ≥ 0`, at most one zero) by the
/// duplication theorem iterated until the arguments are close enough to
/// their mean `μ` for Carlson's series, `R_F ≈ μ^{−1/2}·(1 − E₂/10 + …)`
/// ([`carlson_devs`]).
pub(super) fn carlson_rf(
    x: &BigFloat,
    y: &BigFloat,
    z: &BigFloat,
    wp: usize,
) -> Result<BigFloat, SymplexError> {
    let four = BigFloat::from_i32(4, wp);
    let (mut x, mut y, mut z) = (x.clone(), y.clone(), z.clone());
    for _ in 0..(wp * 2 + 200) {
        // Carlson's series about μ = (x + y + z)/3 (`elliptic::rf_series`).
        let mu = x
            .add(&y, wp, RM)
            .add(&z, wp, RM)
            .div(&BigFloat::from_i32(3, wp), wp, RM);
        if let Some(d) = carlson_devs(&[&x, &y, &z], &mu, wp) {
            let devs: Vec<&Complex> = d.iter().collect();
            if elliptic::series_ready(&devs, wp) {
                let s = elliptic::rf_series(&devs, wp).0;
                return Ok(s.div(&mu.sqrt(wp, RM), wp, RM));
            }
        }
        let (sx, sy, sz) = (x.sqrt(wp, RM), y.sqrt(wp, RM), z.sqrt(wp, RM));
        let lambda =
            sx.mul(&sy, wp, RM)
                .add(&sy.mul(&sz, wp, RM), wp, RM)
                .add(&sz.mul(&sx, wp, RM), wp, RM);
        x = x.add(&lambda, wp, RM).div(&four, wp, RM);
        y = y.add(&lambda, wp, RM).div(&four, wp, RM);
        z = z.add(&lambda, wp, RM).div(&four, wp, RM);
    }
    Err(SymplexError::ComputationFailed {
        operation: "evalf",
        reason: "Carlson R_F did not converge".into(),
    })
}

/// Carlson `R_D(x, y, z) = R_J(x, y, z, z)` (`z > 0`) by duplication:
/// `R_D(x,y,z) = R_D(x',y',z')/4 + 3/(√z (z + λ))`, the remainder by Carlson's
/// series about `μ = (x + y + 3z)/5` ([`carlson_devs`]).
pub(super) fn carlson_rd(
    x: &BigFloat,
    y: &BigFloat,
    z: &BigFloat,
    wp: usize,
) -> Result<BigFloat, SymplexError> {
    let one = BigFloat::from_i32(1, wp);
    let three = BigFloat::from_i32(3, wp);
    let four = BigFloat::from_i32(4, wp);
    let (mut x, mut y, mut z) = (x.clone(), y.clone(), z.clone());
    let mut acc = BigFloat::new(wp);
    let mut scale = one.clone(); // 4^{−k}
    for _ in 0..(wp * 2 + 200) {
        // Carlson's series about μ = (x + y + 3z)/5 (`elliptic::rj_series`).
        let mu = x.add(&y, wp, RM).add(&three.mul(&z, wp, RM), wp, RM).div(
            &BigFloat::from_i32(5, wp),
            wp,
            RM,
        );
        if let Some(d) = carlson_devs(&[&x, &y, &z], &mu, wp) {
            let devs = [&d[0], &d[1], &d[2], &d[2], &d[2]];
            if elliptic::series_ready(&devs, wp) {
                let s = elliptic::rj_series(&devs, wp).0;
                let tail = s.div(&mu.mul(&mu.sqrt(wp, RM), wp, RM), wp, RM);
                return Ok(acc.add(&scale.mul(&tail, wp, RM), wp, RM));
            }
        }
        let (sx, sy, sz) = (x.sqrt(wp, RM), y.sqrt(wp, RM), z.sqrt(wp, RM));
        let lambda =
            sx.mul(&sy, wp, RM)
                .add(&sy.mul(&sz, wp, RM), wp, RM)
                .add(&sz.mul(&sx, wp, RM), wp, RM);
        let term = three.div(&sz.mul(&z.add(&lambda, wp, RM), wp, RM), wp, RM);
        acc = acc.add(&scale.mul(&term, wp, RM), wp, RM);
        scale = scale.div(&four, wp, RM);
        x = x.add(&lambda, wp, RM).div(&four, wp, RM);
        y = y.add(&lambda, wp, RM).div(&four, wp, RM);
        z = z.add(&lambda, wp, RM).div(&four, wp, RM);
    }
    Err(SymplexError::ComputationFailed {
        operation: "evalf",
        reason: "Carlson R_D did not converge".into(),
    })
}

/// Carlson `R_J(x, y, z, p)` for `p > 0` by duplication:
/// `R_J = R_J(x',y',z',p')/4 + 3 R_C(α², β²)` with
/// `α = p(√x + √y + √z) + √x√y√z`, `β = √p (p + λ)`, the remainder by
/// Carlson's series about `μ = (x + y + z + 2p)/5` ([`carlson_devs`]).
fn carlson_rj(
    x: &BigFloat,
    y: &BigFloat,
    z: &BigFloat,
    p: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let wp = w.bits();
    if !is_sign_positive(p) {
        return Err(unevaluable(
            "elliptic_pi: the characteristic n must satisfy n < 1 for a real value",
        ));
    }
    let one = BigFloat::from_i32(1, wp);
    let three = BigFloat::from_i32(3, wp);
    let four = BigFloat::from_i32(4, wp);
    let (mut x, mut y, mut z, mut p) = (x.clone(), y.clone(), z.clone(), p.clone());
    let mut acc = BigFloat::new(wp);
    let mut scale = one.clone();
    for _ in 0..(wp * 2 + 200) {
        // Carlson's series about μ = (x + y + z + 2p)/5 (`elliptic::rj_series`).
        let two_p = p.mul(&BigFloat::from_i32(2, wp), wp, RM);
        let mu = x.add(&y, wp, RM).add(&z, wp, RM).add(&two_p, wp, RM).div(
            &BigFloat::from_i32(5, wp),
            wp,
            RM,
        );
        if let Some(d) = carlson_devs(&[&x, &y, &z, &p], &mu, wp) {
            let devs = [&d[0], &d[1], &d[2], &d[3], &d[3]];
            if elliptic::series_ready(&devs, wp) {
                let s = elliptic::rj_series(&devs, wp).0;
                let tail = s.div(&mu.mul(&mu.sqrt(wp, RM), wp, RM), wp, RM);
                return Ok(acc.add(&scale.mul(&tail, wp, RM), wp, RM));
            }
        }
        let (sx, sy, sz, sp) = (
            x.sqrt(wp, RM),
            y.sqrt(wp, RM),
            z.sqrt(wp, RM),
            p.sqrt(wp, RM),
        );
        let lambda =
            sx.mul(&sy, wp, RM)
                .add(&sy.mul(&sz, wp, RM), wp, RM)
                .add(&sz.mul(&sx, wp, RM), wp, RM);
        let alpha = p.mul(&sx.add(&sy, wp, RM).add(&sz, wp, RM), wp, RM).add(
            &sx.mul(&sy, wp, RM).mul(&sz, wp, RM),
            wp,
            RM,
        );
        let beta = sp.mul(&p.add(&lambda, wp, RM), wp, RM);
        let rc = carlson_rc(&alpha.mul(&alpha, wp, RM), &beta.mul(&beta, wp, RM), w);
        acc = acc.add(&scale.mul(&three.mul(&rc, wp, RM), wp, RM), wp, RM);
        scale = scale.div(&four, wp, RM);
        x = x.add(&lambda, wp, RM).div(&four, wp, RM);
        y = y.add(&lambda, wp, RM).div(&four, wp, RM);
        z = z.add(&lambda, wp, RM).div(&four, wp, RM);
        p = p.add(&lambda, wp, RM).div(&four, wp, RM);
    }
    Err(SymplexError::ComputationFailed {
        operation: "evalf",
        reason: "Carlson R_J did not converge".into(),
    })
}

/// `1 − m` with a domain check (`m < 1` required for a real `K`, `E`, `Π`).
fn one_minus_m(m: &BigFloat, wp: usize, what: &str) -> Result<BigFloat, SymplexError> {
    let one = BigFloat::from_i32(1, wp);
    let v = one.sub(m, wp, RM);
    if v.is_negative() {
        return Err(unevaluable(format!(
            "{what}: parameter m must satisfy m ≤ 1 for a real value"
        )));
    }
    Ok(v)
}

/// Complete elliptic integral of the first kind `K(m) = R_F(0, 1−m, 1)`.
pub(super) fn arb_elliptic_k(m: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    if m.is_nan() || m.is_inf() {
        return Err(unevaluable("elliptic_k of special float value"));
    }
    let wp = prec + 32;
    let y = one_minus_m(m, wp, "elliptic_k")?;
    if y.is_zero() {
        return Err(infinite("elliptic_k(1) is infinite"));
    }
    let _ = w.cc;
    let r = carlson_rf(&BigFloat::new(wp), &y, &BigFloat::from_i32(1, wp), wp)?;
    Ok(round_to(r, prec))
}

/// Complete elliptic integral of the second kind
/// `E(m) = R_F(0, 1−m, 1) − (m/3) R_D(0, 1−m, 1)`.
pub(super) fn arb_elliptic_e(m: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    if m.is_nan() || m.is_inf() {
        return Err(unevaluable("elliptic_e of special float value"));
    }
    let wp = prec + 32;
    let one = BigFloat::from_i32(1, wp);
    let y = one_minus_m(m, wp, "elliptic_e")?;
    if y.is_zero() {
        return Ok(BigFloat::from_i32(1, prec));
    }
    let _ = w.cc;
    let zero = BigFloat::new(wp);
    let rf = carlson_rf(&zero, &y, &one, wp)?;
    let rd = carlson_rd(&zero, &y, &one, wp)?;
    let three = BigFloat::from_i32(3, wp);
    let r = rf.sub(&m.mul(&rd, wp, RM).div(&three, wp, RM), wp, RM);
    Ok(round_to(r, prec))
}

/// Incomplete elliptic integral of the first kind
/// `F(φ | m) = sinφ · R_F(cos²φ, 1 − m sin²φ, 1)` for `|φ| ≤ π/2`, extended
/// by `F(φ + kπ | m) = F(φ | m) + 2k K(m)`.
pub(super) fn arb_elliptic_f(
    phi: &BigFloat,
    m: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    if phi.is_nan() || phi.is_inf() || m.is_nan() || m.is_inf() {
        return Err(unevaluable("elliptic_f of special float value"));
    }
    if phi.is_zero() {
        return Ok(BigFloat::new(prec));
    }
    let wp = prec + 32;
    let one = BigFloat::from_i32(1, wp);
    let pi = w.cc.pi(wp, RM).clone();
    // Reduce φ to [−π/2, π/2]: φ = φ₀ + kπ.
    let k_bf = phi
        .div(&pi, wp, RM)
        .add(&BigFloat::from_f64(0.5, wp), wp, RM)
        .floor();
    let phi0 = phi.sub(&k_bf.mul(&pi, wp, RM), wp, RM);
    let s = phi0.sin(wp, RM, w.cc);
    let c = phi0.cos(wp, RM, w.cc);
    let s2 = s.mul(&s, wp, RM);
    let c2 = c.mul(&c, wp, RM);
    let y = one.sub(&m.mul(&s2, wp, RM), wp, RM);
    if !is_sign_positive(&y) {
        return Err(unevaluable(
            "elliptic_f: 1 − m sin²φ must be positive for a real value",
        ));
    }
    let mut r = s.mul(&carlson_rf(&c2, &y, &one, wp)?, wp, RM);
    if !k_bf.is_zero() {
        let k = arb_elliptic_k(m, &mut w.at(wp))?;
        let two = BigFloat::from_i32(2, wp);
        r = r.add(&two.mul(&k_bf, wp, RM).mul(&k, wp, RM), wp, RM);
    }
    Ok(round_to(r, prec))
}

/// Complete elliptic integral of the third kind
/// `Π(n | m) = R_F(0, 1−m, 1) + (n/3) R_J(0, 1−m, 1, 1−n)` for `n < 1`, `m < 1`.
pub(super) fn arb_elliptic_pi(
    n: &BigFloat,
    m: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    if n.is_nan() || n.is_inf() || m.is_nan() || m.is_inf() {
        return Err(unevaluable("elliptic_pi of special float value"));
    }
    let wp = prec + 32;
    let one = BigFloat::from_i32(1, wp);
    let zero = BigFloat::new(wp);
    let y = one_minus_m(m, wp, "elliptic_pi")?;
    if y.is_zero() {
        return Err(infinite("elliptic_pi(n, 1) is infinite"));
    }
    let p = one.sub(n, wp, RM);
    let rf = carlson_rf(&zero, &y, &one, wp)?;
    if n.is_zero() {
        return Ok(round_to(rf, prec));
    }
    let rj = carlson_rj(&zero, &y, &one, &p, &mut w.at(wp))?;
    let three = BigFloat::from_i32(3, wp);
    let r = rf.add(&n.mul(&rj, wp, RM).div(&three, wp, RM), wp, RM);
    Ok(round_to(r, prec))
}

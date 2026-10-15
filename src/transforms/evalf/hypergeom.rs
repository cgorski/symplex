//! The classical orthogonal polynomials of a non-integer degree, by their
//! hypergeometric definitions, at real arguments where the value is real.
//!
//! The recurrences of `evalf.rs` (`arb_orthopoly`, `arb_gegenbauer`, …)
//! take a non-negative integer degree.  For a real non-integer degree `ν`
//! the functions are (DLMF 18.5.7, 18.5.9, 18.5.12, 14.3.1; mpmath's
//! `legendre`, `chebyt`, `chebyu`, `gegenbauer`, `jacobi`, `laguerre`,
//! `hermite`):
//!
//! | function | value |
//! |---|---|
//! | `legendre(ν, x)` | `₂F₁(−ν, ν + 1; 1; (1 − x)/2)` |
//! | `chebyshev_t(ν, x)` | `cos(ν·acos x)` (`|x| ≤ 1`), `cosh(ν·acosh x)` (`x > 1`) |
//! | `chebyshev_u(ν, x)` | `sin((ν + 1)θ)/sin θ`, `θ = acos x`; `sinh((ν + 1)a)/sinh a`, `a = acosh x` |
//! | `gegenbauer(ν, α, x)` | `Γ(ν + 2α)/(Γ(ν + 1)Γ(2α))·₂F₁(−ν, ν + 2α; α + ½; (1 − x)/2)` |
//! | `jacobi(ν, α, β, x)` | `Γ(ν + α + 1)/(Γ(ν + 1)Γ(α + 1))·₂F₁(−ν, ν + α + β + 1; α + 1; (1 − x)/2)` |
//! | `laguerre(ν, x)` | `₁F₁(−ν; 1; x)` |
//! | `assoc_laguerre(ν, α, x)` | `Γ(ν + α + 1)/(Γ(ν + 1)Γ(α + 1))·₁F₁(−ν; α + 1; x)` |
//! | `hermite(ν, x)` | `2^ν√π·(₁F₁(−ν/2; ½; x²)/Γ((1 − ν)/2) − 2x·₁F₁((1 − ν)/2; 3/2; x²)/Γ(−ν/2))` |
//!
//! The ₂F₁ functions (and `chebyshev_t`, `chebyshev_u`) have a branch cut
//! on `x ≤ −1`, where the value is not real: those arguments stay refused,
//! as do parameters at a pole of a `Γ` or of the series (`α + ½`, `α + 1`
//! a non-positive integer).  For `x > 1` the ₂F₁ is taken through Pfaff's
//! transformation `₂F₁(a, b; c; z) = (1 − z)^(−a)·₂F₁(a, c − b; c;
//! z/(z − 1))` (DLMF 15.8.1), whose argument `(x − 1)/(x + 1)` lies in
//! `(0, 1)`; a ₁F₁ of a negative argument through Kummer's
//! `₁F₁(a; b; x) = e^x·₁F₁(b − a; b; −x)` (DLMF 13.2.39).  Up to 0.53 these
//! values were refused ("degree must be a non-negative integer").
//!
//! # Accuracy
//!
//! A series `Σ t_k` stops at the first `j` past the parameters' moduli
//! where the bound `ρ` of every later term ratio, `|z|·Π(1 + |aᵢ|/j)/Π(1 −
//! |bᵢ|/j)` (and `/j` for ₁F₁), is below 1 and the tail `|t_j|·ρ/(1 − ρ)`
//! is below `2^−wp` of the largest term.  The bits a result loses against
//! its largest term or the terms it is the difference of (a value near a
//! zero, `hermite` for a large `x`) are measured and the evaluation is
//! repeated with them; a series of more than [`MAX_TERMS`] terms (a ₂F₁
//! argument within `2^−12` of 1) is refused.  The node's bound is that of
//! the real routines (`sensitivity`: numerically in every argument).

use super::wp::{RM, Wp};
use crate::base::bigcomplex::Complex;
use crate::base::bigfloat::is_strictly_positive;
use crate::base::errors::SymplexError;
use crate::base::libfn::LibFn;
use crate::base::node::ExprId;
use astro_float::BigFloat;

use super::ball::Cache;

/// Most terms of one series, estimated before summing: a ₁F₁ needs about
/// `3|z| + wp` terms, a ₂F₁ `wp·ln 2/(−ln |z|)`; beyond the budget the
/// value is refused at once (the limit engine evaluates these functions at
/// large arguments, where a futile sum cost seconds each time).
const MAX_TERMS: usize = 8_000;

/// Does `f(args)` go to these routines: an orthogonal polynomial of the
/// families above with a real non-integer degree and real arguments?
pub(super) fn takes(f: LibFn, args: &[ExprId], cache: Cache<'_>) -> bool {
    if !matches!(
        f,
        LibFn::Legendre
            | LibFn::ChebyshevT
            | LibFn::ChebyshevU
            | LibFn::Gegenbauer
            | LibFn::Jacobi
            | LibFn::Laguerre
            | LibFn::AssocLaguerre
            | LibFn::Hermite
    ) {
        return false;
    }
    let vals: Option<Vec<&Complex>> = args.iter().map(|a| cache.get(a)).collect();
    let Some(vals) = vals else {
        return false;
    };
    vals.iter().all(|v| v.1.is_zero())
        && vals
            .first()
            .is_some_and(|n| !n.0.is_zero() && !n.0.is_int() && !n.0.is_nan() && !n.0.is_inf())
}

/// The value of `f(args)` (see [`takes`]), rounded to `prec` bits.
pub(super) fn eval(
    f: LibFn,
    args: &[ExprId],
    cache: Cache<'_>,
    w: &mut Wp<'_>,
) -> Result<Complex, SymplexError> {
    let prec = w.bits();
    let v: Vec<BigFloat> = args
        .iter()
        .map(|a| super::get_cached(cache, *a).map(|z| z.0.clone()))
        .collect::<Result<_, _>>()?;
    // Refusals name the function.
    let named = |e: SymplexError| match e {
        SymplexError::Unevaluable { reason } => {
            super::unevaluable(format!("{}: {reason}", f.name()))
        }
        other => other,
    };
    let mut extra = 32usize;
    for _ in 0..4 {
        let wp = prec + extra;
        let mut ww = w.at(wp);
        let (value, lost) = match f {
            LibFn::Legendre => legendre(&v[0], &v[1], &mut ww),
            LibFn::ChebyshevT => chebyshev(&v[0], &v[1], false, &mut ww),
            LibFn::ChebyshevU => chebyshev(&v[0], &v[1], true, &mut ww),
            LibFn::Gegenbauer => gegenbauer(&v[0], &v[1], &v[2], &mut ww),
            LibFn::Jacobi => jacobi(&v[0], &v[1], &v[2], &v[3], &mut ww),
            LibFn::Laguerre => {
                let zero = BigFloat::new(wp);
                laguerre(&v[0], &zero, &v[1], &mut ww)
            }
            LibFn::AssocLaguerre => laguerre(&v[0], &v[1], &v[2], &mut ww),
            LibFn::Hermite => hermite(&v[0], &v[1], &mut ww),
            _ => return Err(super::unevaluable("not an orthogonal polynomial")),
        }
        .map_err(named)?;
        if lost + 16 <= extra {
            return Ok((super::round_to(value, prec), BigFloat::new(prec)));
        }
        if lost > 4 * prec + 512 {
            break;
        }
        extra = lost + 32;
    }
    Err(super::unevaluable(format!(
        "{}: the value cancels beyond the working precision",
        f.name()
    )))
}

/// `log₂|x|` as an integer (`None` for 0).
fn mag(x: &BigFloat) -> Option<i64> {
    x.exponent().map(i64::from)
}

/// Bits lost forming `v` from terms of magnitude up to `2^top`.
fn lost(top: Option<i64>, v: &BigFloat) -> usize {
    match (top, mag(v)) {
        (Some(t), Some(m)) => usize::try_from(t - m).unwrap_or(0),
        (Some(_), None) => usize::MAX / 8,
        _ => 0,
    }
}

fn int(n: i32, wp: usize) -> BigFloat {
    BigFloat::from_i32(n, wp)
}

/// An upper bound of `|x|` in `f64` (`None` beyond its range).
fn abs_f64(x: &BigFloat) -> Option<f64> {
    let f = crate::base::bigfloat::to_f64(&x.abs())?;
    f.is_finite()
        .then_some(f * (1.0 + 1e-12) + f64::MIN_POSITIVE)
}

/// Is `x` a pole of `Γ` (0 or a negative integer)?
fn gamma_pole(x: &BigFloat) -> bool {
    x.is_zero() || (x.is_negative() && x.is_int())
}

/// `Σ_k Π(aᵢ)_k/Π(bᵢ)_k · z^k/k!` for `(p, q) = (2, 1)` with `|z| < 1`, or
/// `(1, 1)`, and the magnitude of its largest term.
fn series(
    a: &[&BigFloat],
    b: &[&BigFloat],
    z: &BigFloat,
    wp: usize,
) -> Result<(BigFloat, Option<i64>), SymplexError> {
    if b.iter().any(|x| gamma_pole(x)) {
        return Err(super::unevaluable(
            "hypergeometric series with a non-positive integer lower parameter",
        ));
    }
    let za = abs_f64(z).ok_or_else(|| super::unevaluable("series argument out of range"))?;
    let geometric = a.len() > b.len();
    if geometric && za >= 1.0 - 1.0 / 4096.0 {
        return Err(super::unevaluable(
            "hypergeometric series argument too close to 1",
        ));
    }
    let amax: Vec<f64> = a
        .iter()
        .map(|x| abs_f64(x).unwrap_or(f64::INFINITY))
        .collect();
    let bmax: Vec<f64> = b
        .iter()
        .map(|x| abs_f64(x).unwrap_or(f64::INFINITY))
        .collect();
    if amax
        .iter()
        .chain(&bmax)
        .any(|x| !x.is_finite() || *x > 1e15)
    {
        return Err(super::unevaluable("hypergeometric parameters out of range"));
    }
    let bfloor = bmax.iter().copied().fold(0.0, f64::max);
    let pmax = amax.iter().chain(&bmax).copied().fold(0.0, f64::max);
    let estimate = if geometric {
        (wp as f64 + 16.0) * std::f64::consts::LN_2 / -za.ln() + 4.0 * pmax + 16.0
    } else {
        3.0 * za + wp as f64 + 4.0 * pmax + 16.0
    };
    if estimate.is_nan() || estimate >= MAX_TERMS as f64 {
        return Err(super::unevaluable(
            "hypergeometric series beyond its term budget (argument too large or too close to 1)",
        ));
    }
    let mut term = int(1, wp);
    let mut sum = int(1, wp);
    let mut top = mag(&term);
    let threshold = -(wp as f64) - 2.0;
    for k in 0..MAX_TERMS {
        // t_{k+1} = t_k·Π(aᵢ + k)/Π(bᵢ + k)·z/(k + 1)
        let kb = BigFloat::from_u64(k as u64, wp);
        for x in a {
            term = term.mul(&x.add(&kb, wp, RM), wp, RM);
        }
        for x in b {
            term = term.div(&x.add(&kb, wp, RM), wp, RM);
        }
        term = term
            .mul(z, wp, RM)
            .div(&BigFloat::from_u64(k as u64 + 1, wp), wp, RM);
        if term.is_zero() {
            // A terminating series.
            return Ok((sum, top));
        }
        sum = sum.add(&term, wp, RM);
        top = top.max(mag(&term));
        // The bound of every later ratio, from j = k + 1 on.
        let j = (k + 1) as f64;
        if j <= 2.0 * bfloor + 1.0 {
            continue;
        }
        let mut rho = za;
        for &x in &amax {
            rho *= 1.0 + x / j;
        }
        for &x in &bmax {
            rho /= 1.0 - x / j;
        }
        if !geometric {
            rho /= j;
        }
        if rho < 1.0 {
            let tail =
                mag(&term).map_or(f64::NEG_INFINITY, |m| m as f64) + (rho / (1.0 - rho)).log2();
            let scale = top.map_or(0.0, |m| m as f64);
            if tail < scale + threshold {
                return Ok((sum, top));
            }
        }
    }
    Err(super::unevaluable(
        "hypergeometric series did not converge within its term budget",
    ))
}

/// `₂F₁(a, b; c; z)` at `z = (1 − x)/2` for a real `x > −1` (Pfaff's
/// transformation for `x > 1`), and the magnitude of its largest term.
fn f21_at(
    a: &BigFloat,
    b: &BigFloat,
    c: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<(BigFloat, Option<i64>), SymplexError> {
    let wp = w.bits();
    let one = int(1, wp);
    let two = int(2, wp);
    if !is_strictly_positive(&x.add(&one, wp, RM)) {
        return Err(super::unevaluable(
            "orthogonal polynomial of non-integer degree at x ≤ −1 (not real)",
        ));
    }
    let gt_one = x.cmp(&one).is_some_and(|s| s > 0);
    if !gt_one {
        let z = one.sub(x, wp, RM).div(&two, wp, RM);
        return series(&[a, b], &[c], &z, wp);
    }
    // (1 − z)^(−a)·₂F₁(a, c − b; c; z/(z − 1)), 1 − z = (1 + x)/2,
    // z/(z − 1) = (x − 1)/(x + 1).
    let u = x.sub(&one, wp, RM).div(&x.add(&one, wp, RM), wp, RM);
    let cb = c.sub(b, wp, RM);
    let (s, top) = series(&[a, &cb], &[c], &u, wp)?;
    let base = x.add(&one, wp, RM).div(&two, wp, RM);
    let factor = a
        .neg()
        .mul(&base.ln(wp, RM, w.cc), wp, RM)
        .exp(wp, RM, w.cc);
    let v = s.mul(&factor, wp, RM);
    let top = top.and_then(|t| mag(&factor).map(|f| t + f));
    Ok((v, top))
}

/// `Γ(p)/(Γ(q)Γ(r))` for real arguments, refused at a pole of `Γ(p)` and 0
/// at one of the denominator (not reached: callers refuse those).
fn gamma_ratio(
    p: &BigFloat,
    q: &BigFloat,
    r: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let wp = w.bits();
    if gamma_pole(p) || gamma_pole(q) || gamma_pole(r) {
        return Err(super::unevaluable(
            "orthogonal polynomial at a pole of its Γ factors",
        ));
    }
    // `p − q = d` a small integer and `r` a small positive integer (an
    // integer `α`): `(q)_d/(r − 1)!`, a product, without the Stirling series
    // (the series engine evaluates these prefactors thousands of times).
    let d = p.sub(q, wp, RM);
    if d.is_int() && r.is_int() && !r.is_negative() {
        let small = |x: &BigFloat| abs_f64(x).is_some_and(|v| v <= 64.5);
        if small(&d) && small(r) {
            let mut acc = int(1, wp);
            let mut t = q.clone();
            let up = !d.is_negative();
            let steps = crate::base::bigfloat::to_f64(&d.abs())
                .and_then(crate::base::cast::f64_to_i64)
                .and_then(|v| usize::try_from(v).ok())
                .unwrap_or(0);
            for _ in 0..steps {
                if up {
                    acc = acc.mul(&t, wp, RM);
                    t = t.add(&int(1, wp), wp, RM);
                } else {
                    t = t.sub(&int(1, wp), wp, RM);
                    acc = acc.div(&t, wp, RM);
                }
            }
            let mut k = int(1, wp);
            while k.cmp(r).is_some_and(|c| c < 0) {
                acc = acc.div(&k, wp, RM);
                k = k.add(&int(1, wp), wp, RM);
            }
            return Ok(acc);
        }
    }
    let gp = super::arb_gamma_real(p, w)?;
    let gq = super::arb_gamma_real(q, w)?;
    let gr = super::arb_gamma_real(r, w)?;
    Ok(gp.div(&gq.mul(&gr, wp, RM), wp, RM))
}

/// `P_ν(x) = ₂F₁(−ν, ν + 1; 1; (1 − x)/2)`.
fn legendre(
    nu: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<(BigFloat, usize), SymplexError> {
    let wp = w.bits();
    let a = nu.neg();
    let b = nu.add(&int(1, wp), wp, RM);
    let (v, top) = f21_at(&a, &b, &int(1, wp), x, w)?;
    let l = lost(top, &v);
    Ok((v, l))
}

/// `T_ν(x) = cos(ν·acos x)`, `U_ν(x) = sin((ν + 1)θ)/sin θ` for `−1 < x ≤ 1`,
/// and their hyperbolic forms for `x > 1`.
fn chebyshev(
    nu: &BigFloat,
    x: &BigFloat,
    second: bool,
    w: &mut Wp<'_>,
) -> Result<(BigFloat, usize), SymplexError> {
    let wp = w.bits();
    let one = int(1, wp);
    if !is_strictly_positive(&x.add(&one, wp, RM)) {
        return Err(super::unevaluable(
            "Chebyshev function of non-integer degree at x ≤ −1 (not real)",
        ));
    }
    let nu1 = nu.add(&one, wp, RM);
    if x.cmp(&one).is_some_and(|s| s == 0) {
        // T_ν(1) = 1, U_ν(1) = ν + 1.
        return Ok((if second { nu1 } else { one }, 0));
    }
    let gt_one = x.cmp(&one).is_some_and(|s| s > 0);
    let t = if gt_one {
        x.acosh(wp, RM, w.cc)
    } else {
        x.acos(wp, RM, w.cc)
    };
    let (arg, f) = if second {
        (nu1.mul(&t, wp, RM), true)
    } else {
        (nu.mul(&t, wp, RM), false)
    };
    let num = match (gt_one, f) {
        (true, true) => arg.sinh(wp, RM, w.cc),
        (true, false) => arg.cosh(wp, RM, w.cc),
        (false, true) => arg.sin(wp, RM, w.cc),
        (false, false) => arg.cos(wp, RM, w.cc),
    };
    // A value near a zero of the cosine or sine: its rounding is relative
    // to the argument.
    let l = if gt_one {
        0
    } else {
        lost(mag(&arg).map(|m| m.max(0)), &num)
    };
    if !second {
        return Ok((num, l));
    }
    let den = if gt_one {
        t.sinh(wp, RM, w.cc)
    } else {
        t.sin(wp, RM, w.cc)
    };
    Ok((num.div(&den, wp, RM), l))
}

/// `C_ν^(α)(x) = Γ(ν + 2α)/(Γ(ν + 1)Γ(2α))·₂F₁(−ν, ν + 2α; α + ½; (1 − x)/2)`.
fn gegenbauer(
    nu: &BigFloat,
    alpha: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<(BigFloat, usize), SymplexError> {
    let wp = w.bits();
    let two_a = alpha.mul(&int(2, wp), wp, RM);
    let b = nu.add(&two_a, wp, RM);
    let c = alpha.add(&BigFloat::from_f64(0.5, wp), wp, RM);
    let nu1 = nu.add(&int(1, wp), wp, RM);
    let g = gamma_ratio(&b, &nu1, &two_a, w)?;
    let (s, top) = f21_at(&nu.neg(), &b, &c, x, w)?;
    let l = lost(top, &s);
    Ok((g.mul(&s, wp, RM), l))
}

/// `P_ν^(α,β)(x) = Γ(ν + α + 1)/(Γ(ν + 1)Γ(α + 1))·₂F₁(−ν, ν + α + β + 1; α + 1; (1 − x)/2)`.
fn jacobi(
    nu: &BigFloat,
    alpha: &BigFloat,
    beta: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<(BigFloat, usize), SymplexError> {
    let wp = w.bits();
    let one = int(1, wp);
    let a1 = alpha.add(&one, wp, RM);
    let nu1 = nu.add(&one, wp, RM);
    let p = nu.add(&a1, wp, RM);
    let b = p.add(beta, wp, RM);
    let g = gamma_ratio(&p, &nu1, &a1, w)?;
    let (s, top) = f21_at(&nu.neg(), &b, &a1, x, w)?;
    let l = lost(top, &s);
    Ok((g.mul(&s, wp, RM), l))
}

/// `₁F₁(a; b; x)` (Kummer's transformation for `x < 0`) and the magnitude
/// of its largest term (times `e^x` after the transformation).
fn f11(
    a: &BigFloat,
    b: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<(BigFloat, Option<i64>), SymplexError> {
    let wp = w.bits();
    if !x.is_negative() {
        return series(&[a], &[b], x, wp);
    }
    let ba = b.sub(a, wp, RM);
    let (s, top) = series(&[&ba], &[b], &x.neg(), wp)?;
    let e = x.exp(wp, RM, w.cc);
    let top = top.and_then(|t| mag(&e).map(|m| t + m));
    Ok((s.mul(&e, wp, RM), top))
}

/// `L_ν^(α)(x) = Γ(ν + α + 1)/(Γ(ν + 1)Γ(α + 1))·₁F₁(−ν; α + 1; x)`
/// (`α = 0`: `₁F₁(−ν; 1; x)`).
fn laguerre(
    nu: &BigFloat,
    alpha: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<(BigFloat, usize), SymplexError> {
    let wp = w.bits();
    let one = int(1, wp);
    let a1 = alpha.add(&one, wp, RM);
    let (s, top) = f11(&nu.neg(), &a1, x, w)?;
    let l = lost(top, &s);
    if alpha.is_zero() {
        return Ok((s, l));
    }
    let p = nu.add(&a1, wp, RM);
    let nu1 = nu.add(&one, wp, RM);
    let g = gamma_ratio(&p, &nu1, &a1, w)?;
    Ok((g.mul(&s, wp, RM), l))
}

/// `H_ν(x) = 2^ν√π·(₁F₁(−ν/2; ½; x²)/Γ((1 − ν)/2) − 2x·₁F₁((1 − ν)/2; 3/2; x²)/Γ(−ν/2))`.
fn hermite(nu: &BigFloat, x: &BigFloat, w: &mut Wp<'_>) -> Result<(BigFloat, usize), SymplexError> {
    let wp = w.bits();
    let one = int(1, wp);
    let two = int(2, wp);
    let half = BigFloat::from_f64(0.5, wp);
    let x2 = x.mul(x, wp, RM);
    let a1 = nu.neg().div(&two, wp, RM);
    let a2 = one.sub(nu, wp, RM).div(&two, wp, RM);
    let (m1, t1) = f11(&a1, &half, &x2, w)?;
    let (m2, t2) = f11(&a2, &BigFloat::from_f64(1.5, wp), &x2, w)?;
    if gamma_pole(&a2) || gamma_pole(&a1) {
        return Err(super::unevaluable("hermite: integer degree"));
    }
    let g1 = super::arb_gamma_real(&a2, w)?;
    let g2 = super::arb_gamma_real(&a1, w)?;
    let p1 = m1.div(&g1, wp, RM);
    let p2 = two.mul(x, wp, RM).mul(&m2, wp, RM).div(&g2, wp, RM);
    let d = p1.sub(&p2, wp, RM);
    let top = [
        t1.and_then(|t| mag(&g1).map(|g| t - g)),
        t2.and_then(|t| mag(&p2).zip(mag(&m2)).map(|(p, m)| t + p - m)),
        mag(&p1),
        mag(&p2),
    ]
    .into_iter()
    .flatten()
    .max();
    let l = lost(top, &d);
    let scale = nu.mul(&two.ln(wp, RM, w.cc), wp, RM).exp(wp, RM, w.cc).mul(
        &w.cc.pi(wp, RM).clone().sqrt(wp, RM),
        wp,
        RM,
    );
    Ok((d.mul(&scale, wp, RM), l))
}

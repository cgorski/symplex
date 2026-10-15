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
//! `assoc_laguerre(ν, −k, x)` with `k` a positive integer has a pole of
//! `Γ(α + 1)` against one of the series (its lower parameter `1 − k`): the
//! product is the regularized function (DLMF 13.2.5), and with `(−ν)_k =
//! (−1)^kΓ(ν + 1)/Γ(ν − k + 1)` the factors cancel to
//! `L_ν^(−k)(x) = (−x)^k/k!·₁F₁(k − ν; k + 1; x)` (the polynomial identity
//! `L_n^(−k) = (−x)^k(n − k)!/n!·L_{n−k}^(k)` for any degree; mpmath:
//! `laguerre(1/2, −1, 3/2) = −2.361474300`).  Refused before 0.56.
//!
//! # Complex arguments
//!
//! `hermite`, `laguerre` and `assoc_laguerre` with a non-real argument (the
//! variable, the degree or `α`; refused before 0.56: "orthogonal polynomial
//! of complex arguments not supported") are entire in the variable and the
//! degree, and take the same formulas over ℂ: the series summed in complex
//! arithmetic with the bound of the term ratios from the moduli of the
//! parameters, Kummer's transformation for `Re z < 0` (not for a terminating
//! series), the Γ factors by the complex `Γ` (`zeta::gamma_complex`) and
//! `1/Γ` written as 0 at a pole (an integer degree selects the polynomial,
//! `H_{2m}(z) = (−1)^m(2m)!/m!·₁F₁(−m; ½; z²)`).  A real negative-integer
//! degree stays refused, as for real arguments.  Each part is computed to
//! its own precision (the bits a part lies below the modulus are added, up
//! to `sensitivity::gap_cap`), and the node's bound is
//! `sensitivity::complex_route_bound`.
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
use crate::base::bigcomplex::{Complex, c_add, c_div, c_mul, c_sub};
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
    if nonreal_args(f, args, cache) {
        return true;
    }
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

/// Does `f(args)` go to the complex routes: `hermite`, `laguerre` or
/// `assoc_laguerre` with a non-real argument, and a degree that is not a
/// real negative integer?
pub(super) fn nonreal_args(f: LibFn, args: &[ExprId], cache: Cache<'_>) -> bool {
    if !matches!(f, LibFn::Hermite | LibFn::Laguerre | LibFn::AssocLaguerre) {
        return false;
    }
    let vals: Option<Vec<&Complex>> = args.iter().map(|a| cache.get(a)).collect();
    let Some(vals) = vals else {
        return false;
    };
    let finite = |x: &BigFloat| !x.is_nan() && !x.is_inf();
    vals.iter().any(|v| !v.1.is_zero())
        && vals.iter().all(|v| finite(&v.0) && finite(&v.1))
        && vals
            .first()
            .is_some_and(|n| !(n.1.is_zero() && n.0.is_int() && n.0.is_negative()))
}

/// The value of `f(args)` (see [`takes`]), rounded to `prec` bits.
pub(super) fn eval(
    f: LibFn,
    args: &[ExprId],
    cache: Cache<'_>,
    w: &mut Wp<'_>,
) -> Result<Complex, SymplexError> {
    if nonreal_args(f, args, cache) {
        return eval_complex(f, args, cache, w);
    }
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

/// The value of `f(args)` with a non-real argument ([`nonreal_args`]), each
/// part to its own precision (up to `sensitivity::gap_cap` bits below the
/// modulus).
fn eval_complex(
    f: LibFn,
    args: &[ExprId],
    cache: Cache<'_>,
    w: &mut Wp<'_>,
) -> Result<Complex, SymplexError> {
    let prec = w.bits();
    let v: Vec<Complex> = args
        .iter()
        .map(|a| super::get_cached(cache, *a).cloned())
        .collect::<Result<_, _>>()?;
    let named = |e: SymplexError| match e {
        SymplexError::Unevaluable { reason } => {
            super::unevaluable(format!("{}: {reason}", f.name()))
        }
        other => other,
    };
    let cap = super::sensitivity::gap_cap(prec);
    let mut extra = 32usize;
    for _ in 0..4 {
        let wp = prec + extra;
        let mut ww = w.at(wp);
        let zero = (BigFloat::new(wp), BigFloat::new(wp));
        let (value, lost) = match f {
            LibFn::Laguerre => laguerre_c(&v[0], &zero, &v[1], &mut ww),
            LibFn::AssocLaguerre => laguerre_c(&v[0], &v[1], &v[2], &mut ww),
            LibFn::Hermite => hermite_c(&v[0], &v[1], &mut ww),
            _ => return Err(super::unevaluable("not an orthogonal polynomial")),
        }
        .map_err(named)?;
        let gap = part_gap(&value).min(cap);
        if lost + gap + 16 <= extra {
            return Ok((
                super::round_to(value.0, prec),
                super::round_to(value.1, prec),
            ));
        }
        if lost > 4 * prec + 512 {
            break;
        }
        extra = lost + gap + 32;
    }
    Err(super::unevaluable(format!(
        "{}: the value cancels beyond the working precision",
        f.name()
    )))
}

/// How many bits below the modulus of `v` its smaller nonzero part lies.
fn part_gap(v: &Complex) -> usize {
    match (nzmag(&v.0), nzmag(&v.1)) {
        (Some(a), Some(b)) => usize::try_from((a - b).abs()).unwrap_or(usize::MAX / 8),
        _ => 0,
    }
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
/// (`α = 0`: `₁F₁(−ν; 1; x)`; `α = −k` a negative integer: `(−x)^k/k!·
/// ₁F₁(k − ν; k + 1; x)`, see the module documentation).
fn laguerre(
    nu: &BigFloat,
    alpha: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<(BigFloat, usize), SymplexError> {
    let wp = w.bits();
    let one = int(1, wp);
    if let Some(k) = negative_integer(alpha) {
        let kf = BigFloat::from_u64(k, wp);
        let (s, top) = f11(&kf.sub(nu, wp, RM), &kf.add(&one, wp, RM), x, w)?;
        let mut pref = one.clone();
        for j in 1..=k {
            pref = pref
                .mul(&x.neg(), wp, RM)
                .div(&BigFloat::from_u64(j, wp), wp, RM);
        }
        let v = s.mul(&pref, wp, RM);
        let top = top.and_then(|t| mag(&pref).map(|p| t + p));
        return Ok((v.clone(), lost(top, &v)));
    }
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

/// `k` for `x = −k`, a negative integer (at most `2³²`), exactly.
fn negative_integer(x: &BigFloat) -> Option<u64> {
    if x.is_zero() || !x.is_int() || !x.is_negative() {
        return None;
    }
    crate::base::bigfloat::to_f64(&x.neg())
        .and_then(crate::base::cast::f64_to_i64)
        .and_then(|v| u64::try_from(v).ok())
        .filter(|&k| k <= 1 << 32)
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

// ── Complex arguments ─────────────────────────────────────────────────────────────────────

/// [`mag`], `None` for 0 (a part of a complex number may be 0 exactly).
fn nzmag(x: &BigFloat) -> Option<i64> {
    if x.is_zero() { None } else { mag(x) }
}

/// `log₂` of the larger part of `z` (`None` for 0).
fn cmag(z: &Complex) -> Option<i64> {
    match (nzmag(&z.0), nzmag(&z.1)) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (a, b) => a.or(b),
    }
}

/// Bits lost forming `v` from terms of magnitude up to `2^top`.
fn lost_c(top: Option<i64>, v: &Complex) -> usize {
    match (top, cmag(v)) {
        (Some(t), Some(m)) => usize::try_from(t - m).unwrap_or(0),
        (Some(_), None) => usize::MAX / 8,
        _ => 0,
    }
}

/// An upper bound of `|z|` in `f64` (`None` beyond its range).
fn cabs_f64(z: &Complex) -> Option<f64> {
    Some(abs_f64(&z.0)? + abs_f64(&z.1)?)
}

/// `z` real and a pole of `Γ`?
fn cgamma_pole(z: &Complex) -> bool {
    z.1.is_zero() && gamma_pole(&z.0)
}

fn cint(n: i32, wp: usize) -> Complex {
    (int(n, wp), BigFloat::new(wp))
}

/// `₁F₁(a; b; z) = Σ (a)_k/(b)_k·z^k/k!` over ℂ and the magnitude of its
/// largest term, stopped as [`series`] (the ratio bound from the moduli of
/// `a` and `b`: `|a + k| ≤ |a| + k`, `|b + k| ≥ k − |b|`).
fn series_c(
    a: &Complex,
    b: &Complex,
    z: &Complex,
    wp: usize,
) -> Result<(Complex, Option<i64>), SymplexError> {
    if cgamma_pole(b) {
        return Err(super::unevaluable(
            "hypergeometric series with a non-positive integer lower parameter",
        ));
    }
    let za = cabs_f64(z).ok_or_else(|| super::unevaluable("series argument out of range"))?;
    let amax = cabs_f64(a).unwrap_or(f64::INFINITY);
    let bmax = cabs_f64(b).unwrap_or(f64::INFINITY);
    if !amax.is_finite() || !bmax.is_finite() || amax.max(bmax) > 1e15 {
        return Err(super::unevaluable("hypergeometric parameters out of range"));
    }
    let estimate = 3.0 * za + wp as f64 + 4.0 * amax.max(bmax) + 16.0;
    if estimate.is_nan() || estimate >= MAX_TERMS as f64 {
        return Err(super::unevaluable(
            "hypergeometric series beyond its term budget (argument too large)",
        ));
    }
    let mut term = cint(1, wp);
    let mut sum = cint(1, wp);
    let mut top = cmag(&term);
    let threshold = -(wp as f64) - 2.0;
    for k in 0..MAX_TERMS {
        let kb = (BigFloat::from_u64(k as u64, wp), BigFloat::new(wp));
        let num = c_add(a, &kb, wp, RM);
        let den = c_add(b, &kb, wp, RM);
        term = c_div(&c_mul(&term, &num, wp, RM), &den, wp, RM);
        let k1 = BigFloat::from_u64(k as u64 + 1, wp);
        term = c_mul(&term, z, wp, RM);
        term = (term.0.div(&k1, wp, RM), term.1.div(&k1, wp, RM));
        if term.0.is_zero() && term.1.is_zero() {
            return Ok((sum, top));
        }
        sum = c_add(&sum, &term, wp, RM);
        top = top.max(cmag(&term));
        let j = (k + 1) as f64;
        if j <= 2.0 * bmax + 1.0 {
            continue;
        }
        let rho = za * (1.0 + amax / j) / (1.0 - bmax / j) / j;
        if rho < 1.0 {
            let tail =
                cmag(&term).map_or(f64::NEG_INFINITY, |m| m as f64) + (rho / (1.0 - rho)).log2();
            if tail < top.map_or(0.0, |m| m as f64) + threshold {
                return Ok((sum, top));
            }
        }
    }
    Err(super::unevaluable(
        "hypergeometric series did not converge within its term budget",
    ))
}

/// `₁F₁(a; b; z)` over ℂ, by Kummer's transformation for `Re z < 0` unless
/// the series terminates (`a` a non-positive integer), and the magnitude of
/// its largest term (times `|e^z|` after the transformation).
fn f11_c(
    a: &Complex,
    b: &Complex,
    z: &Complex,
    w: &mut Wp<'_>,
) -> Result<(Complex, Option<i64>), SymplexError> {
    let wp = w.bits();
    if !z.0.is_negative() || cgamma_pole(a) {
        return series_c(a, b, z, wp);
    }
    let ba = c_sub(b, a, wp, RM);
    let (s, top) = series_c(&ba, b, &(z.0.neg(), z.1.neg()), wp)?;
    let e = super::c_exp(z, w);
    let top = top.and_then(|t| cmag(&e).map(|m| t + m + 1));
    Ok((c_mul(&s, &e, wp, RM), top))
}

/// `Γ(z)` (`None` at a pole), real or complex.
fn cgamma(z: &Complex, w: &mut Wp<'_>) -> Result<Option<Complex>, SymplexError> {
    let wp = w.bits();
    if cgamma_pole(z) {
        return Ok(None);
    }
    if z.1.is_zero() {
        return Ok(Some((super::arb_gamma_real(&z.0, w)?, BigFloat::new(wp))));
    }
    super::zeta::gamma_complex(z, w).map(Some)
}

/// `1/Γ(z)`, 0 at a pole.
fn crgamma(z: &Complex, w: &mut Wp<'_>) -> Result<Complex, SymplexError> {
    let wp = w.bits();
    Ok(match cgamma(z, w)? {
        Some(g) => c_div(&cint(1, wp), &g, wp, RM),
        None => cint(0, wp),
    })
}

/// `L_ν^(α)(z)` over ℂ (see [`laguerre`]), and the bits lost to the largest
/// term.
fn laguerre_c(
    nu: &Complex,
    alpha: &Complex,
    z: &Complex,
    w: &mut Wp<'_>,
) -> Result<(Complex, usize), SymplexError> {
    let wp = w.bits();
    let one = cint(1, wp);
    let neg_k = if alpha.1.is_zero() {
        negative_integer(&alpha.0)
    } else {
        None
    };
    if let Some(k) = neg_k {
        let kc = (BigFloat::from_u64(k, wp), BigFloat::new(wp));
        let (s, top) = f11_c(&c_sub(&kc, nu, wp, RM), &c_add(&kc, &one, wp, RM), z, w)?;
        let mut pref = one.clone();
        let mz = (z.0.neg(), z.1.neg());
        for j in 1..=k {
            let jb = BigFloat::from_u64(j, wp);
            pref = c_mul(&pref, &mz, wp, RM);
            pref = (pref.0.div(&jb, wp, RM), pref.1.div(&jb, wp, RM));
        }
        let v = c_mul(&s, &pref, wp, RM);
        let top = top.and_then(|t| cmag(&pref).map(|p| t + p + 1));
        return Ok((v.clone(), lost_c(top, &v)));
    }
    let a1 = c_add(alpha, &one, wp, RM);
    let (s, top) = f11_c(&(nu.0.neg(), nu.1.neg()), &a1, z, w)?;
    let l = lost_c(top, &s);
    if alpha.0.is_zero() && alpha.1.is_zero() {
        return Ok((s, l));
    }
    let p = c_add(nu, &a1, wp, RM);
    let nu1 = c_add(nu, &one, wp, RM);
    let g = if [nu, alpha].iter().all(|v| v.1.is_zero()) {
        (gamma_ratio(&p.0, &nu1.0, &a1.0, w)?, BigFloat::new(wp))
    } else {
        let Some(gp) = cgamma(&p, w)? else {
            return Err(super::unevaluable(
                "orthogonal polynomial at a pole of its Γ factors",
            ));
        };
        c_mul(
            &gp,
            &c_mul(&crgamma(&nu1, w)?, &crgamma(&a1, w)?, wp, RM),
            wp,
            RM,
        )
    };
    Ok((c_mul(&g, &s, wp, RM), l))
}

/// `H_ν(z)` over ℂ (see [`hermite`]; `1/Γ` is 0 at a pole, so an integer
/// degree gives the polynomial), and the bits lost to the largest term.
fn hermite_c(nu: &Complex, z: &Complex, w: &mut Wp<'_>) -> Result<(Complex, usize), SymplexError> {
    let wp = w.bits();
    let half = BigFloat::from_f64(0.5, wp);
    let z2 = c_mul(z, z, wp, RM);
    let a1 = (nu.0.mul(&half, wp, RM).neg(), nu.1.mul(&half, wp, RM).neg());
    let a2 = c_add(&a1, &(half.clone(), BigFloat::new(wp)), wp, RM);
    let r1 = crgamma(&a2, w)?;
    let r2 = crgamma(&a1, w)?;
    let zero = |c: &Complex| c.0.is_zero() && c.1.is_zero();
    let mut tops = Vec::new();
    let p1 = if zero(&r1) {
        cint(0, wp)
    } else {
        let (m1, t1) = f11_c(&a1, &(half.clone(), BigFloat::new(wp)), &z2, w)?;
        tops.push(t1.and_then(|t| cmag(&r1).map(|r| t + r + 1)));
        c_mul(&m1, &r1, wp, RM)
    };
    let p2 = if zero(&r2) {
        cint(0, wp)
    } else {
        let three_half = (BigFloat::from_f64(1.5, wp), BigFloat::new(wp));
        let (m2, t2) = f11_c(&a2, &three_half, &z2, w)?;
        let f = c_mul(&c_add(z, z, wp, RM), &r2, wp, RM);
        tops.push(t2.and_then(|t| cmag(&f).map(|r| t + r + 1)));
        c_mul(&m2, &f, wp, RM)
    };
    let d = c_sub(&p1, &p2, wp, RM);
    tops.push(cmag(&p1));
    tops.push(cmag(&p2));
    let top = tops.into_iter().flatten().max();
    let l = lost_c(top, &d);
    // 2^ν√π = e^(ν ln 2)√π.
    let ln2 = int(2, wp).ln(wp, RM, w.cc);
    let e = super::c_exp(&(nu.0.mul(&ln2, wp, RM), nu.1.mul(&ln2, wp, RM)), w);
    let sqrt_pi = w.cc.pi(wp, RM).clone().sqrt(wp, RM);
    let scale = (e.0.mul(&sqrt_pi, wp, RM), e.1.mul(&sqrt_pi, wp, RM));
    Ok((c_mul(&d, &scale, wp, RM), l))
}

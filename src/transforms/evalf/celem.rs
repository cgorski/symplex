//! Complex arithmetic helpers (part of [`evalf`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Complex arithmetic helpers
// ═══════════════════════════════════════════════════════════════════════════

pub(super) fn c_exp(z: &Complex, w: &mut Wp<'_>) -> Complex {
    let prec = w.bits();
    // exp(a+bi) = exp(a)(cos(b) + i·sin(b))
    let exp_a = z.0.exp(prec, RM, w.cc);
    let (sin_b, cos_b) = sin_cos(&z.1, w);
    (exp_a.mul(&cos_b, prec, RM), exp_a.mul(&sin_b, prec, RM))
}

/// `(sin x, cos x)`, exact at `x ∈ {0, ±π/2, ±π}` when `x` is bit for bit
/// the working-precision constant.
///
/// The evaluator produces exactly those constants itself — `arg` of a
/// negative real is π, `asin` of a real beyond 1 has real part ±π/2 — and
/// `sin` of the *rounded* π is a residue of about `2^(−prec)`, not 0.
/// Carried on, that residue picks the side of the next branch cut by its
/// sign: `exp(ln(−5/7))` came out `−5/7 + 1e-50·i`, and
/// `acosh(1/exp(ln x))` at `x = −5/7` the conjugate of `acosh(−7/5)`.  A
/// value equal to the rounded π is indistinguishable from π at this
/// precision, so its exact sine is the right answer.
pub(super) fn sin_cos(x: &BigFloat, w: &mut Wp<'_>) -> (BigFloat, BigFloat) {
    let prec = w.bits();
    let zero = BigFloat::new(prec);
    let one = BigFloat::from_i32(1, prec);
    if x.is_zero() {
        return (zero, one);
    }
    let pi = w.cc.pi(prec, RM).clone();
    let half_pi = pi.div(&BigFloat::from_i32(2, prec), prec, RM);
    if *x == half_pi {
        return (one, zero);
    }
    if *x == half_pi.neg() {
        return (one.neg(), zero);
    }
    if *x == pi || *x == pi.neg() {
        return (zero, one.neg());
    }
    (x.sin(prec, RM, w.cc), x.cos(prec, RM, w.cc))
}

pub(super) fn c_ln(z: &Complex, w: &mut Wp<'_>) -> Complex {
    let prec = w.bits();
    // ln(z) = ln|z| + i·arg(z)
    let r = c_abs(z, prec, RM);
    let theta = atan2_bf(&z.1, &z.0, w);
    (r.ln(prec, RM, w.cc), theta)
}

pub(super) fn c_sin(z: &Complex, w: &mut Wp<'_>) -> Complex {
    let prec = w.bits();
    // sin(a+bi) = sin(a)cosh(b) + i·cos(a)sinh(b)
    let (sin_a, cos_a) = sin_cos(&z.0, w);
    let cosh_b = z.1.cosh(prec, RM, w.cc);
    let sinh_b = z.1.sinh(prec, RM, w.cc);
    (sin_a.mul(&cosh_b, prec, RM), cos_a.mul(&sinh_b, prec, RM))
}

pub(super) fn c_cos(z: &Complex, w: &mut Wp<'_>) -> Complex {
    let prec = w.bits();
    // cos(a+bi) = cos(a)cosh(b) - i·sin(a)sinh(b)
    let (sin_a, cos_a) = sin_cos(&z.0, w);
    let cosh_b = z.1.cosh(prec, RM, w.cc);
    let sinh_b = z.1.sinh(prec, RM, w.cc);
    (
        cos_a.mul(&cosh_b, prec, RM),
        sin_a.mul(&sinh_b, prec, RM).neg(),
    )
}

pub(super) fn c_sinh(z: &Complex, w: &mut Wp<'_>) -> Complex {
    let prec = w.bits();
    // sinh(a+bi) = sinh(a)cos(b) + i·cosh(a)sin(b)
    let sinh_a = z.0.sinh(prec, RM, w.cc);
    let cosh_a = z.0.cosh(prec, RM, w.cc);
    let (sin_b, cos_b) = sin_cos(&z.1, w);
    (sinh_a.mul(&cos_b, prec, RM), cosh_a.mul(&sin_b, prec, RM))
}

pub(super) fn c_cosh(z: &Complex, w: &mut Wp<'_>) -> Complex {
    let prec = w.bits();
    // cosh(a+bi) = cosh(a)cos(b) + i·sinh(a)sin(b)
    let cosh_a = z.0.cosh(prec, RM, w.cc);
    let sinh_a = z.0.sinh(prec, RM, w.cc);
    let (sin_b, cos_b) = sin_cos(&z.1, w);
    (cosh_a.mul(&cos_b, prec, RM), sinh_a.mul(&sin_b, prec, RM))
}

pub(super) fn c_pow(base: &Complex, exp: &Complex, w: &mut Wp<'_>) -> Complex {
    let prec = w.bits();
    // base^exp = exp(exp * ln(base))
    // Special case: base is zero.
    if base.0.is_zero() && base.1.is_zero() {
        // 0^0 = 1 by convention; 0^(positive) = 0.
        if exp.0.is_zero() && exp.1.is_zero() {
            return c_one(prec);
        }
        return c_zero(prec);
    }
    let ln_base = c_ln(base, w);
    let product = c_mul(exp, &ln_base, prec, RM);
    c_exp(&product, w)
}

pub(super) fn c_sqrt(z: &Complex, w: &mut Wp<'_>) -> Complex {
    let prec = w.bits();
    if z.0.is_zero() && z.1.is_zero() {
        return c_zero(prec);
    }
    let half = c_from_real(
        BigFloat::from_i32(1, prec).div(&BigFloat::from_i32(2, prec), prec, RM),
        prec,
    );
    c_pow(z, &half, w)
}

// ── Inverse trig (complex) ─────────────────────────────────────────

/// `asin`, `atan`, `asinh` and `atanh` of a complex `z` are `z·(1 + O(z²))`,
/// but their logarithmic formulas take `ln(1 + O(z))`: rounding `1 ± iz`
/// to `prec` bits is an absolute error of `2^−prec`, `log₂(1/|z|)` bits of
/// the value that the node's bound (its rounding at the value's magnitude)
/// does not know of.  `f(z, prec)` is evaluated with that many more bits and
/// rounded; below `2^−prec` the value is `z` itself.  Before 0.30 neither:
/// `atan(i·10⁻³⁰)` was `9.99999998762973·10⁻³¹·i` at 16 digits (truly
/// `1·10⁻³⁰·i`).
fn small_argument(z: &Complex, w: &mut Wp<'_>, f: fn(&Complex, &mut Wp<'_>) -> Complex) -> Complex {
    let prec = w.bits();
    let m = accuracy::mag(z).unwrap_or(0);
    if m >= 0 {
        return f(z, w);
    }
    let lost = usize::try_from(-m).unwrap_or(usize::MAX);
    if lost > prec + 8 {
        return (round_to(z.0.clone(), prec), round_to(z.1.clone(), prec));
    }
    let v = f(z, &mut w.at(prec + lost + 16));
    (round_to(v.0, prec), round_to(v.1, prec))
}

pub(super) fn c_asin(z: &Complex, w: &mut Wp<'_>) -> Complex {
    small_argument(z, w, c_asin_at)
}

fn c_asin_at(z: &Complex, w: &mut Wp<'_>) -> Complex {
    let prec = w.bits();
    // asin(z) = −i·ln(iz + √(1 − z²)).  In half the plane the two terms
    // nearly cancel for large |z|; there the sum is taken as 1/(√(1 − z²) −
    // iz), the same number ((iz + s)(s − iz) = 1), without the cancellation.
    let i_unit = c_i(prec);
    let one = c_one(prec);
    let z_sq = c_mul(z, z, prec, RM);
    let one_minus_z_sq = c_sub(&one, &z_sq, prec, RM);
    let sqrt_term = c_sqrt(&one_minus_z_sq, w);
    let iz = c_mul(&i_unit, z, prec, RM);
    let sum = sum_without_cancellation(&iz, &sqrt_term, prec);
    let ln_sum = c_ln(&sum, w);
    let neg_i = c_neg(&i_unit);
    c_mul(&neg_i, &ln_sum, prec, RM)
}

/// `a + s` for `a`, `s` with `s² − a² = 1` (then `(a + s)(s − a) = 1`):
/// the direct sum when it does not cancel, `1/(s − a)` when it does.
fn sum_without_cancellation(a: &Complex, s: &Complex, prec: usize) -> Complex {
    let plus = c_add(a, s, prec, RM);
    let minus = c_sub(s, a, prec, RM);
    if c_abs(&plus, prec, RM) >= c_abs(&minus, prec, RM) {
        plus
    } else {
        c_div(&c_one(prec), &minus, prec, RM)
    }
}

pub(super) fn c_acos(z: &Complex, w: &mut Wp<'_>) -> Complex {
    let prec = w.bits();
    // acos(z) = pi/2 - asin(z)
    let half_pi_val =
        w.cc.pi(prec, RM)
            .div(&BigFloat::from_i32(2, prec), prec, RM);
    let half_pi = c_from_real(half_pi_val, prec);
    let asin_z = c_asin(z, w);
    c_sub(&half_pi, &asin_z, prec, RM)
}

pub(super) fn c_atan(z: &Complex, w: &mut Wp<'_>) -> Complex {
    small_argument(z, w, c_atan_at)
}

fn c_atan_at(z: &Complex, w: &mut Wp<'_>) -> Complex {
    let prec = w.bits();
    // atan(z) = (i/2)·(ln(1 − iz) − ln(1 + iz)).  The quotient form
    // (i/2)·ln((i + z)/(i − z)) differs from it by π on the cut
    // z ∈ i·(1, ∞) (atan(2i) would be −π/2 + …, mpmath says +π/2 + …).
    let one = c_one(prec);
    let iz = c_mul(&c_i(prec), z, prec, RM);
    let ln_minus = c_ln(&c_sub(&one, &iz, prec, RM), w);
    let ln_plus = c_ln(&c_add(&one, &iz, prec, RM), w);
    let diff = c_sub(&ln_minus, &ln_plus, prec, RM);
    let half = BigFloat::from_i32(1, prec).div(&BigFloat::from_i32(2, prec), prec, RM);
    // (i/2)·(a + bi) = −b/2 + (a/2)·i
    (
        diff.1.mul(&half, prec, RM).neg(),
        diff.0.mul(&half, prec, RM),
    )
}

// ── Inverse hyperbolic (complex) ──────────────────────────────────

pub(super) fn c_asinh(z: &Complex, w: &mut Wp<'_>) -> Complex {
    small_argument(z, w, c_asinh_at)
}

fn c_asinh_at(z: &Complex, w: &mut Wp<'_>) -> Complex {
    let prec = w.bits();
    // asinh(z) = ln(z + √(z² + 1)), the sum taken without cancellation
    // (it cancels for Re z < 0 and large |z|).
    let one = c_one(prec);
    let z_sq = c_mul(z, z, prec, RM);
    let z_sq_plus_one = c_add(&z_sq, &one, prec, RM);
    let sqrt_term = c_sqrt(&z_sq_plus_one, w);
    let sum = sum_without_cancellation(z, &sqrt_term, prec);
    c_ln(&sum, w)
}

pub(super) fn c_acosh(z: &Complex, w: &mut Wp<'_>) -> Complex {
    let prec = w.bits();
    // acosh(z) = ln(z + √(z + 1)·√(z − 1)).  The single root √(z² − 1)
    // is the other branch whenever Re z < 0 (acosh(−2 + i/2) came out
    // negated); the product of two roots is the principal value everywhere.
    let one = c_one(prec);
    let root_plus = c_sqrt(&c_add(z, &one, prec, RM), w);
    let root_minus = c_sqrt(&c_sub(z, &one, prec, RM), w);
    let sum = c_add(z, &c_mul(&root_plus, &root_minus, prec, RM), prec, RM);
    c_ln(&sum, w)
}

pub(super) fn c_atanh(z: &Complex, w: &mut Wp<'_>) -> Complex {
    small_argument(z, w, c_atanh_at)
}

fn c_atanh_at(z: &Complex, w: &mut Wp<'_>) -> Complex {
    let prec = w.bits();
    // atanh(z) = (ln(1 + z) − ln(1 − z))/2.  The quotient form
    // ln((1 + z)/(1 − z))/2 differs from it by πi on the cut z ∈ (1, ∞).
    let one = c_one(prec);
    let ln_plus = c_ln(&c_add(&one, z, prec, RM), w);
    let ln_minus = c_ln(&c_sub(&one, z, prec, RM), w);
    let diff = c_sub(&ln_plus, &ln_minus, prec, RM);
    let half = BigFloat::from_i32(1, prec).div(&BigFloat::from_i32(2, prec), prec, RM);
    (diff.0.mul(&half, prec, RM), diff.1.mul(&half, prec, RM))
}

/// The real part of a trigonometric node's argument, and the bits it
/// carries beyond `prec`.  A rational child (`sin(10^100)`) is converted
/// again with `log₂|r|` extra bits, so that reducing it modulo 2π starts
/// from a value correct to `prec` bits after the point, not one rounded to
/// `prec` significant bits.
pub(super) fn trig_arg(
    arena: &Arena,
    inner: ExprId,
    val: &Complex,
    prec: usize,
) -> (BigFloat, usize) {
    if let ExprNode::Num(nid) = arena.node(inner)
        && let Some(e) = val.0.exponent()
        && e > 0
        && val.0.inexact()
    {
        let extra = usize::try_from(e).unwrap_or(0);
        if extra <= arena.config.max_evalf_precision as usize {
            return (ratio_to_bigfloat(arena.num(*nid), prec + extra), extra);
        }
    }
    (val.0.clone(), 0)
}

/// π/2 at `prec` bits.
pub(super) fn half_pi_bf(w: &mut Wp<'_>) -> BigFloat {
    let prec = w.bits();
    w.cc.pi(prec, RM)
        .div(&BigFloat::from_i32(2, prec), prec, RM)
}

/// `|x| ≤ 1`: the real domain of `asin`, `acos` and (open) `atanh`.
pub(super) fn abs_le_one(x: &BigFloat, prec: usize) -> bool {
    x.abs() <= BigFloat::from_i32(1, prec)
}

/// Binary exponent above which an *inexact* trigonometric argument is
/// refused.  Reducing `x` modulo 2π turns its relative rounding error into
/// an absolute error of about `2^(e − prec)`, and [`evalf`] carries at least
/// 64 guard bits beyond the digits requested, so past `2^64` the reduction
/// eats into the digits that are printed.
const TRIG_ARG_MAX_INEXACT_EXPONENT: i64 = 64;

/// Is the argument `c` of a trigonometric node one that [`check_trig_arg`]
/// refuses at every precision although its error bound is tracked: real,
/// inexact, beyond `2⁶⁴` (and within the configured maximum precision),
/// and not a rational literal (which [`trig_arg`] converts again with more
/// bits)?  `eval_ball` evaluates it from its value and bound.
pub(super) fn large_inexact_trig_argument(arena: &Arena, c: ExprId, cache: Cache<'_>) -> bool {
    if matches!(arena.node(c), ExprNode::Num(_)) {
        return false;
    }
    let Some(v) = cache.get(&c) else {
        return false;
    };
    if !v.1.is_zero() || !v.0.inexact() || v.0.is_nan() || v.0.is_inf() {
        return false;
    }
    v.0.exponent().is_some_and(|e| {
        let e = i64::from(e);
        e > TRIG_ARG_MAX_INEXACT_EXPONENT && e <= i64::from(arena.config.max_evalf_precision)
    })
}

/// Refuse a trigonometric argument whose value is not known well enough to
/// reduce it modulo 2π at this precision: an inexact one of magnitude
/// ≥ `2^64` (see [`TRIG_ARG_MAX_INEXACT_EXPONENT`]; `exp(100) ≈ 2^144` at
/// 30 digits is known only to within about `2^(144 − 166)`), or any beyond
/// `EvalConfig::max_evalf_precision` bits — astro-float would compute π to
/// that many bits, and `cos(cosh(cosh(4i)))` ran for minutes.
///
/// `extra` is the number of bits `x` carries beyond `prec` (see
/// [`trig_arg`]); they widen the inexact bound by as much.
pub(super) fn check_trig_arg(
    x: &BigFloat,
    extra: usize,
    prec: usize,
    arena: &Arena,
) -> Result<(), SymplexError> {
    let Some(e) = x.exponent() else {
        return Ok(());
    };
    let e = i64::from(e);
    let cap = i64::from(arena.config.max_evalf_precision);
    let bound =
        TRIG_ARG_MAX_INEXACT_EXPONENT.saturating_add(i64::try_from(extra).unwrap_or(i64::MAX));
    if e > cap || (x.inexact() && e > bound) {
        let digits = u32::try_from(prec.saturating_sub(64) * 10 / 34).unwrap_or(u32::MAX);
        return Err(SymplexError::PrecisionExhausted {
            requested: digits,
            achieved: 0,
        });
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Real-valued helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Compute atan2(y, x) using BigFloat arithmetic.
pub(super) fn atan2_bf(y: &BigFloat, x: &BigFloat, w: &mut Wp<'_>) -> BigFloat {
    let prec = w.bits();
    if x.is_zero() && y.is_zero() {
        return BigFloat::new(prec);
    }
    // On the real axis the sign of a zero `y` is an accident of how the
    // value was computed (`c_neg` turns `+0` into `−0`), not a side of the
    // cut: a negative real has argument +π on the principal branch.
    if y.is_zero() {
        return if x.is_negative() {
            w.cc.pi(prec, RM).clone()
        } else {
            BigFloat::new(prec)
        };
    }

    let pi_val = w.cc.pi(prec, RM);
    let two = BigFloat::from_i32(2, prec);
    let half_pi = pi_val.div(&two, prec, RM);

    if x.is_zero() {
        return if is_sign_positive(y) {
            half_pi
        } else {
            half_pi.neg()
        };
    }

    let ratio = y.div(x, prec, RM);
    let atan_val = ratio.atan(prec, RM, w.cc);

    if is_strictly_positive(x) {
        atan_val
    } else {
        let pi_val2 = w.cc.pi(prec, RM);
        if y.is_negative() {
            atan_val.sub(&pi_val2, prec, RM)
        } else {
            atan_val.add(&pi_val2, prec, RM)
        }
    }
}

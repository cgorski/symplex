//! Arbitrary-precision numerical evaluation.
//!
//! This module implements [`evalf`], which evaluates a symbolic expression
//! to a decimal string with a specified number of significant digits,
//! using the [`astro_float`] crate for arbitrary-precision arithmetic.
//!
//! # Design
//!
//! Evaluation is performed bottom-up using an explicit post-order
//! traversal (no recursion).  Each sub-expression is evaluated to a
//! [`BigFloat`] and cached, so shared sub-expressions (common in a
//! hash-consed DAG) are only evaluated once.
//!
//! The working precision is set higher than the requested output
//! precision to absorb rounding errors from intermediate computations.
//! If a sub-expression cannot be evaluated (e.g., it contains free
//! symbols), the function returns an error.

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::Zero;
use rustc_hash::FxHashMap;

use astro_float::{BigFloat, Consts, Radix, RoundingMode, Sign};

use crate::arena::Arena;
use crate::errors::SymplexError;
use crate::node::{ExprId, ExprNode};
use crate::walk;
use tracing::debug;

// ═══════════════════════════════════════════════════════════════════════════
// Public entry point
// ═══════════════════════════════════════════════════════════════════════════

/// Evaluate `expr` numerically to `digits` decimal digits of precision.
///
/// Returns the decimal string representation of the result, or an error
/// if the expression contains free symbols, infinities, or NaN.
pub(crate) fn evalf(arena: &Arena, expr: ExprId, digits: u32) -> Result<String, SymplexError> {
    // Convert decimal digits to binary precision with guard bits.
    // log2(10) ≈ 3.3219, so we use digits * 3.4 + 64 extra bits.
    let binary_prec = (digits as usize) * 34 / 10 + 64;
    // Ensure a reasonable minimum.
    let prec = binary_prec.max(128);

    debug!(
        digits = digits,
        binary_precision = prec,
        "evalf: starting numerical evaluation"
    );

    // Enforce the configured maximum precision.
    let max_prec = arena.config.max_evalf_precision as usize;
    if prec > max_prec {
        return Err(SymplexError::PrecisionExhausted {
            requested: digits,
            achieved: (max_prec * 10 / 34).saturating_sub(6) as u32,
        });
    }

    let rm = RoundingMode::ToEven;
    let mut cc = Consts::new().map_err(|e| {
        SymplexError::NotImplemented(format!("astro-float constants init failed: {e:?}"))
    })?;

    // Post-order traversal — evaluate children before parents.
    let post_order = walk::post_order_ids(arena, expr);
    let mut cache: FxHashMap<ExprId, BigFloat> = FxHashMap::default();

    for &id in &post_order {
        let value = eval_node(arena, id, &cache, prec, rm, &mut cc)?;
        cache.insert(id, value);
    }

    let result = cache.get(&expr).ok_or_else(|| {
        SymplexError::NotImplemented("evalf: expression not found in cache".into())
    })?;

    if result.is_nan() {
        return Err(SymplexError::PrecisionExhausted {
            requested: digits,
            achieved: 0,
        });
    }

    format_decimal(result, digits, rm, &mut cc)
}

// ═══════════════════════════════════════════════════════════════════════════
// Node evaluation
// ═══════════════════════════════════════════════════════════════════════════

/// Evaluate a single node given its children's values in the cache.
fn eval_node(
    arena: &Arena,
    id: ExprId,
    cache: &FxHashMap<ExprId, BigFloat>,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<BigFloat, SymplexError> {
    let node = arena.node(id);

    match node {
        // ── Atoms ──────────────────────────────────────────────────
        ExprNode::Num(nid) => {
            let r = arena.num(*nid);
            Ok(ratio_to_bigfloat(r, prec, rm))
        }

        ExprNode::Pi => Ok(cc.pi(prec, rm).clone()),

        ExprNode::E => Ok(cc.e(prec, rm).clone()),

        ExprNode::ImaginaryUnit => Err(SymplexError::Unevaluable {
            reason: "complex numbers not yet supported".into(),
        }),

        ExprNode::Infinity | ExprNode::NegInfinity | ExprNode::ComplexInfinity => {
            Err(SymplexError::Unevaluable {
                reason: "cannot evaluate infinity to finite precision".into(),
            })
        }

        ExprNode::NaN => Err(SymplexError::Unevaluable {
            reason: "cannot evaluate NaN".into(),
        }),

        ExprNode::Symbol(sid) => {
            let name = arena.symbol_name(*sid);
            Err(SymplexError::FreeSymbol {
                name: name.to_owned(),
            })
        }

        // ── Add ────────────────────────────────────────────────────
        ExprNode::Add(children) => {
            let mut sum = BigFloat::from_i32(0, prec);
            for &child in children.iter() {
                let val = get_cached(cache, child)?;
                sum = sum.add(val, prec, rm);
            }
            Ok(sum)
        }

        // ── Mul ────────────────────────────────────────────────────
        ExprNode::Mul(children) => {
            let mut product = BigFloat::from_i32(1, prec);
            for &child in children.iter() {
                let val = get_cached(cache, child)?;
                product = product.mul(val, prec, rm);
            }
            Ok(product)
        }

        // ── Pow ────────────────────────────────────────────────────
        ExprNode::Pow(base, exp) => {
            let b = get_cached(cache, *base)?;
            let e = get_cached(cache, *exp)?;

            // Special case: small integer exponents use powi for accuracy.
            if let Some(n) = try_as_small_int(arena, *exp) {
                if n >= 0 {
                    return Ok(b.powi(n as usize, prec, rm));
                } else {
                    let pow_pos = b.powi((-n) as usize, prec, rm);
                    let one = BigFloat::from_i32(1, prec);
                    return Ok(one.div(&pow_pos, prec, rm));
                }
            }

            // General case: b^e via the library's pow.
            Ok(b.pow(e, prec, rm, cc))
        }

        // ── Neg ────────────────────────────────────────────────────
        ExprNode::Neg(inner) => {
            let val = get_cached(cache, *inner)?;
            Ok(val.neg())
        }

        // ── Trig ───────────────────────────────────────────────────
        ExprNode::Sin(inner) => {
            let val = get_cached(cache, *inner)?;
            Ok(val.sin(prec, rm, cc))
        }

        ExprNode::Cos(inner) => {
            let val = get_cached(cache, *inner)?;
            Ok(val.cos(prec, rm, cc))
        }

        ExprNode::Tan(inner) => {
            let val = get_cached(cache, *inner)?;
            Ok(val.tan(prec, rm, cc))
        }

        // ── Exp / Ln ───────────────────────────────────────────────
        ExprNode::Exp(inner) => {
            let val = get_cached(cache, *inner)?;
            Ok(val.exp(prec, rm, cc))
        }

        ExprNode::Ln(inner) => {
            let val = get_cached(cache, *inner)?;
            Ok(val.ln(prec, rm, cc))
        }

        // ── Sqrt ───────────────────────────────────────────────────
        ExprNode::Sqrt(inner) => {
            let val = get_cached(cache, *inner)?;
            Ok(val.sqrt(prec, rm))
        }

        // ── Abs ────────────────────────────────────────────────────
        ExprNode::Abs(inner) => {
            let val = get_cached(cache, *inner)?;
            Ok(val.abs())
        }

        // ── Inverse trig ──────────────────────────────────────────
        ExprNode::Asin(inner) => {
            let val = get_cached(cache, *inner)?;
            Ok(val.asin(prec, rm, cc))
        }

        ExprNode::Acos(inner) => {
            let val = get_cached(cache, *inner)?;
            Ok(val.acos(prec, rm, cc))
        }

        ExprNode::Atan(inner) => {
            let val = get_cached(cache, *inner)?;
            Ok(val.atan(prec, rm, cc))
        }

        // ── Hyperbolic ────────────────────────────────────────────
        ExprNode::Sinh(inner) => {
            let val = get_cached(cache, *inner)?;
            Ok(val.sinh(prec, rm, cc))
        }

        ExprNode::Cosh(inner) => {
            let val = get_cached(cache, *inner)?;
            Ok(val.cosh(prec, rm, cc))
        }

        ExprNode::Tanh(inner) => {
            let val = get_cached(cache, *inner)?;
            Ok(val.tanh(prec, rm, cc))
        }

        // ── Unevaluable ────────────────────────────────────────────
        ExprNode::Apply(sid, _) => {
            let name = arena.symbol_name(*sid);
            Err(SymplexError::Unevaluable {
                reason: format!("cannot evaluate user function '{name}'"),
            })
        }

        ExprNode::Derivative(_, _) => Err(SymplexError::Unevaluable {
            reason: "cannot evaluate unevaluated derivative".into(),
        }),

        ExprNode::Integral(_, _) => Err(SymplexError::Unevaluable {
            reason: "cannot evaluate unevaluated integral".into(),
        }),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Look up a cached value, returning an error if not found.
fn get_cached(cache: &FxHashMap<ExprId, BigFloat>, id: ExprId) -> Result<&BigFloat, SymplexError> {
    cache.get(&id).ok_or_else(|| SymplexError::Unevaluable {
        reason: format!("sub-expression {id:?} not in cache (likely contains free symbols)"),
    })
}

/// Convert a `Ratio<BigInt>` to a `BigFloat` with the given precision in bits.
fn ratio_to_bigfloat(r: &Ratio<BigInt>, prec: usize, rm: RoundingMode) -> BigFloat {
    if r.is_zero() {
        return BigFloat::from_i32(0, prec);
    }

    let numer = r.numer();
    let denom = r.denom();

    let n_bf = bigint_to_bigfloat(numer, prec);

    if denom == &BigInt::from(1) {
        return n_bf;
    }

    let d_bf = bigint_to_bigfloat(denom, prec);
    n_bf.div(&d_bf, prec, rm)
}

/// Convert a `BigInt` to a `BigFloat`.
///
/// Uses i128 for values that fit, f64 for larger values (with some
/// precision loss for very large integers — acceptable for MVP).
fn bigint_to_bigfloat(n: &BigInt, prec: usize) -> BigFloat {
    // Try i128 first (covers most practical integers).
    if let Ok(v) = <BigInt as TryInto<i128>>::try_into(n.clone()) {
        return BigFloat::from_i128(v, prec);
    }
    // Fallback: parse through f64 (loses precision for huge integers).
    let s = n.to_string();
    if let Ok(f) = s.parse::<f64>()
        && f.is_finite()
    {
        return BigFloat::from_f64(f, prec);
    }
    // Last resort.
    BigFloat::from_i32(0, prec)
}

/// Try to extract a small integer exponent (fits in i32) from an ExprId.
fn try_as_small_int(arena: &Arena, id: ExprId) -> Option<i32> {
    let r = arena.as_num(id)?;
    if !r.is_integer() {
        return None;
    }
    let n = r.to_integer();
    let val: i64 = n.try_into().ok()?;
    if val >= i32::MIN as i64 && val <= i32::MAX as i64 {
        Some(val as i32)
    } else {
        None
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Decimal formatting
// ═══════════════════════════════════════════════════════════════════════════

/// Format a `BigFloat` as a decimal string with `digits` significant digits.
///
/// Uses astro-float's `convert_to_radix` for reliable base-10 conversion,
/// then formats the result with proper decimal point placement.
fn format_decimal(
    val: &BigFloat,
    digits: u32,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<String, SymplexError> {
    if val.is_zero() {
        return Ok("0".to_string());
    }

    if val.is_inf_pos() {
        return Ok("oo".to_string());
    }
    if val.is_inf_neg() {
        return Ok("-oo".to_string());
    }

    // Use the library's decimal conversion.
    let (sign, mantissa, exponent) = val
        .convert_to_radix(Radix::Dec, rm, cc)
        .map_err(|e| SymplexError::NotImplemented(format!("decimal conversion failed: {e:?}")))?;

    let sign_str = if sign == Sign::Neg { "-" } else { "" };

    // mantissa is Vec<u8> of decimal digits [d1, d2, d3, ...]
    // representing the value 0.d1d2d3... × 10^exponent.
    // So the actual value is d1.d2d3... × 10^(exponent-1).

    // Truncate to requested digit count.
    let n = (digits as usize).min(mantissa.len());
    if n == 0 {
        return Ok("0".to_string());
    }

    let digit_chars: Vec<char> = mantissa[..n].iter().map(|&d| (b'0' + d) as char).collect();

    // The "adjusted exponent" is exponent-1 (shifting from 0.ddd to d.ddd notation).
    let adj_exp = exponent as i64 - 1;

    // Decide between plain decimal and scientific notation.
    if adj_exp >= 0 && (adj_exp as usize) < n {
        // The decimal point falls within the digit string.
        // E.g., digits=[3,1,4,1,5], adj_exp=0 → "3.1415"
        let dot_pos = adj_exp as usize + 1;
        let mut s = String::with_capacity(n + 2);
        s.push_str(sign_str);
        for (i, &ch) in digit_chars.iter().enumerate() {
            if i == dot_pos && dot_pos < n {
                s.push('.');
            }
            s.push(ch);
        }
        // Trim trailing zeros after the decimal point.
        if s.contains('.') {
            let trimmed = s.trim_end_matches('0').trim_end_matches('.');
            Ok(trimmed.to_string())
        } else {
            Ok(s)
        }
    } else if adj_exp >= 0 && (adj_exp as usize) < n + 6 {
        // Small integer — pad with trailing zeros.
        let mut s = String::with_capacity(adj_exp as usize + 2);
        s.push_str(sign_str);
        for &ch in &digit_chars {
            s.push(ch);
        }
        let needed_zeros = (adj_exp as usize + 1).saturating_sub(digit_chars.len());
        for _ in 0..needed_zeros {
            s.push('0');
        }
        Ok(s)
    } else if (-4..0).contains(&adj_exp) {
        // Small fraction like 0.00123.
        let mut s = String::with_capacity(n + (-adj_exp) as usize + 3);
        s.push_str(sign_str);
        s.push_str("0.");
        let leading_zeros = (-adj_exp - 1) as usize;
        for _ in 0..leading_zeros {
            s.push('0');
        }
        for &ch in &digit_chars {
            s.push(ch);
        }
        // Trim trailing zeros.
        let trimmed = s.trim_end_matches('0');
        Ok(trimmed.to_string())
    } else {
        // Scientific notation: d1.d2d3...dn × 10^adj_exp.
        let mut s = String::with_capacity(n + 10);
        s.push_str(sign_str);
        s.push(digit_chars[0]);
        if n > 1 {
            s.push('.');
            for &ch in &digit_chars[1..] {
                s.push(ch);
            }
            // Trim trailing zeros in the fractional part.
            while s.ends_with('0') {
                s.pop();
            }
            if s.ends_with('.') {
                s.pop();
            }
        }
        s.push_str(&format!("e{adj_exp}"));
        Ok(s)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::Arena;

    /// Helper: evaluate and assert the result starts with expected prefix.
    fn assert_evalf_starts_with(arena: &Arena, expr: ExprId, digits: u32, prefix: &str) {
        let result = evalf(arena, expr, digits).unwrap();
        assert!(
            result.starts_with(prefix),
            "expected to start with '{prefix}', got: '{result}'"
        );
    }

    /// Helper: evaluate and assert exact string match.
    fn assert_evalf_eq(arena: &Arena, expr: ExprId, digits: u32, expected: &str) {
        let result = evalf(arena, expr, digits).unwrap();
        assert_eq!(result, expected, "evalf mismatch");
    }

    // ── Integers ────────────────────────────────────────────────────

    #[test]
    fn integer_zero() {
        let a = Arena::new();
        assert_evalf_eq(&a, a.zero, 10, "0");
    }

    #[test]
    fn integer_one() {
        let a = Arena::new();
        assert_evalf_eq(&a, a.one, 10, "1");
    }

    #[test]
    fn integer_negative_one() {
        let a = Arena::new();
        assert_evalf_eq(&a, a.neg_one, 10, "-1");
    }

    #[test]
    fn integer_five() {
        let mut a = Arena::new();
        let five = a.int(5);
        assert_evalf_eq(&a, five, 10, "5");
    }

    #[test]
    fn integer_large() {
        let mut a = Arena::new();
        let n = a.int(123456);
        assert_evalf_eq(&a, n, 10, "123456");
    }

    // ── Rationals ───────────────────────────────────────────────────

    #[test]
    fn rational_half() {
        let mut a = Arena::new();
        let half = a.rational(1, 2);
        assert_evalf_eq(&a, half, 10, "0.5");
    }

    #[test]
    fn rational_one_third() {
        let mut a = Arena::new();
        let third = a.rational(1, 3);
        assert_evalf_starts_with(&a, third, 10, "0.333333333");
    }

    #[test]
    fn rational_negative() {
        let mut a = Arena::new();
        let r = a.rational(-3, 4);
        assert_evalf_starts_with(&a, r, 10, "-0.75");
    }

    // ── Constants ───────────────────────────────────────────────────

    #[test]
    fn constant_pi_15_digits() {
        let a = Arena::new();
        assert_evalf_starts_with(&a, a.pi, 15, "3.14159265358979");
    }

    #[test]
    fn constant_pi_50_digits() {
        let a = Arena::new();
        let result = evalf(&a, a.pi, 50).unwrap();
        assert!(
            result.starts_with("3.1415926535897932384626433832795"),
            "pi to 50 digits: {result}"
        );
    }

    #[test]
    fn constant_e_15_digits() {
        let a = Arena::new();
        assert_evalf_starts_with(&a, a.e_const, 15, "2.71828182845904");
    }

    // ── Arithmetic ──────────────────────────────────────────────────

    #[test]
    fn add_integers() {
        let mut a = Arena::new();
        let two = a.int(2);
        let three = a.int(3);
        let expr = a.add(&[two, three]);
        assert_evalf_eq(&a, expr, 10, "5");
    }

    #[test]
    fn mul_integers() {
        let mut a = Arena::new();
        let two = a.int(2);
        let three = a.int(3);
        let expr = a.mul(&[two, three]);
        assert_evalf_eq(&a, expr, 10, "6");
    }

    #[test]
    fn pow_integer() {
        let mut a = Arena::new();
        let two = a.int(2);
        let ten = a.int(10);
        let expr = a.pow(two, ten);
        assert_evalf_eq(&a, expr, 10, "1024");
    }

    #[test]
    fn pow_negative_exponent() {
        let mut a = Arena::new();
        let two = a.int(2);
        let neg_one = a.int(-1);
        let expr = a.pow(two, neg_one);
        assert_evalf_eq(&a, expr, 10, "0.5");
    }

    #[test]
    fn pi_plus_one() {
        let mut a = Arena::new();
        let pi = a.pi;
        let one = a.one;
        let expr = a.add(&[pi, one]);
        assert_evalf_starts_with(&a, expr, 15, "4.14159265358979");
    }

    // ── Transcendental functions ─────────────────────────────────────

    #[test]
    fn sin_zero() {
        let mut a = Arena::new();
        let zero = a.zero;
        let expr = a.sin(zero);
        assert_evalf_eq(&a, expr, 10, "0");
    }

    #[test]
    fn cos_zero() {
        let mut a = Arena::new();
        let zero = a.zero;
        let expr = a.cos(zero);
        assert_evalf_eq(&a, expr, 10, "1");
    }

    #[test]
    fn exp_zero() {
        let mut a = Arena::new();
        let zero = a.zero;
        let expr = a.exp(zero);
        assert_evalf_eq(&a, expr, 10, "1");
    }

    #[test]
    fn exp_one() {
        let mut a = Arena::new();
        let one = a.one;
        let expr = a.exp(one);
        assert_evalf_starts_with(&a, expr, 15, "2.71828182845904");
    }

    #[test]
    fn ln_one() {
        let mut a = Arena::new();
        let one = a.one;
        let expr = a.ln(one);
        assert_evalf_eq(&a, expr, 10, "0");
    }

    #[test]
    fn ln_e() {
        let mut a = Arena::new();
        let e = a.e_const;
        let expr = a.ln(e);
        // ln(e) should be 1, or very close.
        let result = evalf(&a, expr, 15).unwrap();
        let val: f64 = result.parse().unwrap_or(999.0);
        assert!(
            (val - 1.0).abs() < 1e-10,
            "ln(e) should be ~1, got: {result}"
        );
    }

    #[test]
    fn sqrt_four() {
        let mut a = Arena::new();
        let four = a.int(4);
        let expr = a.sqrt(four);
        assert_evalf_eq(&a, expr, 10, "2");
    }

    #[test]
    fn sqrt_two_30_digits() {
        let mut a = Arena::new();
        let two = a.int(2);
        let expr = a.sqrt(two);
        assert_evalf_starts_with(&a, expr, 30, "1.4142135623730950488");
    }

    #[test]
    fn sin_pi_is_near_zero() {
        let mut a = Arena::new();
        let pi = a.pi;
        let expr = a.sin(pi);
        let result = evalf(&a, expr, 20).unwrap();
        // sin(pi) is mathematically 0, numerically very small.
        let val: f64 = result.parse().unwrap_or(999.0);
        assert!(val.abs() < 1e-15, "sin(pi) should be ~0, got: {result}");
    }

    // ── Neg and Abs ─────────────────────────────────────────────────

    #[test]
    fn neg_pi() {
        let mut a = Arena::new();
        let pi = a.pi;
        let expr = a.neg(pi);
        assert_evalf_starts_with(&a, expr, 10, "-3.14159265");
    }

    #[test]
    fn abs_neg_pi() {
        let mut a = Arena::new();
        let pi = a.pi;
        let neg_pi = a.neg(pi);
        let expr = a.abs(neg_pi);
        assert_evalf_starts_with(&a, expr, 10, "3.14159265");
    }

    #[test]
    fn abs_negative_integer() {
        let mut a = Arena::new();
        let neg_five = a.int(-5);
        let expr = a.abs(neg_five);
        assert_evalf_eq(&a, expr, 10, "5");
    }

    // ── Error cases ─────────────────────────────────────────────────

    #[test]
    fn free_symbol_errors() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = evalf(&a, x, 10);
        assert!(result.is_err(), "evalf of free symbol should error");
    }

    #[test]
    fn infinity_errors() {
        let a = Arena::new();
        let result = evalf(&a, a.infinity, 10);
        assert!(result.is_err(), "evalf of infinity should error");
    }

    #[test]
    fn nan_errors() {
        let a = Arena::new();
        let result = evalf(&a, a.nan, 10);
        assert!(result.is_err(), "evalf of NaN should error");
    }

    #[test]
    fn imaginary_unit_errors() {
        let a = Arena::new();
        let result = evalf(&a, a.i_unit, 10);
        assert!(
            result.is_err(),
            "evalf of i should error (complex not supported)"
        );
    }

    // ── Composite expressions ───────────────────────────────────────

    #[test]
    fn polynomial_at_value() {
        // Evaluate x^2 + 2*x + 1 at x=3 → 16.
        // We substitute first, then evalf.
        let mut a = Arena::new();
        let x = a.symbol("x");
        let two = a.int(2);
        let x_sq = a.pow(x, two);
        let two_x = a.mul(&[two, x]);
        let expr = a.add(&[x_sq, two_x, a.one]);

        // Substitute x → 3.
        let three = a.int(3);
        let evaluated = crate::subs::subs(&mut a, expr, x, three);
        assert_evalf_eq(&a, evaluated, 10, "16");
    }

    #[test]
    fn sin_cos_identity() {
        // sin^2(1) + cos^2(1) should be very close to 1.
        let mut a = Arena::new();
        let one = a.one;
        let two = a.int(2);
        let sin_1 = a.sin(one);
        let cos_1 = a.cos(one);
        let sin_sq = a.pow(sin_1, two);
        let cos_sq = a.pow(cos_1, two);
        let expr = a.add(&[sin_sq, cos_sq]);
        let result = evalf(&a, expr, 15).unwrap();
        let val: f64 = result.parse().unwrap_or(0.0);
        assert!(
            (val - 1.0).abs() < 1e-12,
            "sin^2(1) + cos^2(1) should be ~1, got: {result}"
        );
    }

    // ── Idempotence ─────────────────────────────────────────────────

    #[test]
    fn same_precision_same_result() {
        let a = Arena::new();
        let r1 = evalf(&a, a.pi, 20).unwrap();
        let r2 = evalf(&a, a.pi, 20).unwrap();
        assert_eq!(r1, r2, "same precision should give same result");
    }
}

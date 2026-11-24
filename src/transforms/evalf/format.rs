//! Decimal / complex formatting (part of [`evalf`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Decimal / complex formatting
// ═══════════════════════════════════════════════════════════════════════════

/// Determine whether `part` is negligibly small compared to `other`.
///
/// Returns `true` if `part` is zero, or if its binary exponent is more
/// than `digits * log2(10)` bits below `other`'s exponent (meaning it
/// is below the requested output precision).
pub(super) fn is_negligible_part(part: &BigFloat, other: &BigFloat, digits: u32) -> bool {
    if part.is_zero() {
        return true;
    }
    if other.is_zero() {
        return false;
    }
    match (part.exponent(), other.exponent()) {
        (Some(p_exp), Some(o_exp)) => {
            let bit_threshold = (digits as i64) * 34 / 10 + 4;
            (o_exp as i64 - p_exp as i64) > bit_threshold
        }
        _ => false,
    }
}

/// Format a complex result as a string.
///
/// A part that is 0 is not printed: a real number, a pure imaginary
/// number, or `a + b*i` / `a - b*i`.  [`settle`] has set to 0 every part
/// whose ball contains 0 (it is below the printed digits of the value)
/// and certified the digits of every other part, however small next to
/// the other one, so a part certified nonzero is always printed (SymPy's
/// `N` agrees: `Chi(−60)` is `9.68091107·10²³ + 3.141592654i` at 10
/// digits).  Up to 0.52 a part negligible next to the other
/// ([`is_negligible_part`]) was dropped whatever its ball: `Chi(−60)`
/// printed `9.68091107e23`, `10³⁰ + i` printed `1e30` and `1 + 10³⁰·i`
/// printed `1e30*i`.
pub(super) fn format_complex(z: &Complex, digits: u32) -> Result<String, SymplexError> {
    let re = &z.0;
    let im = &z.1;

    if re.is_zero() && im.is_zero() {
        return Ok("0".to_string());
    }

    if im.is_zero() {
        return format_decimal(re, digits);
    }

    if re.is_zero() {
        // Pure imaginary.
        let im_str = format_decimal(im, digits)?;
        return if im_str == "1" {
            Ok("i".to_string())
        } else if im_str == "-1" {
            Ok("-i".to_string())
        } else {
            Ok(format!("{im_str}*i"))
        };
    }

    // Both parts present.
    let re_str = format_decimal(re, digits)?;
    if im.is_negative() {
        let im_abs_str = format_decimal(&im.abs(), digits)?;
        if im_abs_str == "1" {
            Ok(format!("{re_str} - i"))
        } else {
            Ok(format!("{re_str} - {im_abs_str}*i"))
        }
    } else {
        let im_str = format_decimal(im, digits)?;
        if im_str == "1" {
            Ok(format!("{re_str} + i"))
        } else {
            Ok(format!("{re_str} + {im_str}*i"))
        }
    }
}

/// Round a decimal digit string `0.d1d2d3…` to `digits` significant digits.
///
/// Rounds to nearest, ties to even (the digits beyond the requested
/// count are exact decimal digits of the binary value, so an exact tie is
/// a genuine `…5000…`).  Carries propagate leftwards; a carry out of the
/// leading digit (`9.9996 → 10.00`) is reported as an exponent increment
/// so the caller can re-place the decimal point.
///
/// Returns `(rounded_digits, exponent_increment)`; when `mantissa` has no
/// more than `digits` digits it is returned unchanged.
pub(super) fn round_decimal_digits(mantissa: &[u8], digits: usize) -> (Vec<u8>, i64) {
    if digits == 0 || mantissa.len() <= digits {
        return (mantissa.to_vec(), 0);
    }
    let mut kept: Vec<u8> = mantissa[..digits].to_vec();
    let next = mantissa[digits];
    let rest_nonzero = mantissa[digits + 1..].iter().any(|&d| d != 0);
    let round_up = match next.cmp(&5) {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Less => false,
        // Exactly half: ties to even.
        std::cmp::Ordering::Equal => rest_nonzero || kept[digits - 1] % 2 == 1,
    };
    if !round_up {
        return (kept, 0);
    }
    // Propagate the carry from the last kept digit leftwards.
    for d in kept.iter_mut().rev() {
        if *d == 9 {
            *d = 0;
        } else {
            *d += 1;
            return (kept, 0);
        }
    }
    // Carried out of the leading digit: 0.999… → 1.000… × 10.
    kept[0] = 1;
    (kept, 1)
}

/// Format a `BigFloat` as a decimal string with `digits` significant digits.
///
/// Converts with [`crate::base::bigfloat::to_decimal_digits`] (exact,
/// from the binary mantissa), rounds to `digits` significant digits, then
/// formats the result with proper decimal point placement.
pub(super) fn format_decimal(val: &BigFloat, digits: u32) -> Result<String, SymplexError> {
    if val.is_zero() {
        return Ok("0".to_string());
    }

    if val.is_inf_pos() {
        return Ok("oo".to_string());
    }
    if val.is_inf_neg() {
        return Ok("-oo".to_string());
    }

    let crate::base::bigfloat::DecimalDigits {
        sign,
        digits: mantissa,
        exponent,
    } = crate::base::bigfloat::to_decimal_digits(val)
        .map_err(|e| SymplexError::NotImplemented(format!("decimal conversion failed: {e:?}")))?;

    let sign_str = if sign == Sign::Neg { "-" } else { "" };

    // mantissa is Vec<u8> of decimal digits [d1, d2, d3, ...]
    // representing the value 0.d1d2d3... × 10^exponent.
    // So the actual value is d1.d2d3... × 10^(exponent-1).

    // Round (not truncate) to the requested digit count.
    let (rounded, exp_carry) = round_decimal_digits(&mantissa, digits as usize);
    let n = rounded.len();
    if n == 0 {
        return Ok("0".to_string());
    }

    let digit_chars: Vec<char> = rounded.iter().map(|&d| (b'0' + d) as char).collect();

    // The "adjusted exponent" is exponent-1 (shifting from 0.ddd to d.ddd notation).
    let adj_exp = exponent as i64 - 1 + exp_carry;

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
        // Small integer — pad with trailing zeros (`adj_exp < n + 6`).
        let mut s = String::with_capacity(n + 8);
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
        // Small fraction like 0.00123 (`−4 ≤ adj_exp < 0`).
        let mut s = String::with_capacity(n + 7);
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

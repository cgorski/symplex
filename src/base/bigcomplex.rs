//! Arbitrary-precision complex numbers as `(re, im)` pairs of
//! [`BigFloat`]s, with the field operations every numeric algorithm needs
//! (`evalf`, the polynomial root finders, algebraic-number verification).
//!
//! Only the operations that need no transcendental constants live here;
//! `exp`, `ln`, the trigonometric family and `sqrt` stay in
//! `transforms::evalf`, next to the constant cache they draw on.

use astro_float::{BigFloat, RoundingMode};

/// A complex number at some working precision: `(real part, imaginary part)`.
pub(crate) type Complex = (BigFloat, BigFloat);

/// `0 + 0i` at `prec` bits.
pub(crate) fn c_zero(prec: usize) -> Complex {
    (BigFloat::new(prec), BigFloat::new(prec))
}

/// `1 + 0i` at `prec` bits.
pub(crate) fn c_one(prec: usize) -> Complex {
    (BigFloat::from_i32(1, prec), BigFloat::new(prec))
}

/// `0 + 1i` at `prec` bits.
pub(crate) fn c_i(prec: usize) -> Complex {
    (BigFloat::new(prec), BigFloat::from_i32(1, prec))
}

/// `r + 0i`.
pub(crate) fn c_from_real(r: BigFloat, prec: usize) -> Complex {
    (r, BigFloat::new(prec))
}

pub(crate) fn c_add(a: &Complex, b: &Complex, prec: usize, rm: RoundingMode) -> Complex {
    (a.0.add(&b.0, prec, rm), a.1.add(&b.1, prec, rm))
}

pub(crate) fn c_sub(a: &Complex, b: &Complex, prec: usize, rm: RoundingMode) -> Complex {
    (a.0.sub(&b.0, prec, rm), a.1.sub(&b.1, prec, rm))
}

pub(crate) fn c_mul(a: &Complex, b: &Complex, prec: usize, rm: RoundingMode) -> Complex {
    // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    let ac = a.0.mul(&b.0, prec, rm);
    let bd = a.1.mul(&b.1, prec, rm);
    let ad = a.0.mul(&b.1, prec, rm);
    let bc = a.1.mul(&b.0, prec, rm);
    (ac.sub(&bd, prec, rm), ad.add(&bc, prec, rm))
}

pub(crate) fn c_div(a: &Complex, b: &Complex, prec: usize, rm: RoundingMode) -> Complex {
    // (a+bi)/(c+di) = ((ac+bd) + (bc-ad)i) / (c²+d²)
    let ac = a.0.mul(&b.0, prec, rm);
    let bd = a.1.mul(&b.1, prec, rm);
    let bc = a.1.mul(&b.0, prec, rm);
    let ad = a.0.mul(&b.1, prec, rm);
    let denom =
        b.0.mul(&b.0, prec, rm)
            .add(&b.1.mul(&b.1, prec, rm), prec, rm);
    let re = ac.add(&bd, prec, rm).div(&denom, prec, rm);
    let im = bc.sub(&ad, prec, rm).div(&denom, prec, rm);
    (re, im)
}

pub(crate) fn c_neg(a: &Complex) -> Complex {
    (a.0.neg(), a.1.neg())
}

/// `|z| = √(re² + im²)`.
pub(crate) fn c_abs(a: &Complex, prec: usize, rm: RoundingMode) -> BigFloat {
    let re2 = a.0.mul(&a.0, prec, rm);
    let im2 = a.1.mul(&a.1, prec, rm);
    let sum = re2.add(&im2, prec, rm);
    sum.sqrt(prec, rm)
}

/// `zⁿ` by binary powering.
pub(crate) fn c_powi(base: &Complex, n: usize, prec: usize, rm: RoundingMode) -> Complex {
    if n == 0 {
        return c_one(prec);
    }
    let mut result = c_one(prec);
    let mut b = base.clone();
    let mut exp = n;
    while exp > 0 {
        if exp & 1 == 1 {
            result = c_mul(&result, &b, prec, rm);
        }
        b = c_mul(&b, &b, prec, rm);
        exp >>= 1;
    }
    result
}

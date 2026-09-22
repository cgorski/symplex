//! Polynomial interpolation over a field.
//!
//! One Newton divided-difference kernel, [`interpolate`], and thin
//! adapters for the point shapes the crate uses (integer abscissae with
//! field- or integer-valued ordinates).  The interpolant of degree
//! `< n` through `n` points with distinct abscissae is unique, so every
//! adapter returns exactly the polynomial the Lagrange-basis construction
//! it replaced did — in `O(n²)` field operations instead of the `O(n³)`
//! of rebuilding each Lagrange basis polynomial.

use num_bigint::BigInt;
use num_rational::Ratio;

use super::dense::Poly;
use super::generic::GenPoly;
use super::traits::{Field, IntegralCoeff, Ring};

/// The unique polynomial of degree `< xs.len()` with `p(xsᵢ) = ysᵢ`, by
/// Newton's divided differences.
///
/// Returns `None` if two abscissae coincide or the slices differ in
/// length.  An empty input yields the zero polynomial.
pub(crate) fn interpolate<C: Field>(xs: &[C], ys: &[C]) -> Option<GenPoly<C>> {
    if xs.len() != ys.len() {
        return None;
    }
    let n = xs.len();
    if n == 0 {
        return Some(GenPoly::zero());
    }

    // Divided differences in place: after pass j, coef[i] = f[x_{i−j}, …, x_i].
    let mut coef: Vec<C> = ys.to_vec();
    for j in 1..n {
        for i in (j..n).rev() {
            let denom = Ring::sub(&xs[i], &xs[i - j]);
            if denom.is_zero() {
                return None;
            }
            coef[i] = Field::div(&Ring::sub(&coef[i], &coef[i - 1]), &denom);
        }
    }

    // Horner form of the Newton basis:
    //   p = c_{n−1};  p ← p·(x − x_i) + c_i  for i = n−2, …, 0.
    let mut p: Vec<C> = vec![coef[n - 1].clone()];
    for i in (0..n - 1).rev() {
        // p ← p·x − x_i·p + c_i
        let mut next = vec![C::zero(); p.len() + 1];
        for (k, pk) in p.iter().enumerate() {
            next[k + 1] = Ring::add(&next[k + 1], pk);
            next[k] = Ring::sub(&next[k], &Ring::mul(&xs[i], pk));
        }
        next[0] = Ring::add(&next[0], &coef[i]);
        p = next;
    }
    Some(GenPoly::from_coeffs(p))
}

/// Interpolation through `(xᵢ, yᵢ)` with integer abscissae embedded into
/// `C`.  Abscissae are distinct in every use (`0, 1, …, n`), so a repeated
/// one — impossible in characteristic zero — yields the zero polynomial.
pub(crate) fn interpolate_at_integers<C: Field + IntegralCoeff>(points: &[(i64, C)]) -> GenPoly<C> {
    let xs: Vec<C> = points.iter().map(|(x, _)| C::from_i64(*x)).collect();
    let ys: Vec<C> = points.iter().map(|(_, y)| y.clone()).collect();
    interpolate(&xs, &ys).unwrap_or_else(GenPoly::zero)
}

/// Interpolation through integer points, as a polynomial over ℚ (the
/// coefficients need not be integers — the caller checks).  `None` if two
/// abscissae coincide.
pub(crate) fn interpolate_integer_points(points: &[(i64, BigInt)]) -> Option<Poly> {
    let xs: Vec<Ratio<BigInt>> = points
        .iter()
        .map(|(x, _)| Ratio::from_integer(BigInt::from(*x)))
        .collect();
    let ys: Vec<Ratio<BigInt>> = points
        .iter()
        .map(|(_, y)| Ratio::from_integer(y.clone()))
        .collect();
    interpolate(&xs, &ys)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn r(n: i64, d: i64) -> Ratio<BigInt> {
        Ratio::new(BigInt::from(n), BigInt::from(d))
    }

    fn ri(n: i64) -> Ratio<BigInt> {
        Ratio::from_integer(BigInt::from(n))
    }

    /// The Lagrange-basis construction this module replaced, kept as the
    /// oracle.
    fn lagrange(points: &[(Ratio<BigInt>, Ratio<BigInt>)]) -> Option<Poly> {
        let n = points.len();
        let mut result = Poly::zero();
        for i in 0..n {
            let (xi, yi) = &points[i];
            let mut basis = Poly::from_int(1);
            let mut denom = ri(1);
            for (j, (xj, _)) in points.iter().enumerate() {
                if i == j {
                    continue;
                }
                let diff = xi - xj;
                if diff.is_zero() {
                    return None;
                }
                basis = &basis * &Poly::from_coeffs(vec![-xj.clone(), ri(1)]);
                denom *= diff;
            }
            if yi.is_zero() {
                continue;
            }
            result = &result + &basis.scale(&(yi / denom));
        }
        Some(result)
    }

    #[test]
    fn empty_and_single_point() {
        assert_eq!(interpolate::<Ratio<BigInt>>(&[], &[]), Some(Poly::zero()));
        assert_eq!(
            interpolate(&[ri(3)], &[r(7, 2)]),
            Some(Poly::constant(r(7, 2)))
        );
        assert_eq!(interpolate(&[ri(3)], &[ri(0)]), Some(Poly::zero()));
    }

    #[test]
    fn reproduces_known_polynomial() {
        // p = 2x³ − x/2 + 3
        let p = Poly::from_coeffs(vec![ri(3), r(-1, 2), ri(0), ri(2)]);
        let xs: Vec<_> = [-2, -1, 0, 1].iter().map(|&k| ri(k)).collect();
        let ys: Vec<_> = xs.iter().map(|x| p.eval(x)).collect();
        assert_eq!(interpolate(&xs, &ys), Some(p.clone()));
        // Extra points (degree < n − 1) still give p.
        let xs: Vec<_> = [-3, -2, -1, 0, 1, 2, 3].iter().map(|&k| ri(k)).collect();
        let ys: Vec<_> = xs.iter().map(|x| p.eval(x)).collect();
        assert_eq!(interpolate(&xs, &ys), Some(p));
    }

    #[test]
    fn repeated_abscissa_is_none() {
        assert!(interpolate(&[ri(1), ri(2), ri(1)], &[ri(0), ri(1), ri(2)]).is_none());
        assert!(interpolate(&[ri(1), ri(2)], &[ri(0)]).is_none());
    }

    #[test]
    fn matches_lagrange_on_rational_points() {
        let mut s = 0x9E37_79B9u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for n in 1..=9usize {
            let mut xs: Vec<Ratio<BigInt>> = Vec::new();
            while xs.len() < n {
                let x = r((next() % 41) as i64 - 20, (next() % 4) as i64 + 1);
                if !xs.contains(&x) {
                    xs.push(x);
                }
            }
            let ys: Vec<Ratio<BigInt>> = (0..n)
                .map(|_| r((next() % 61) as i64 - 30, (next() % 6) as i64 + 1))
                .collect();
            let pts: Vec<_> = xs.iter().cloned().zip(ys.iter().cloned()).collect();
            assert_eq!(interpolate(&xs, &ys), lagrange(&pts), "n = {n}");
        }
    }

    #[test]
    fn integer_adapters() {
        let pts = [(0i64, ri(1)), (1, ri(2)), (2, ri(5))];
        assert_eq!(
            interpolate_at_integers(&pts),
            Poly::from_coeffs(vec![ri(1), ri(0), ri(1)])
        );
        let zpts = [
            (0i64, BigInt::from(1)),
            (1, BigInt::from(2)),
            (2, BigInt::from(5)),
        ];
        assert_eq!(
            interpolate_integer_points(&zpts),
            Some(Poly::from_coeffs(vec![ri(1), ri(0), ri(1)]))
        );
        // Half-integer slope: coefficients need not be integers.
        let zpts = [(0i64, BigInt::from(0)), (2, BigInt::from(1))];
        assert_eq!(
            interpolate_integer_points(&zpts),
            Some(Poly::from_coeffs(vec![ri(0), r(1, 2)]))
        );
        assert!(
            interpolate_integer_points(&[(1, BigInt::from(0)), (1, BigInt::from(1))]).is_none()
        );
    }
}

//! Exact values of complex-rational sub-expressions.
//!
//! A sub-expression built from rationals and `i` by `+`, `·`, `−`, integer
//! powers, `conj`, `re` and `im` has an exact value `p + q·i` with rational
//! `p` and `q`, and the canonical form does not always fold it:
//! `8/(1 + i) + 4i − 5` stays a sum, although it is exactly `−1`.  Evaluated
//! in floating point, `8/(1 + i)` is rounded, the imaginary parts `−4` and
//! `4` cancel to a rounding residue within its error of 0, and the side of
//! a branch cut is undecidable at every precision (see `accuracy.rs`): before
//! 0.31 `√(8/(1 + i) + 4i − 5)` (`= i`) and `ln(8/(1 + i) + 4i − 5)`
//! (`= iπ`) were refused at every number of digits.  Such a node's
//! floating-point value is refined by its exact value ([`refine`]): an exact
//! 0 part where the exact value has one, so the side of a cut is decided by
//! convention (the principal value) as for a literal, and the exact value
//! rounded once where the floating-point part's sign is already certain.
//!
//! The arithmetic is bounded: an integer power beyond [`MAX_EXPONENT`] or a
//! result whose numerator and denominator together exceed [`MAX_BITS`] is
//! left to the floating-point evaluation.

use astro_float::{BigFloat, RoundingMode};
use num_traits::{One, Zero};
use rustc_hash::FxHashMap;

use super::accuracy::{self, Bound};
use crate::base::arena::Arena;
use crate::base::bigcomplex::Complex;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::Q;

/// Largest `|n|` of an integer power computed exactly.
const MAX_EXPONENT: u32 = 256;

/// Largest number of bits of numerator plus denominator (per part) of an
/// exact value.
const MAX_BITS: u64 = 1 << 15;

/// An exact complex rational `re + im·i`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Gauss {
    /// The real part.
    pub(super) re: Q,
    /// The imaginary part.
    pub(super) im: Q,
}

/// The exact values of the composite nodes evaluated so far.
pub(super) type ExactMap = FxHashMap<ExprId, Gauss>;

impl Gauss {
    pub(super) fn real(re: Q) -> Gauss {
        Gauss { re, im: Q::zero() }
    }

    pub(super) fn is_zero(&self) -> bool {
        self.re.is_zero() && self.im.is_zero()
    }

    pub(super) fn add(&self, o: &Gauss) -> Gauss {
        Gauss {
            re: &self.re + &o.re,
            im: &self.im + &o.im,
        }
    }

    pub(super) fn mul(&self, o: &Gauss) -> Gauss {
        if self.im.is_zero() && o.im.is_zero() {
            return Gauss::real(&self.re * &o.re);
        }
        Gauss {
            re: &self.re * &o.re - &self.im * &o.im,
            im: &self.re * &o.im + &self.im * &o.re,
        }
    }

    pub(super) fn neg(&self) -> Gauss {
        Gauss {
            re: -self.re.clone(),
            im: -self.im.clone(),
        }
    }

    /// `1/z` for `z ≠ 0`.
    fn recip(&self) -> Option<Gauss> {
        if self.is_zero() {
            return None;
        }
        if self.im.is_zero() {
            return Some(Gauss::real(self.re.recip()));
        }
        let n = &self.re * &self.re + &self.im * &self.im;
        Some(Gauss {
            re: &self.re / &n,
            im: -(&self.im / &n),
        })
    }

    /// Within [`MAX_BITS`]?
    pub(super) fn small(&self) -> bool {
        let bits = |q: &Q| q.numer().bits() + q.denom().bits();
        bits(&self.re) <= MAX_BITS && bits(&self.im) <= MAX_BITS
    }

    /// `zⁿ` by repeated squaring (`n ≥ 0`), `None` past [`MAX_BITS`].
    fn powu(&self, mut n: u32) -> Option<Gauss> {
        let mut result = Gauss::real(Q::one());
        let mut base = self.clone();
        while n > 0 {
            if n & 1 == 1 {
                result = result.mul(&base);
                if !result.small() {
                    return None;
                }
            }
            n >>= 1;
            if n > 0 {
                base = base.mul(&base);
                if !base.small() {
                    return None;
                }
            }
        }
        Some(result)
    }
}

/// The exact value of a leaf (`Num`, `i`) or of an already recorded node.
pub(super) fn operand(arena: &Arena, id: ExprId, exact: &ExactMap) -> Option<Gauss> {
    match arena.node(id) {
        ExprNode::Num(nid) => Some(Gauss::real(arena.num(*nid).clone())),
        ExprNode::ImaginaryUnit => Some(Gauss {
            re: Q::zero(),
            im: Q::one(),
        }),
        _ => exact.get(&id).cloned(),
    }
}

/// Does `id` have an exact value (see [`operand`])?  A cheap test run before
/// any arithmetic.
fn has_operand(arena: &Arena, id: ExprId, exact: &ExactMap) -> bool {
    matches!(arena.node(id), ExprNode::Num(_) | ExprNode::ImaginaryUnit) || exact.contains_key(&id)
}

/// The exact value of the composite node `id` when all its operands have
/// one (see the module documentation); `None` for a leaf, any other node,
/// or a result beyond the size limits.
pub(super) fn exact_value(arena: &Arena, id: ExprId, exact: &ExactMap) -> Option<Gauss> {
    let node = arena.node(id);
    match node {
        ExprNode::Neg(a) => Some(operand(arena, *a, exact)?.neg()),
        ExprNode::Conjugate(a) => {
            let z = operand(arena, *a, exact)?;
            Some(Gauss {
                re: z.re,
                im: -z.im,
            })
        }
        ExprNode::Re(a) => Some(Gauss::real(operand(arena, *a, exact)?.re)),
        ExprNode::Im(a) => Some(Gauss::real(operand(arena, *a, exact)?.im)),
        ExprNode::Add(cs) | ExprNode::Mul(cs) => {
            if !cs.iter().all(|&c| has_operand(arena, c, exact)) {
                return None;
            }
            let is_add = matches!(node, ExprNode::Add(_));
            let mut acc = if is_add {
                Gauss::real(Q::zero())
            } else {
                Gauss::real(Q::one())
            };
            for &c in cs.iter() {
                let z = operand(arena, c, exact)?;
                acc = if is_add { acc.add(&z) } else { acc.mul(&z) };
                if !acc.small() {
                    return None;
                }
            }
            Some(acc)
        }
        ExprNode::Pow(b, e) => {
            let n = arena.as_num(*e)?;
            if !n.is_integer() || !has_operand(arena, *b, exact) {
                return None;
            }
            let n: i64 = n.to_integer().try_into().ok()?;
            let m = u32::try_from(n.unsigned_abs())
                .ok()
                .filter(|&m| m <= MAX_EXPONENT)?;
            let z = operand(arena, *b, exact)?;
            let z = if n < 0 { z.recip()? } else { z };
            z.powu(m)
        }
        _ => None,
    }
}

/// The value of `z` at `prec` bits and its bound: each part rounded once,
/// exact when it is 0 or representable (see [`accuracy::rational_part`]).
pub(super) fn to_value(z: &Gauss, prec: usize, rm: RoundingMode) -> (Complex, Bound) {
    let part = |q: &Q| -> BigFloat {
        if q.is_zero() {
            BigFloat::new(prec)
        } else {
            crate::base::numeric::ratio_to_bigfloat(q, prec, rm)
        }
    };
    let re = part(&z.re);
    let im = part(&z.im);
    let bound = Bound {
        re: accuracy::rational_part(&z.re, &re, prec),
        im: accuracy::rational_part(&z.im, &im, prec),
    };
    ((re, im), bound)
}

/// The floating-point value `value ± e` of a node whose exact value is `z`,
/// refined part by part: a part that is exactly 0 becomes an exact 0 (the
/// point of this module: the side of a cut is then decided by convention);
/// a part whose floating-point ball already excludes 0 takes the exact
/// value, rounded once (the same sign, a tighter bound); a nonzero part
/// whose ball contains 0 keeps its floating-point value and bound.  That
/// last part is below the working precision, as a cancellation left it: a
/// function of it is evaluated off the real axis (`polylog(2, 29/(5 + 2i) +
/// (2 + 10⁻³⁰⁰)i − 11/16 − 5)`, whose argument `−11/16 + 10⁻³⁰⁰·i` has an
/// imaginary part that is 0 within its error at 128 bits; before 0.31
/// `evalf` did not take polylog of a complex argument), and the side of a
/// cut that it decides is found at the precision that resolves it.
pub(super) fn refine(
    z: &Gauss,
    value: Complex,
    e: Bound,
    prec: usize,
    rm: RoundingMode,
) -> (Complex, Bound) {
    let ((xre, xim), xb) = to_value(z, prec, rm);
    let pick = |q: &Q, fv: BigFloat, fe: accuracy::ErrExp, xv: BigFloat, xe: accuracy::ErrExp| {
        if q.is_zero() {
            (BigFloat::new(prec), accuracy::EXACT)
        } else if !accuracy::part_contains_zero(&fv, fe) {
            (xv, xe)
        } else {
            (fv, fe)
        }
    };
    let (re, bre) = pick(&z.re, value.0, e.re, xre, xb.re);
    let (im, bim) = pick(&z.im, value.1, e.im, xim, xb.im);
    ((re, im), Bound { re: bre, im: bim })
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;
    use num_traits::Signed;

    fn q(n: i64, d: i64) -> Q {
        Q::new(BigInt::from(n), BigInt::from(d))
    }

    #[test]
    fn eight_over_one_plus_i_plus_4i_minus_5_is_minus_one() {
        let one_plus_i = Gauss {
            re: q(1, 1),
            im: q(1, 1),
        };
        let z = Gauss::real(q(8, 1))
            .mul(&one_plus_i.recip().unwrap())
            .add(&Gauss {
                re: q(-5, 1),
                im: q(4, 1),
            });
        assert_eq!(z, Gauss::real(q(-1, 1)));
        assert!(z.im.is_zero() && z.re.is_negative());
    }

    #[test]
    fn powers_are_bounded() {
        let z = Gauss {
            re: q(3, 7),
            im: q(-2, 5),
        };
        assert!(z.powu(5).is_some());
        let big = Gauss::real(Q::from_integer(BigInt::one() << 20_000u32));
        assert!(big.powu(3).is_none());
    }
}

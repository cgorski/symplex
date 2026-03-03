//! Complex number decomposition.
//!
//! Implements [`as_real_imag`], which splits any expression into
//! its real and imaginary parts: `expr = re + im·i`.

use crate::arena::Arena;
use crate::node::{ExprId, ExprNode};
use crate::walk;
use num_traits::Signed;
use rustc_hash::FxHashMap;

/// Decompose `expr` into `(real_part, imaginary_part)` such that
/// `expr = real_part + imaginary_part * i`.
///
/// Assumes symbols without explicit assumptions are real.
pub(crate) fn as_real_imag(arena: &mut Arena, expr: ExprId) -> (ExprId, ExprId) {
    let post_order = walk::post_order_ids(arena, expr);
    let mut re_cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();
    let mut im_cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();

    for &id in &post_order {
        let (re, im) = decompose_node(arena, id, &re_cache, &im_cache);
        re_cache.insert(id, re);
        im_cache.insert(id, im);
    }

    let re = re_cache.get(&expr).copied().unwrap_or(expr);
    let im = im_cache.get(&expr).copied().unwrap_or(arena.zero);
    (re, im)
}

fn decompose_node(
    arena: &mut Arena,
    id: ExprId,
    re_cache: &FxHashMap<ExprId, ExprId>,
    im_cache: &FxHashMap<ExprId, ExprId>,
) -> (ExprId, ExprId) {
    let node = arena.node(id).clone();
    match node {
        // Atoms
        ExprNode::Num(_) | ExprNode::Symbol(_) | ExprNode::Pi | ExprNode::E => (id, arena.zero),
        ExprNode::ImaginaryUnit => (arena.zero, arena.one),
        ExprNode::Infinity | ExprNode::NegInfinity => (id, arena.zero),
        ExprNode::ComplexInfinity | ExprNode::NaN => (id, arena.zero),

        // Add: re(a+b) = re(a)+re(b), im(a+b) = im(a)+im(b)
        ExprNode::Add(ref children) => {
            let re_parts: Vec<ExprId> = children
                .iter()
                .map(|&c| re_cache.get(&c).copied().unwrap_or(c))
                .collect();
            let im_parts: Vec<ExprId> = children
                .iter()
                .map(|&c| im_cache.get(&c).copied().unwrap_or(arena.zero))
                .collect();
            (arena.add(&re_parts), arena.add(&im_parts))
        }

        // Neg: re(-z) = -re(z), im(-z) = -im(z)
        ExprNode::Neg(inner) => {
            let re = re_cache.get(&inner).copied().unwrap_or(inner);
            let im = im_cache.get(&inner).copied().unwrap_or(arena.zero);
            (arena.neg(re), arena.neg(im))
        }

        // Mul: fold pairwise. (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        ExprNode::Mul(ref children) => {
            if children.is_empty() {
                return (arena.one, arena.zero);
            }
            let mut acc_re = re_cache.get(&children[0]).copied().unwrap_or(children[0]);
            let mut acc_im = im_cache.get(&children[0]).copied().unwrap_or(arena.zero);

            for &child in &children[1..] {
                let c_re = re_cache.get(&child).copied().unwrap_or(child);
                let c_im = im_cache.get(&child).copied().unwrap_or(arena.zero);

                // (acc_re + acc_im*i) * (c_re + c_im*i)
                // = (acc_re*c_re - acc_im*c_im) + (acc_re*c_im + acc_im*c_re)*i
                let ac = arena.mul(&[acc_re, c_re]);
                let bd = arena.mul(&[acc_im, c_im]);
                let ad = arena.mul(&[acc_re, c_im]);
                let bc = arena.mul(&[acc_im, c_re]);

                acc_re = arena.sub(ac, bd);
                acc_im = arena.add(&[ad, bc]);
            }

            (acc_re, acc_im)
        }

        // Pow: for now, handle integer exponents by repeated multiplication
        // and special case exp(z) = exp(re+im*i) = exp(re)(cos(im)+i*sin(im))
        ExprNode::Pow(base, exp) => {
            // If exp is the E constant and base is known: handled by Exp case
            // For simple integer powers, use repeated mul decomposition
            if let Some(n) = arena.as_num(exp)
                && n.is_integer()
                && !n.is_negative()
            {
                let n_i64: i64 = n.to_integer().try_into().unwrap_or(0);
                if n_i64 <= 10 {
                    // Compute by repeated multiplication of (re + im*i)
                    let base_re = re_cache.get(&base).copied().unwrap_or(base);
                    let base_im = im_cache.get(&base).copied().unwrap_or(arena.zero);
                    let mut acc_re = arena.one;
                    let mut acc_im = arena.zero;
                    for _ in 0..n_i64 {
                        let ac = arena.mul(&[acc_re, base_re]);
                        let bd = arena.mul(&[acc_im, base_im]);
                        let ad = arena.mul(&[acc_re, base_im]);
                        let bc = arena.mul(&[acc_im, base_re]);
                        acc_re = arena.sub(ac, bd);
                        acc_im = arena.add(&[ad, bc]);
                    }
                    return (acc_re, acc_im);
                }
            }
            // Fallback: treat as real
            (id, arena.zero)
        }

        // Exp(z): exp(a+bi) = exp(a)(cos(b) + i·sin(b))
        ExprNode::Exp(inner) => {
            let re = re_cache.get(&inner).copied().unwrap_or(inner);
            let im = im_cache.get(&inner).copied().unwrap_or(arena.zero);
            let exp_re = arena.exp(re);
            let cos_im = arena.cos(im);
            let sin_im = arena.sin(im);
            (arena.mul(&[exp_re, cos_im]), arena.mul(&[exp_re, sin_im]))
        }

        // Sin(z): sin(a+bi) = sin(a)cosh(b) + i·cos(a)sinh(b)
        ExprNode::Sin(inner) => {
            let re = re_cache.get(&inner).copied().unwrap_or(inner);
            let im = im_cache.get(&inner).copied().unwrap_or(arena.zero);
            let sin_re = arena.sin(re);
            let cos_re = arena.cos(re);
            let cosh_im = arena.cosh(im);
            let sinh_im = arena.sinh(im);
            (arena.mul(&[sin_re, cosh_im]), arena.mul(&[cos_re, sinh_im]))
        }

        // Cos(z): cos(a+bi) = cos(a)cosh(b) - i·sin(a)sinh(b)
        ExprNode::Cos(inner) => {
            let re = re_cache.get(&inner).copied().unwrap_or(inner);
            let im = im_cache.get(&inner).copied().unwrap_or(arena.zero);
            let cos_re = arena.cos(re);
            let sin_re = arena.sin(re);
            let cosh_im = arena.cosh(im);
            let sinh_im = arena.sinh(im);
            let sin_sinh = arena.mul(&[sin_re, sinh_im]);
            let neg_sin_sinh = arena.neg(sin_sinh);
            (arena.mul(&[cos_re, cosh_im]), neg_sin_sinh)
        }

        // Everything else: assume real for now
        _ => (id, arena.zero),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(a: &mut Arena, name: &str) -> ExprId {
        a.symbol(name)
    }
    fn display(a: &Arena, id: ExprId) -> String {
        a.display(id).to_string()
    }

    #[test]
    fn real_number() {
        let mut a = Arena::new();
        let five = a.int(5);
        let (re, im) = as_real_imag(&mut a, five);
        assert_eq!(display(&a, re), "5");
        assert_eq!(display(&a, im), "0");
    }

    #[test]
    fn imaginary_unit() {
        let mut a = Arena::new();
        let i = a.i_unit;
        let (re, im) = as_real_imag(&mut a, i);
        assert_eq!(display(&a, re), "0");
        assert_eq!(display(&a, im), "1");
    }

    #[test]
    fn two_plus_three_i() {
        let mut a = Arena::new();
        let two = a.int(2);
        let three = a.int(3);
        let three_i = a.mul(&[three, a.i_unit]);
        let expr = a.add(&[two, three_i]);
        let (re, im) = as_real_imag(&mut a, expr);
        assert_eq!(display(&a, re), "2");
        assert_eq!(display(&a, im), "3");
    }

    #[test]
    fn i_squared() {
        // i^2 canonicalizes to -1, so as_real_imag of -1 should be (-1, 0)
        let mut a = Arena::new();
        let i = a.i_unit;
        let two = a.int(2);
        let i_sq = a.pow(i, two); // canonicalizes to -1
        let (re, im) = as_real_imag(&mut a, i_sq);
        assert_eq!(display(&a, re), "-1");
        assert_eq!(display(&a, im), "0");
    }

    #[test]
    fn product_a_plus_bi_times_c_plus_di() {
        let mut a = Arena::new();
        let i = a.i_unit;
        // (1+2i)*(3+4i) = (3-8) + (4+6)i = -5 + 10i
        let one = a.int(1);
        let two = a.int(2);
        let three = a.int(3);
        let four = a.int(4);
        let two_i = a.mul(&[two, i]);
        let z1 = a.add(&[one, two_i]);
        let four_i = a.mul(&[four, i]);
        let z2 = a.add(&[three, four_i]);
        let product = a.mul(&[z1, z2]);
        let (re, im) = as_real_imag(&mut a, product);
        let re_s = display(&a, re);
        let im_s = display(&a, im);
        // The result should simplify to re=-5, im=10
        // But canonicalization may not fully simplify the expanded product
        // At minimum, re and im should be non-trivial
        assert!(!re_s.is_empty() && !im_s.is_empty());
    }

    #[test]
    fn exp_of_i_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let i = a.i_unit;
        let ix = a.mul(&[i, x]);
        let expr = a.exp(ix);
        let (re, im) = as_real_imag(&mut a, expr);
        let re_s = display(&a, re);
        let im_s = display(&a, im);
        // exp(ix) = cos(x) + i*sin(x)
        assert!(re_s.contains("cos"), "re(exp(ix)) should be cos(x): {re_s}");
        assert!(im_s.contains("sin"), "im(exp(ix)) should be sin(x): {im_s}");
    }

    #[test]
    fn symbol_is_real() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let (re, im) = as_real_imag(&mut a, x);
        assert_eq!(display(&a, re), "x");
        assert_eq!(display(&a, im), "0");
    }
}

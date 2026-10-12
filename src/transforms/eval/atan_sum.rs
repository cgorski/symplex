//! Exact zero test of sums of arctangents of rationals and rational
//! multiples of π ([`is_zero`]): `atan(1/2) + atan(1/3) − π/4`, Machin's
//! `4·atan(1/5) − atan(1/239) − π/4`.
//!
//! For `p/q` with `q > 0`, `atan(p/q)` is the argument of the Gaussian
//! integer `q + ip`, so `Θ = Σ nᵢ·atan(rᵢ)` (integers `nᵢ`) is the argument
//! of `z = ∏ (qᵢ + ipᵢ)^{nᵢ}` (a negative `nᵢ` takes the conjugate) up to a
//! multiple of `2π`.  With the coefficients brought to integers by their
//! common denominator `D`, `D·d = Θ + m·π` (`m` an integer) is a multiple
//! of `2π` exactly when `z·(−1)^m` is a positive real number — an exact
//! test in `ℤ[i]` — and then a 16-digit value of `D·d` tells which multiple
//! (`0` when it is below `π`).  When `z·(−1)^m` is not a positive real, `d`
//! is not 0.  No recursion: the terms of `d` are read in one pass.

use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Signed, ToPrimitive, Zero};

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::Q;

/// The most bits of the Gaussian product `z` (the sum of `|nᵢ|·bits(qᵢ +
/// ipᵢ)`), so that the powers stay cheap.
const MAX_PRODUCT_BITS: u64 = 1 << 14;

/// The sum `d` as `Σ cᵢ·atan(rᵢ) + c_π·π` with rational `cᵢ`, `rᵢ`, `c_π`
/// (at least one arctangent, no other term); `None` otherwise.
fn atan_terms(arena: &Arena, d: ExprId) -> Option<(Vec<(Q, Q)>, Q)> {
    let terms: Vec<ExprId> = match arena.node(d) {
        ExprNode::Add(ch) => ch.to_vec(),
        _ => vec![d],
    };
    let mut atans: Vec<(Q, Q)> = Vec::new();
    let mut pi = Q::zero();
    for t in terms {
        let (mut coeff, mut factors) = (Q::one(), Vec::new());
        let mut node = t;
        if let ExprNode::Neg(inner) = *arena.node(node) {
            coeff = -coeff;
            node = inner;
        }
        match arena.node(node) {
            ExprNode::Mul(ch) => {
                for &f in ch {
                    match arena.as_num(f) {
                        Some(q) => coeff *= q,
                        None => factors.push(f),
                    }
                }
            }
            _ => factors.push(node),
        }
        let [f] = factors[..] else {
            return None;
        };
        match *arena.node(f) {
            ExprNode::Pi => pi += coeff,
            ExprNode::Atan(r) => atans.push((coeff, arena.as_num(r)?.clone())),
            _ => return None,
        }
    }
    (!atans.is_empty()).then_some((atans, pi))
}

/// `d` without its arctangent part when that part is exactly 0: for a sum
/// with terms `c·atan(r)` (rational `c`, `r`) and `c·π` beside other
/// terms, whose arctangent terms and multiples of `π` add up to 0 by
/// [`is_zero`], the sum of the other terms (`cos(39π/32) − √(…) + 4·atan(1/5)
/// − atan(1/239) − π/4` is 0 when `cos(39π/32) − √(…)` is); `d` itself
/// otherwise.
pub(super) fn without_zero_atan_part(arena: &mut Arena, d: ExprId) -> ExprId {
    let ExprNode::Add(terms) = arena.node(d).clone() else {
        return d;
    };
    let (mut atan_part, mut rest) = (Vec::new(), Vec::new());
    for &t in &terms {
        if atan_or_pi_term(arena, t) {
            atan_part.push(t);
        } else {
            rest.push(t);
        }
    }
    if atan_part.is_empty() || rest.is_empty() {
        return d;
    }
    let atans = arena.add(&atan_part);
    if is_zero(arena, atans) != Some(true) {
        return d;
    }
    arena.add(&rest)
}

/// Is the term `t` a rational multiple of `π` or of `atan(r)` (`r`
/// rational)?
fn atan_or_pi_term(arena: &Arena, t: ExprId) -> bool {
    let node = match *arena.node(t) {
        ExprNode::Neg(inner) => inner,
        _ => t,
    };
    let factors: Vec<ExprId> = match arena.node(node) {
        ExprNode::Mul(ch) => ch
            .iter()
            .filter(|&&f| arena.as_num(f).is_none())
            .copied()
            .collect(),
        _ => vec![node],
    };
    let mut factors = factors.into_iter();
    match (factors.next(), factors.next()) {
        (Some(f), None) => match *arena.node(f) {
            ExprNode::Pi => true,
            ExprNode::Atan(r) => arena.as_num(r).is_some(),
            _ => false,
        },
        _ => false,
    }
}

/// `(a + ib)·(c + id)`.
fn gauss_mul(x: &(BigInt, BigInt), y: &(BigInt, BigInt)) -> (BigInt, BigInt) {
    (&x.0 * &y.0 - &x.1 * &y.1, &x.0 * &y.1 + &x.1 * &y.0)
}

/// `x^n` for `n ≥ 0`, by squaring.
fn gauss_pow(x: &(BigInt, BigInt), mut n: u64) -> (BigInt, BigInt) {
    let mut out = (BigInt::one(), BigInt::zero());
    let mut base = x.clone();
    while n > 0 {
        if n & 1 == 1 {
            out = gauss_mul(&out, &base);
        }
        n >>= 1;
        if n > 0 {
            base = gauss_mul(&base, &base);
        }
    }
    out
}

/// Is the symbol-free constant `d = Σ cᵢ·atan(rᵢ) + c_π·π` (rational
/// `cᵢ`, `rᵢ`, `c_π`) zero?  `Some(true)` or `Some(false)` exactly (module
/// documentation); `None` when `d` is not of that form, the product grows
/// beyond [`MAX_PRODUCT_BITS`], or the 16-digit value does not tell the
/// multiple of `2π`.
pub(super) fn is_zero(arena: &Arena, d: ExprId) -> Option<bool> {
    let (atans, pi) = atan_terms(arena, d)?;
    let mut den = pi.denom().clone();
    for (c, _) in &atans {
        den = den.lcm(c.denom());
    }
    let den_q = Q::from_integer(den.clone());
    let mut bits: u64 = 0;
    let mut z = (BigInt::one(), BigInt::zero());
    let mut value = (&pi * &den_q).to_f64()? * std::f64::consts::PI;
    let mut scale = value.abs();
    for (c, r) in &atans {
        let n = (c * &den_q).to_integer();
        let (p, q) = (r.numer().clone(), r.denom().clone());
        let n_abs = n.abs().to_u64()?;
        bits = bits.saturating_add(n_abs.saturating_mul(p.bits().max(q.bits()) + 1));
        if bits > MAX_PRODUCT_BITS {
            return None;
        }
        // arg(q + ip) = atan(p/q) for q > 0; the conjugate for n < 0.
        let g = if n.is_negative() { (q, -p) } else { (q, p) };
        z = gauss_mul(&z, &gauss_pow(&g, n_abs));
        let term = n.to_f64()? * r.to_f64()?.atan();
        value += term;
        scale += term.abs();
    }
    let m = (&pi * &den_q).to_integer();
    // D·d ∈ 2πℤ ⟺ z·(−1)^m > 0.
    let sign = if m.is_odd() {
        -z.0.signum()
    } else {
        z.0.signum()
    };
    if !z.1.is_zero() || !sign.is_positive() {
        return Some(false);
    }
    // D·d = 2πk: k from the value, whose rounding errors stay far below π.
    let tolerance = 1e-9 * scale.max(1.0);
    if tolerance >= 0.5 || !value.is_finite() {
        return None;
    }
    let k = (value / (2.0 * std::f64::consts::PI)).round();
    if (value - k * 2.0 * std::f64::consts::PI).abs() > tolerance {
        return None;
    }
    Some(k == 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::context::Context;

    fn decide(ctx: &Context, s: &str) -> Option<bool> {
        let e = ctx.parse(s).expect("parses");
        let inner = e.inner.read();
        is_zero(&inner.arena, e.raw_id())
    }

    #[test]
    fn machin_like_sums_are_zero() {
        let ctx = Context::new();
        for s in [
            "atan(1/2) + atan(1/3) - pi/4",
            "4*atan(1/5) - atan(1/239) - pi/4",
            "atan(1/2) + atan(1/5) + atan(1/8) - pi/4",
            "atan(2) + atan(3) - 3*pi/4",
            "12*atan(1/18) + 8*atan(1/57) - 5*atan(1/239) - pi/4",
            "atan(1/3) + atan(1/4) + atan(2/9) - pi/4",
            "2*atan(2) + atan(4/3) - pi",
        ] {
            assert_eq!(decide(&ctx, s), Some(true), "{s}");
        }
    }

    #[test]
    fn zero_arctangent_part_is_split_off() {
        let ctx = Context::new();
        let d = ctx
            .parse("sqrt(3 + 2*sqrt(2)) - 1 - sqrt(2) + atan(1/2) + atan(1/3) - pi/4")
            .expect("parses");
        let rest = ctx
            .parse("sqrt(3 + 2*sqrt(2)) - 1 - sqrt(2)")
            .expect("parses");
        let kept = ctx
            .parse("sqrt(3 + 2*sqrt(2)) - 1 - sqrt(2) + atan(1/2) + atan(1/3) - pi/5")
            .expect("parses");
        let mut inner = d.inner.write();
        let arena = &mut inner.arena;
        assert_eq!(without_zero_atan_part(arena, d.raw_id()), rest.raw_id());
        assert_eq!(without_zero_atan_part(arena, kept.raw_id()), kept.raw_id());
    }

    #[test]
    fn near_misses_are_nonzero() {
        let ctx = Context::new();
        for s in [
            "atan(1/2) + atan(1/3) - pi/5",
            "atan(1/2) + atan(1/3) + 7*pi/4",
            "atan(2) + atan(3) + pi/4",
            "atan(1/2) - atan(1/3)",
        ] {
            assert_eq!(decide(&ctx, s), Some(false), "{s}");
        }
        // Not a sum of arctangents of rationals and multiples of π.
        assert_eq!(decide(&ctx, "atan(1/2) + atan(1/3) - pi/4 + 1/10"), None);
        assert_eq!(decide(&ctx, "atan(sqrt(2)) - pi/4"), None);
    }
}
